from pathlib import Path
from collections import defaultdict
import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import threading
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend import (
    close_database_pool,
    resume_travel_agent,
    run_travel_agent,
)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        ),
    )

logger = logging.getLogger("tripmate.api")

BASE_DIR = Path(__file__).resolve().parent
MAX_ACTIVE_PLANS = int(os.getenv("MAX_ACTIVE_PLANS", "4"))
if MAX_ACTIVE_PLANS < 1:
    raise ValueError("MAX_ACTIVE_PLANS must be at least 1.")

_plan_slots = threading.BoundedSemaphore(MAX_ACTIVE_PLANS)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        yield
    finally:
        await run_in_threadpool(close_database_pool)


app = FastAPI(
    title="TripMate AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


REQUEST_METRICS: dict[str, object] = {
    "requests_total": 0,
    "errors_total": 0,
    "by_path": defaultdict(
        lambda: {
            "count": 0,
            "errors": 0,
            "total_duration_ms": 0.0,
        }
    ),
}


class TravelRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: str | None = Field(default=None, max_length=128)


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=128)
    approved: bool
    feedback: str = Field(default="", max_length=4000)


def _busy_response():
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "5"},
        content={
            "success": False,
            "error": "The planner is busy. Please retry shortly.",
        },
    )


def _sse_event(event: str, data) -> str:
    payload = json.dumps(
        jsonable_encoder(data),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return f"event: {event}\ndata: {payload}\n\n"


def _stream_planner_response(request: Request, planner, **planner_args):
    if not _plan_slots.acquire(blocking=False):
        return _busy_response()

    request_id = getattr(request.state, "request_id", "unknown")
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    disconnected = threading.Event()

    def enqueue(event: str, payload=None):
        if disconnected.is_set():
            return
        try:
            loop.call_soon_threadsafe(events.put_nowait, (event, payload))
        except RuntimeError:
            disconnected.set()

    def worker():
        try:
            result = planner(
                **planner_args,
                token_callback=lambda text: enqueue(
                    "token",
                    {"text": text},
                ),
            )
            enqueue("result", {"success": True, **result})
        except Exception:
            logger.exception(
                "Streaming planner failed request_id=%s",
                request_id,
            )
            enqueue(
                "error",
                {
                    "success": False,
                    "error": "Internal server error.",
                    "request_id": request_id,
                },
            )
        finally:
            _plan_slots.release()
            enqueue("done")

    try:
        loop.run_in_executor(None, worker)
    except RuntimeError:
        _plan_slots.release()
        logger.exception(
            "Could not start streaming planner request_id=%s",
            request_id,
        )
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": "The planner is temporarily unavailable.",
                "request_id": request_id,
            },
        )

    async def stream_events():
        try:
            while True:
                try:
                    event, payload = await asyncio.wait_for(
                        events.get(),
                        timeout=15,
                    )
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue

                if event == "done":
                    break
                yield _sse_event(event, payload)
        finally:
            disconnected.set()

    return StreamingResponse(
        stream_events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


def _record_request_metrics(path: str, status_code: int, duration_ms: float):
    path_metrics = REQUEST_METRICS["by_path"][path]
    path_metrics["count"] += 1
    path_metrics["total_duration_ms"] += duration_ms
    REQUEST_METRICS["requests_total"] += 1

    if status_code >= 500:
        path_metrics["errors"] += 1
        REQUEST_METRICS["errors_total"] += 1


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id

    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        _record_request_metrics(request.url.path, 500, duration_ms)

        logger.exception(
            "Unhandled request error request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )

        response = JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error.",
                "request_id": request_id,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    body_iterator = getattr(response, "body_iterator", None)
    if body_iterator is not None:

        async def measured_body_iterator():
            try:
                async for chunk in body_iterator:
                    yield chunk
            finally:
                duration_ms = (time.perf_counter() - start) * 1000
                _record_request_metrics(
                    request.url.path,
                    response.status_code,
                    duration_ms,
                )
                logger.info(
                    "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
                    request_id,
                    request.method,
                    request.url.path,
                    response.status_code,
                    duration_ms,
                )

        setattr(response, "body_iterator", measured_body_iterator())
        response.headers["X-Request-ID"] = request_id
        return response

    duration_ms = (time.perf_counter() - start) * 1000
    _record_request_metrics(
        request.url.path,
        response.status_code,
        duration_ms,
    )

    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.2f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/travel")
async def travel_planner(request: Request, request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        if not _plan_slots.acquire(blocking=False):
            return _busy_response()

        try:
            result = await run_in_threadpool(
                run_travel_agent,
                user_input=user_message,
                thread_id=request_data.thread_id,
            )
        finally:
            _plan_slots.release()

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            "Travel planner failed request_id=%s",
            request_id,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error.",
                "request_id": request_id,
            },
        )


@app.post("/api/travel/stream")
async def travel_planner_stream(
    request: Request,
    request_data: TravelRequest,
):
    user_message = request_data.message.strip()
    if not user_message:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Message cannot be empty.",
            },
        )

    return _stream_planner_response(
        request,
        run_travel_agent,
        user_input=user_message,
        thread_id=request_data.thread_id,
    )


@app.post("/api/travel/approve")
async def approve_travel_plan(request: Request, request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        if not _plan_slots.acquire(blocking=False):
            return _busy_response()

        try:
            result = await run_in_threadpool(
                resume_travel_agent,
                thread_id=request_data.thread_id,
                approved=request_data.approved,
                feedback=request_data.feedback,
            )
        finally:
            _plan_slots.release()

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        request_id = getattr(request.state, "request_id", "unknown")
        logger.exception(
            "Approval flow failed request_id=%s",
            request_id,
        )

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error.",
                "request_id": request_id,
            },
        )


@app.post("/api/travel/approve/stream")
async def approve_travel_plan_stream(
    request: Request,
    request_data: ApprovalRequest,
):
    if not request_data.approved and not request_data.feedback.strip():
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Please provide revision feedback when rejecting the draft.",
            },
        )

    return _stream_planner_response(
        request,
        resume_travel_agent,
        thread_id=request_data.thread_id,
        approved=request_data.approved,
        feedback=request_data.feedback,
    )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "TripMate AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
        ],
    }


@app.get("/metrics")
async def metrics():
    per_path: dict[str, dict[str, float | int]] = {}

    for path, values in REQUEST_METRICS["by_path"].items():
        count = values["count"]
        total_duration_ms = values["total_duration_ms"]
        average_ms = (total_duration_ms / count) if count else 0.0
        per_path[path] = {
            "count": count,
            "errors": values["errors"],
            "avg_duration_ms": round(average_ms, 2),
        }

    return {
        "requests_total": REQUEST_METRICS["requests_total"],
        "errors_total": REQUEST_METRICS["errors_total"],
        "paths": per_path,
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
