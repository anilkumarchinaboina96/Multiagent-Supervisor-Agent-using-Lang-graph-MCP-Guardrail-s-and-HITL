import os
import certifi
from dotenv import load_dotenv
import logging
from contextvars import ContextVar
from collections.abc import Callable
import atexit

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from pydantic import SecretStr
from psycopg import Connection
from psycopg_pool import ConnectionPool
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


logger = logging.getLogger(__name__)

_token_callback: ContextVar[Callable[[str], None] | None] = ContextVar(
    "travel_token_callback",
    default=None,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM
# =========================
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=SecretStr(GROQ_API_KEY),
    max_tokens=1000
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _prompt_context(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    marker = "\n[truncated]"
    if len(text) > max_chars:
        return text[: max_chars - len(marker)].rstrip() + marker
    return text


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.bind(max_tokens=1000).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _invoke_user_facing_llm(messages):
    callback = _token_callback.get()
    if callback is None:
        return llm.invoke(messages)

    response = None
    for chunk in llm.stream(messages):
        if isinstance(chunk.content, str) and chunk.content:
            callback(chunk.content)
        response = chunk if response is None else response + chunk

    if response is None:
        raise RuntimeError("The language model returned no response chunks.")

    return response


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        logger.warning(
            "Guardrail fallback used due to parsing/model error: %s",
            exc,
        )
        allowed = True
        guardrail_reason = "Guardrail validation fallback allowed the request."

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        logger.warning(
            "Supervisor fallback used due to parsing/model error: %s",
            exc,
        )
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - original behavior kept
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    logger.info("Running flight agent")
    query = state["user_query"]

    try:
        async def load_flight_reference_data():
            return await asyncio.gather(
                aviation_mcp_call("list_airports"),
                aviation_mcp_call("list_airlines"),
            )

        airports, airlines = asyncio.run(load_flight_reference_data())

        logger.debug("Flight agent airports payload: %s", airports)
        logger.debug("Flight agent airlines payload: %s", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=_prompt_context(airports, 1000),
            airline_data=_prompt_context(airlines, 1000),
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        logger.warning("Flight agent fallback due to tool/model error: %s", exc)
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query = (
        f"Best hotels for "
        f"{state['user_query']}"
    )

    try:
        hotel_results = asyncio.run(
            tavily_mcp_search(query)
        )

    except Exception as exc:
        logger.warning(
            "Hotel agent MCP error: %s: %s",
            type(exc).__name__,
            exc,
        )

        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    hotel_results = _prompt_context(hotel_results, 1500)

    return {
        "hotel_results": hotel_results,
        "messages": [
            AIMessage(
                content="Hotel information processed."
            )
        ],
        "llm_calls": (
            state.get("llm_calls", 0) + 1
        ),
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    constraints = state.get("trip_constraints")
    city = ""
    if isinstance(constraints, dict):
        city = str(constraints.get("destination") or "").strip()

    if not city:
        try:
            city = extract_destination(
                state.get("user_query", "")
            ).strip()
        except Exception:
            logger.exception(
                "Weather destination extraction failed; skipping live weather."
            )

    if not city:
        weather_results = (
            "Live weather information is temporarily unavailable because "
            "the destination could not be determined. Give general seasonal "
            "guidance and advise the traveler to verify the forecast before "
            "departure."
        )
    else:
        try:
            async def load_weather_data():
                return await asyncio.gather(
                    weather_mcp_search(city),
                    forecast_mcp_search(city),
                )

            weather_data, forecast_data = asyncio.run(load_weather_data())

            weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""

        except Exception:
            logger.exception(
                "Weather agent MCP error for destination %s; using guidance fallback.",
                city,
            )

            weather_results = (
                f"Live weather information for {city} "
                "is temporarily unavailable. Give general "
                "seasonal guidance and advise the traveler "
                "to verify the forecast before departure."
            )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


def _run_research_agent(agent_name, agent, state):
    started_at = perf_counter()
    try:
        return agent(state)
    finally:
        logger.info(
            "Research agent completed agent=%s duration_ms=%.2f",
            agent_name,
            (perf_counter() - started_at) * 1000,
        )


def parallel_research_agent(state: TravelState):
    selected = set(_selected_agents(state))
    agents = [
        ("flight_agent", flight_agent),
        ("hotel_agent", hotel_agent),
        ("weather_agent", weather_agent),
    ]
    selected_agents = [
        (name, agent)
        for name, agent in agents
        if name in selected
    ]
    if not selected_agents:
        return {}

    base_llm_calls = state.get("llm_calls", 0)
    llm_call_delta = 0
    updates: dict[str, Any] = {}
    messages = []
    started_at = perf_counter()

    with ThreadPoolExecutor(
        max_workers=len(selected_agents),
        thread_name_prefix="travel-agent",
    ) as executor:
        futures = [
            (
                name,
                executor.submit(
                    _run_research_agent,
                    name,
                    agent,
                    state,
                ),
            )
            for name, agent in selected_agents
        ]

        for name, future in futures:
            result = future.result()
            messages.extend(result.get("messages", []))
            if "llm_calls" in result:
                llm_call_delta += max(
                    0,
                    result["llm_calls"] - base_llm_calls,
                )
            updates.update(
                {
                    key: value
                    for key, value in result.items()
                    if key not in {"messages", "llm_calls"}
                }
            )

    if messages:
        updates["messages"] = messages
    if llm_call_delta:
        updates["llm_calls"] = base_llm_calls + llm_call_delta

    logger.info(
        "Parallel research completed agents=%s duration_ms=%.2f",
        ",".join(name for name, _ in selected_agents),
        (perf_counter() - started_at) * 1000,
    )
    return updates


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{_prompt_context(state.get('trip_constraints', {}), 800)}

Flight Results:
{_prompt_context(state.get('flight_results', ''))}

Hotel Results:
{_prompt_context(state.get('hotel_results', ''))}

Weather Results:
{_prompt_context(state.get('weather_results', ''))}

Return concise Markdown with these sections:
## Budget snapshot
Summarize the estimated total and per-person range when traveler count and cost
data are available.
## Cost breakdown
Use a table with columns Category, Estimated cost (INR), and Booking notes.
Include only categories supported by the request or available results. Use low–
high ranges when prices are uncertain.
## Budget fit
Explain whether the costs fit the user's stated budget. Do not invent a budget
percentage; calculate one only when a numeric total budget is provided.
## Ways to save
List a few practical, destination-specific suggestions.

If exact live prices are unavailable, clearly label estimates as approximate.
Do not imply that anything has been booked or that a quoted price is live.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{_prompt_context(state.get('trip_constraints', {}), 800)}

Flight Results:
{_prompt_context(state.get('flight_results', ''))}

Hotel Results:
{_prompt_context(state.get('hotel_results', ''))}

Weather Results:
{_prompt_context(state.get('weather_results', ''))}

Budget Results:
{_prompt_context(state.get('budget_results', ''))}

Return concise Markdown with these sections:
## Trip Summary
Give the route, dates or duration, traveler count, and travel style when known.
## Transport
Summarize flight or ground transport options and clearly label unavailable live
prices.
## Stay
Give suitable accommodation options and booking considerations.
## Weather and Packing
Include useful, destination-specific advice when weather results are available.
## Day-by-Day Itinerary
Use a ### Day 1 — ... heading for each day, followed by short time-ordered
bullets.
## Estimated Budget
Use a table with Category, Estimated cost (INR), and Booking notes columns.
Show a total low–high estimate and per-person cost when the inputs allow it.
Do not invent booking availability or a percentage of budget.
## Practical Tips
List concise, actionable recommendations.

Keep the draft practical, budget-aware, and ready for human review. Clearly
label estimates when exact live prices are unavailable.
"""

    response = _invoke_user_facing_llm(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{_prompt_context(state.get('trip_constraints', {}), 800)}

Flights:
{_prompt_context(state.get('flight_results', ''))}

Hotels:
{_prompt_context(state.get('hotel_results', ''))}

Weather:
{_prompt_context(state.get('weather_results', ''))}

Budget Analysis:
{_prompt_context(state.get('budget_results', ''))}

Draft Itinerary:
{_prompt_context(state.get('itinerary', ''))}

Write clean Markdown using these sections in order:
## Trip Summary
Give the route, dates or duration, traveler count, and requested style.
## Flight Information
Summarize practical options. Say when live fares are unavailable; never present
an estimate as a confirmed fare.
## Stay
List suitable stays with location, approximate price range if supported, and
one-line booking considerations.
## Weather and Local Transport
Include useful weather and transport advice based on the available results.
## Day-by-Day Itinerary
Use a ### Day 1 — ... heading for each day and concise, time-ordered bullets.
## Estimated Budget
Use a Markdown table with columns Category, Estimated cost (INR), and Booking
notes. Show a total low–high estimate and a per-person amount if traveler count
and estimates are available. Clearly label approximate values.
## Final Recommendations
List a few actionable next steps.

Use readable headings, short paragraphs, and bullets; avoid dense prose. Do not
invent current prices, availability, bookings, or a percentage of the total
budget. Only show a budget percentage when a numeric user budget is available,
and calculate it from that budget. Incorporate human feedback when revising.
"""

    response = _invoke_user_facing_llm(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "research_agents": "research_agents",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = set(_selected_agents(state))
    if selected.intersection(
        {"flight_agent", "hotel_agent", "weather_agent"}
    ):
        return "research_agents"
    if "budget_agent" in selected:
        return "budget_agent"
    return "itinerary_agent"


def route_after_research(state: TravelState) -> str:
    return (
        "budget_agent"
        if "budget_agent" in _selected_agents(state)
        else "itinerary_agent"
    )


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("research_agents", parallel_research_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges("research_agents", route_after_research, ROUTE_MAP)
graph.add_edge("budget_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
DATABASE_POOL_MIN_SIZE = int(os.getenv("DATABASE_POOL_MIN_SIZE", "1"))
DATABASE_POOL_MAX_SIZE = int(os.getenv("DATABASE_POOL_MAX_SIZE", "10"))
if DATABASE_POOL_MIN_SIZE < 1:
    raise ValueError("DATABASE_POOL_MIN_SIZE must be at least 1.")
if DATABASE_POOL_MAX_SIZE < DATABASE_POOL_MIN_SIZE:
    raise ValueError(
        "DATABASE_POOL_MAX_SIZE must be greater than or equal to "
        "DATABASE_POOL_MIN_SIZE."
    )

database_pool: ConnectionPool[Connection[dict[str, Any]]] = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=DATABASE_POOL_MIN_SIZE,
    max_size=DATABASE_POOL_MAX_SIZE,
    kwargs={
        "autocommit": True,
        "row_factory": dict_row,
    },
)
checkpointer = PostgresSaver(database_pool)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


def close_database_pool():
    database_pool.close()


atexit.register(close_database_pool)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(
    user_input: str,
    thread_id: str | None = None,
    token_callback: Callable[[str], None] | None = None,
):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    callback_token = _token_callback.set(token_callback)
    try:
        result = travel_graph.invoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "user_query": user_input,
                "guardrail_allowed": True,
                "guardrail_reason": "",
                "selected_agents": [],
                "trip_constraints": _empty_constraints(),
                "supervisor_reasoning": "",
                "flight_results": "",
                "hotel_results": "",
                "weather_results": "",
                "budget_results": "",
                "itinerary": "",
                "approval_request": "",
                "approved": False,
                "human_feedback": "",
                "final_response": "",
                "llm_calls": 0,
            },
            config=config,
        )
    finally:
        _token_callback.reset(callback_token)

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
    token_callback: Callable[[str], None] | None = None,
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    callback_token = _token_callback.set(token_callback)
    try:
        result = travel_graph.invoke(
            Command(
                resume={
                    "approved": approved,
                    "feedback": feedback.strip(),
                }
            ),
            config=config,
        )
    finally:
        _token_callback.reset(callback_token)

    return _serialize_result(result, thread_id)
