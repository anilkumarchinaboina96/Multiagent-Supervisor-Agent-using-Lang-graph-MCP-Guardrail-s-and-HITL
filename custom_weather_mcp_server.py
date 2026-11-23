import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

mcp = FastMCP("Weather MCP Server")

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

WEATHER_API_BASE_URL = "https://api.openweathermap.org/data/2.5"
REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("OPENWEATHER_TIMEOUT_SECONDS", "20")
)
REQUEST_MAX_RETRIES = int(
    os.getenv("OPENWEATHER_MAX_RETRIES", "2")
)


def _build_session() -> requests.Session:
    retry = Retry(
        total=REQUEST_MAX_RETRIES,
        connect=REQUEST_MAX_RETRIES,
        read=REQUEST_MAX_RETRIES,
        status=REQUEST_MAX_RETRIES,
        backoff_factor=0.5,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


HTTP_SESSION = _build_session()


def _get_api_key() -> str:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing "
            "from the project .env file."
        )

    return OPENWEATHER_API_KEY


def _validate_city(city: str) -> str:
    normalized = city.strip()

    if not normalized:
        raise ValueError("city cannot be empty")

    if len(normalized) > 120:
        raise ValueError("city is too long; use a concise city name")

    return normalized


def _safe_response_snippet(response: requests.Response | None) -> str:
    if response is None:
        return ""

    body = response.text[:500].replace("\n", " ").strip()
    if not body:
        return ""
    return f" Response: {body}"


def _request_json(
    url: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    try:
        response = HTTP_SESSION.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError(
                "OpenWeather returned non-JSON data."
            ) from exc

        if not isinstance(payload, dict):
            raise RuntimeError(
                "OpenWeather response shape is invalid."
            )

        return payload

    except requests.Timeout as exc:
        raise RuntimeError(
            "OpenWeather request timed out. "
            "Please retry."
        ) from exc

    except requests.ConnectionError as exc:
        raise RuntimeError(
            "OpenWeather service is unreachable. "
            "Please check network connectivity."
        ) from exc

    except requests.HTTPError as exc:
        details = _safe_response_snippet(
            getattr(exc, "response", None)
        )

        raise RuntimeError(
            f"OpenWeather request failed: {exc}.{details}"
        ) from exc

    except requests.RequestException as exc:
        details = _safe_response_snippet(
            getattr(exc, "response", None)
        )

        raise RuntimeError(
            f"OpenWeather request failed: {exc}.{details}"
        ) from exc


@mcp.tool()
def get_current_weather(
    city: str,
) -> dict[str, Any]:
    """Return the current weather for a city."""

    city = _validate_city(city)

    data = _request_json(
        f"{WEATHER_API_BASE_URL}/weather",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    try:
        weather_items = data["weather"]
        first_weather = weather_items[0]
        main = data["main"]
        wind = data["wind"]
        city_name = data["name"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "OpenWeather current weather payload is incomplete."
        ) from exc

    return {
        "city": city_name,
        "temperature_c": main["temp"],
        "feels_like_c": main["feels_like"],
        "humidity": main["humidity"],
        "condition": first_weather["description"],
        "wind_speed": wind["speed"],
    }


@mcp.tool()
def get_forecast(
    city: str,
) -> dict[str, Any]:
    """
    Return the first five three-hour
    forecast entries for a city.
    """

    city = _validate_city(city)

    data = _request_json(
        f"{WEATHER_API_BASE_URL}/forecast",
        {
            "q": city,
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    entries = data.get("list", [])
    if not isinstance(entries, list):
        raise RuntimeError(
            "OpenWeather forecast payload is invalid."
        )

    forecast = [
        {
            "datetime": item["dt_txt"],
            "temperature_c": item["main"]["temp"],
            "condition": item["weather"][0]["description"],
        }
        for item in entries[:5]
    ]

    return {
        "city": data.get(
            "city",
            {},
        ).get(
            "name",
            city,
        ),
        "forecast": forecast,
    }


if __name__ == "__main__":
    # mcp_client.py launches this as a stdio subprocess.
    mcp.run(
        transport="stdio",
    )