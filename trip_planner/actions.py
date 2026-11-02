import json
import random
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
FAULT_RATE = 0.3

# Who may write to whom. Nothing else is allowed.
LINKS = {
    "agent_a": {"human_a", "agent_b"},
    "agent_b": {"agent_a", "human_b"},
}

KNEE_MAX_DESCENT_M = 600
ALLOWED_DIFFICULTY = {
    "low": {"low"},
    "medium": {"low", "medium"},
    "high": {"low", "medium", "high"},
}
RAINY_PROBABILITY = 40
RAINY_MM = 1.0
COLD_MIN_C = 5.0


@dataclass
class Message:
    sender: str
    to: str
    text: str


def load_json(name: str) -> dict:
    return json.loads((DATA_DIR / f"{name}.json").read_text(encoding="utf-8"))


def flaky(action):
    """Simulates an unreliable service: the call fails at random and the LLM has to repeat it."""

    @wraps(action)
    def wrapper(self, *args, **kwargs):
        if self.rng.random() < FAULT_RATE:
            return {"error": f"{action.__name__} timed out. It is safe to call it again."}
        return action(self, *args, **kwargs)

    return wrapper


def booking_id_for(route_id: str, date: str) -> str:
    return f"BK-{route_id}-{date}"


def eligible_routes(routes: list[dict], client: dict) -> list[dict]:
    """Hard limits: the LLM never sees a route outside them."""
    return [
        r
        for r in routes
        if r["difficulty"] in ALLOWED_DIFFICULTY[client["fitness"]]
        and r["duration_min"] <= client["max_duration_min"]
        and (not client["knee_issue"] or r["descent_m"] <= KNEE_MAX_DESCENT_M)
    ]


def check_gear(items: list[dict], day: dict, client: dict) -> dict[str, list[str]]:
    conditions = {"always"}
    if day["rain_probability"] >= RAINY_PROBABILITY or day["rain_mm"] >= RAINY_MM:
        conditions.add("rain")
    if day["temp_min_c"] <= COLD_MIN_C:
        conditions.add("cold")
    if client["knee_issue"]:
        conditions.add("knee_issue")

    needed = sorted(i["id"] for i in items if conditions & set(i["needed_when"]))
    return {
        "have": [g for g in needed if g in client["gear"]],
        "missing": [g for g in needed if g not in client["gear"]],
    }


class Environment:
    """State one agent can reach. The two environments share nothing."""

    def __init__(self, name: str, seed: int):
        self.name = name
        self.rng = random.Random(f"{seed}:{name}")
        self.outbox: list[Message] = []

    def send_message(self, to: str, text: str) -> dict:
        """Send a message. A reply, if any, arrives later as a new message."""
        if to not in LINKS[self.name]:
            return {"error": f"{self.name} cannot write to {to}. Allowed: {sorted(LINKS[self.name])}"}
        self.outbox.append(Message(sender=self.name, to=to, text=text))
        return {"sent_to": to}

    def take_outbox(self) -> list[Message]:
        messages, self.outbox = self.outbox, []
        return messages


class ClientEnvironment(Environment):
    """Agent A: the client's profile and public data. It cannot see closures or the guide's calendar."""

    def __init__(self, seed: int, client: dict):
        super().__init__("agent_a", seed)
        self.client = client
        self.routes = load_json("routes")["routes"]
        self.gear = load_json("gear")["items"]
        self.days = {
            d["date"]: d
            for d in load_json("weather")["days"]
            if client["earliest_date"] <= d["date"] <= client["latest_date"]
        }
        self.plan: dict | None = None

    def actions(self) -> list:
        return [self.send_message, self.get_trip_options, self.finish]

    @flaky
    def get_trip_options(self) -> dict:
        """Get the routes that fit the client and the forecast for the client's dates. May fail."""
        routes = eligible_routes(self.routes, self.client)
        return {
            "routes": [{k: v for k, v in r.items() if k != "url"} for r in routes],
            "forecast": list(self.days.values()),
        }

    def finish(self, route_id: str, date: str, booking_id: str) -> dict:
        """Check the trip against the data, work out the gear, store the plan and send it to the client."""
        allowed = {r["id"]: r for r in eligible_routes(self.routes, self.client)}
        if route_id not in allowed:
            return {"error": f"Route {route_id} does not fit the client"}
        if date not in self.days:
            return {"error": f"No forecast for {date}"}
        if booking_id != booking_id_for(route_id, date):
            return {"error": f"Booking {booking_id} does not match {route_id} on {date}"}

        gear = check_gear(self.gear, self.days[date], self.client)
        self.plan = {
            "client_name": self.client["name"],
            "route": allowed[route_id],
            "date": date,
            "weather": self.days[date],
            "gear_to_bring": gear["have"],
            "gear_missing": gear["missing"],
            "booking_id": booking_id,
        }
        text = (
            f"Your trip is booked: {allowed[route_id]['name']} on {date}, booking {booking_id}. "
            f"Bring: {', '.join(gear['have'])}. You still need: {', '.join(gear['missing']) or 'nothing'}."
        )
        return self.send_message("human_a", text)


class GuideEnvironment(Environment):
    """Agent B: the company's private calendar and the bookings it writes."""

    def __init__(self, seed: int):
        super().__init__("agent_b", seed)
        guide = load_json("guide")
        self.free_dates = set(guide["available_dates"])
        self.closed = guide["closed_routes"]
        self.bookings: dict[str, dict] = {}

    def actions(self) -> list:
        return [self.send_message, self.check_trip, self.book_trip]

    def refusal(self, route_id: str, date: str) -> str | None:
        if route_id in self.closed:
            return f"Route {route_id} is closed on every date: {self.closed[route_id]}"
        if date not in self.free_dates:
            return f"The guide is not free on {date}"
        return None

    @flaky
    def check_trip(self, route_id: str, date: str) -> dict:
        """Check that a route is open and the guide is free on a date. May fail."""
        reason = self.refusal(route_id, date)
        return {"available": False, "reason": reason} if reason else {"available": True}

    @flaky
    def book_trip(self, client_name: str, route_id: str, date: str) -> dict:
        """Book the trip. May fail. Safe to repeat: the same trip returns the same booking."""
        booking_id = booking_id_for(route_id, date)
        if booking_id in self.bookings:
            return self.bookings[booking_id]
        reason = self.refusal(route_id, date)
        if reason:
            return {"error": reason}
        self.free_dates.remove(date)
        self.bookings[booking_id] = {
            "booking_id": booking_id,
            "client_name": client_name,
            "route_id": route_id,
            "date": date,
        }
        return self.bookings[booking_id]
