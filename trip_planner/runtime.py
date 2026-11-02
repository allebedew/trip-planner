import json
import time
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

from langsmith import traceable

from trip_planner.actions import ClientEnvironment, GuideEnvironment
from trip_planner.agent import AGENT_A_INSTRUCTIONS, AGENT_B_INSTRUCTIONS, Agent
from trip_planner.humans import CLIENT, human_a_opening, human_b

MAX_MESSAGES = 30


def trace_outputs(run: dict) -> dict:
    """The trace already holds every step, so the root keeps only the result."""
    return {key: run[key] for key in ("status", "outcome", "error", "metrics")}


@traceable(name="trip_simulation", process_outputs=trace_outputs)
async def simulate(seed: int) -> dict:
    started = time.perf_counter()
    events: list[dict] = []

    def log(actor: str, event: dict) -> None:
        events.append({"time_s": round(time.perf_counter() - started, 2), "actor": actor, **event})

    client_env, guide_env = ClientEnvironment(seed, CLIENT), GuideEnvironment(seed)
    handlers = {
        "agent_a": Agent(client_env, AGENT_A_INSTRUCTIONS, log).handle,
        "agent_b": Agent(guide_env, AGENT_B_INSTRUCTIONS, log).handle,
        "human_b": human_b,
    }

    # Messages are handled one at a time, in the order they were sent. human_a only reads.
    pending, sent, error = deque([human_a_opening()]), 0, None
    try:
        while pending:
            message = pending.popleft()
            log(message.sender, {"kind": "message", **asdict(message)})
            sent += 1
            if sent > MAX_MESSAGES:
                raise RuntimeError(f"More than {MAX_MESSAGES} messages")
            if message.to in handlers:
                pending.extend(await handlers[message.to](message))
    except Exception as failure:
        # An API failure, the step limit or the message limit: the run is recorded as failed.
        error = repr(failure)

    plan = client_env.plan
    return {
        "seed": seed,
        "status": "success" if plan else "failed",
        "outcome": f"{plan['route']['id']} {plan['date']}" if plan else None,
        "error": error,
        "plan": plan,
        "metrics": metrics(events, time.perf_counter() - started),
        "events": events,
    }


def metrics(events: list[dict], duration_s: float) -> dict:
    llm_calls = [e for e in events if e["kind"] == "llm_call"]
    actions = [e for e in events if e["kind"] == "action"]
    return {
        "duration_s": round(duration_s, 3),
        "messages": sum(e["kind"] == "message" for e in events),
        "actions": len(actions),
        "faults": sum("timed out" in e["result"] for e in actions),
        "llm_calls": len(llm_calls),
        **{
            key: round(sum(e[key] for e in llm_calls), 6)
            for key in ("input_tokens", "output_tokens", "thinking_tokens", "cost_usd")
        },
    }


def summarize(runs: list[dict]) -> dict:
    return {
        "runs": len(runs),
        "success_rate": sum(r["status"] == "success" for r in runs) / len(runs),
        "outcomes": dict(Counter(str(r["outcome"]) for r in runs)),
        "avg_duration_s": round(sum(r["metrics"]["duration_s"] for r in runs) / len(runs), 3),
        **{
            f"total_{key}": round(sum(r["metrics"][key] for r in runs), 6)
            for key in ("faults", "llm_calls", "input_tokens", "output_tokens", "thinking_tokens", "cost_usd")
        },
    }


def save_run(run: dict, folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    path = folder / f"seed-{run['seed']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return path
