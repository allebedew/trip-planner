import asyncio
import json
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from langsmith import trace, traceable

from trip_planner.actions import ClientEnvironment, GuideEnvironment
from trip_planner.agent import AGENT_A_INSTRUCTIONS, AGENT_B_INSTRUCTIONS, Agent
from trip_planner.humans import CLIENT, human_a_opening, human_b

MAX_TICKS = 30


def trace_outputs(run: dict) -> dict:
    """The trace already holds every step, so the root keeps only the result."""
    return {key: run[key] for key in ("status", "outcome", "error", "metrics")}


@traceable(name="trip_simulation", process_outputs=trace_outputs)
async def simulate(seed: int) -> dict:
    """Runs ticks until nobody has anything to say. A message sent on one tick arrives on the next,
    so the order of events does not depend on how fast each LLM answers."""
    started = time.perf_counter()
    client_env, guide_env = ClientEnvironment(seed, CLIENT), GuideEnvironment(seed)
    agents = {
        "agent_a": Agent("agent_a", client_env.actions(), AGENT_A_INSTRUCTIONS),
        "agent_b": Agent("agent_b", guide_env.actions(), AGENT_B_INSTRUCTIONS),
    }
    mailbox = [human_a_opening()]
    events = []
    error = None

    try:
        for tick in range(MAX_TICKS):
            async with trace(f"tick {tick}", inputs={"messages": [asdict(m) for m in mailbox]}) as span:
                inboxes = defaultdict(list)
                for message in mailbox:
                    inboxes[message.to].append(message)
                    events.append({"tick": tick, "actor": message.sender, "kind": "message", **asdict(message)})

                mailbox = [human_b(message) for message in inboxes["human_b"]]

                active = [name for name in agents if inboxes[name]]
                results = await asyncio.gather(*(agents[name].handle(inboxes[name]) for name in active))
                for name, agent_events in zip(active, results):
                    events += [{"tick": tick, "actor": name, **event} for event in agent_events]

                mailbox += client_env.take_outbox() + guide_env.take_outbox()
                span.end(outputs={"messages": [asdict(m) for m in mailbox]})

            if not mailbox:
                break
    except Exception as failure:
        # An API failure or the step limit: the run is recorded as failed.
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
        "ticks": events[-1]["tick"] + 1,
        "duration_s": round(duration_s, 3),
        "messages": sum(e["kind"] == "message" for e in events),
        "actions": len(actions),
        "faults": sum("timed out" in e["result"] for e in actions),
        "llm_calls": len(llm_calls),
        "input_tokens": sum(e["input_tokens"] for e in llm_calls),
        "output_tokens": sum(e["output_tokens"] for e in llm_calls),
    }


def summarize(runs: list[dict]) -> dict:
    outcomes = Counter(r["outcome"] for r in runs)
    return {
        "runs": len(runs),
        "success_rate": sum(r["status"] == "success" for r in runs) / len(runs),
        "outcomes": {str(outcome): count for outcome, count in outcomes.items()},
        "avg_ticks": sum(r["metrics"]["ticks"] for r in runs) / len(runs),
        "total_faults": sum(r["metrics"]["faults"] for r in runs),
        "total_llm_calls": sum(r["metrics"]["llm_calls"] for r in runs),
        "total_input_tokens": sum(r["metrics"]["input_tokens"] for r in runs),
        "total_output_tokens": sum(r["metrics"]["output_tokens"] for r in runs),
    }


def save_run(run: dict, folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    path = folder / f"seed-{run['seed']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return path
