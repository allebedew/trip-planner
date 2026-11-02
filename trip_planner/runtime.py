import asyncio
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from langsmith import traceable

from trip_planner.actions import ClientEnvironment, GuideEnvironment, Message
from trip_planner.agent import AGENT_A_INSTRUCTIONS, AGENT_B_INSTRUCTIONS, Agent
from trip_planner.humans import CLIENT, human_a_opening, human_b

MAX_MESSAGES = 30


def trace_outputs(run: dict) -> dict:
    """The trace already holds every step, so the root keeps only the result."""
    return {key: run[key] for key in ("status", "outcome", "error", "metrics")}


@traceable(name="trip_simulation", process_outputs=trace_outputs)
async def simulate(seed: int) -> dict:
    """Every participant has a queue and handles its messages one by one, as soon as they arrive.
    The run ends when no message is left unhandled."""
    started = time.perf_counter()
    client_env, guide_env = ClientEnvironment(seed, CLIENT), GuideEnvironment(seed)
    agents = {
        "agent_a": Agent("agent_a", client_env.actions(), AGENT_A_INSTRUCTIONS),
        "agent_b": Agent("agent_b", guide_env.actions(), AGENT_B_INSTRUCTIONS),
    }
    envs = {"agent_a": client_env, "agent_b": guide_env}
    # human_a only reads, so it has no queue.
    queues = {name: asyncio.Queue() for name in ("agent_a", "agent_b", "human_b")}
    events = []
    unhandled = 0
    finished = asyncio.Event()
    error = None

    def log(actor: str, event: dict) -> None:
        events.append({"time_s": round(time.perf_counter() - started, 2), "actor": actor, **event})

    def send(message: Message) -> None:
        nonlocal unhandled
        log(message.sender, {"kind": "message", **asdict(message)})
        if sum(e["kind"] == "message" for e in events) > MAX_MESSAGES:
            raise RuntimeError(f"More than {MAX_MESSAGES} messages")
        if message.to in queues:
            unhandled += 1
            queues[message.to].put_nowait(message)

    async def reply(name: str, message: Message) -> list[Message]:
        if name == "human_b":
            return [human_b(message)]
        for event in await agents[name].handle([message]):
            log(name, event)
        return envs[name].take_outbox()

    async def worker(name: str) -> None:
        nonlocal unhandled, error
        while True:
            message = await queues[name].get()
            try:
                for answer in await reply(name, message):
                    send(answer)
            except Exception as failure:
                # An API failure, the step limit or the message limit: the run is recorded as failed.
                error = repr(failure)
                finished.set()
            # Answers are queued before this, so zero means nobody has anything left to do.
            unhandled -= 1
            if unhandled == 0:
                finished.set()

    workers = [asyncio.create_task(worker(name)) for name in queues]
    send(human_a_opening())
    await finished.wait()
    for task in workers:
        task.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

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
        "input_tokens": sum(e["input_tokens"] for e in llm_calls),
        "output_tokens": sum(e["output_tokens"] for e in llm_calls),
        "thinking_tokens": sum(e["thinking_tokens"] for e in llm_calls),
        "cost_usd": round(sum(e["cost_usd"] for e in llm_calls), 6),
    }


def summarize(runs: list[dict]) -> dict:
    outcomes = Counter(r["outcome"] for r in runs)
    return {
        "runs": len(runs),
        "success_rate": sum(r["status"] == "success" for r in runs) / len(runs),
        "outcomes": {str(outcome): count for outcome, count in outcomes.items()},
        "avg_duration_s": round(sum(r["metrics"]["duration_s"] for r in runs) / len(runs), 3),
        "total_faults": sum(r["metrics"]["faults"] for r in runs),
        "total_llm_calls": sum(r["metrics"]["llm_calls"] for r in runs),
        "total_input_tokens": sum(r["metrics"]["input_tokens"] for r in runs),
        "total_output_tokens": sum(r["metrics"]["output_tokens"] for r in runs),
        "total_thinking_tokens": sum(r["metrics"]["thinking_tokens"] for r in runs),
        "total_cost_usd": round(sum(r["metrics"]["cost_usd"] for r in runs), 6),
    }


def save_run(run: dict, folder: Path) -> Path:
    folder.mkdir(exist_ok=True)
    path = folder / f"seed-{run['seed']}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return path
