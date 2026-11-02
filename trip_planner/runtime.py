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


class Mailroom:
    """Every participant has a queue and handles its messages one by one, as soon as they arrive.
    The run ends when no message is left unhandled."""

    def __init__(self, agents: list[Agent]):
        self.agents = {agent.name: agent for agent in agents}
        # human_a only reads, so it has no queue.
        self.queues = {name: asyncio.Queue() for name in [*self.agents, "human_b"]}
        self.unhandled = 0
        self.finished = asyncio.Event()
        self.started = time.perf_counter()
        self.events: list[dict] = []
        self.error: str | None = None

    async def run(self, first: Message) -> None:
        workers = [asyncio.create_task(self.worker(name)) for name in self.queues]
        self.send(first)
        await self.finished.wait()
        for task in workers:
            task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    def send(self, message: Message) -> None:
        self.log(message.sender, {"kind": "message", **asdict(message)})
        if sum(e["kind"] == "message" for e in self.events) > MAX_MESSAGES:
            raise RuntimeError(f"More than {MAX_MESSAGES} messages")
        if message.to in self.queues:
            self.unhandled += 1
            self.queues[message.to].put_nowait(message)

    async def worker(self, name: str) -> None:
        while True:
            message = await self.queues[name].get()
            try:
                for answer in await self.reply(name, message):
                    self.send(answer)
            except Exception as failure:
                # An API failure, the step limit or the message limit: the run is recorded as failed.
                self.error = repr(failure)
                self.finished.set()
            # Answers are queued before this, so zero means nobody has anything left to do.
            self.unhandled -= 1
            if self.unhandled == 0:
                self.finished.set()

    async def reply(self, name: str, message: Message) -> list[Message]:
        if name == "human_b":
            return [human_b(message)]
        agent = self.agents[name]
        for event in await agent.handle(message):
            self.log(name, event)
        return agent.env.take_outbox()

    def log(self, actor: str, event: dict) -> None:
        self.events.append({"time_s": round(time.perf_counter() - self.started, 2), "actor": actor, **event})


@traceable(name="trip_simulation", process_outputs=trace_outputs)
async def simulate(seed: int) -> dict:
    started = time.perf_counter()
    client_env, guide_env = ClientEnvironment(seed, CLIENT), GuideEnvironment(seed)
    mailroom = Mailroom([Agent(client_env, AGENT_A_INSTRUCTIONS), Agent(guide_env, AGENT_B_INSTRUCTIONS)])
    await mailroom.run(human_a_opening())

    plan = client_env.plan
    return {
        "seed": seed,
        "status": "success" if plan else "failed",
        "outcome": f"{plan['route']['id']} {plan['date']}" if plan else None,
        "error": mailroom.error,
        "plan": plan,
        "metrics": metrics(mailroom.events, time.perf_counter() - started),
        "events": mailroom.events,
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
