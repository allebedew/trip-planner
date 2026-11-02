import argparse
import asyncio
import json
from pathlib import Path

from trip_planner.runtime import save_run, simulate, summarize

RUNS_DIR = Path("runs")


def describe(event: dict) -> str:
    if event["kind"] == "message":
        return f"{event['actor']} -> {event['to']}: {event['text']}"
    if event["kind"] == "llm_call":
        calls = ", ".join(call["action"] for call in event["calls"]) or "wait"
        return f"{event['actor']}  LLM -> {calls}"
    return f"{event['actor']}  {event['action']}: {event['result'][:100]}"


def main() -> None:
    parser = argparse.ArgumentParser(prog="trip")
    parser.add_argument("--runs", type=int, default=5, help="how many times to run the scenario, seeds 1..N")
    args = parser.parse_args()

    results = []
    for seed in range(1, args.runs + 1):
        result = asyncio.run(simulate(seed))
        for event in result["events"]:
            print(f"{event['time_s']:>6.2f}s  {describe(event)}")
        print(f"{result['status']}: {result['outcome'] or result['error']}")
        print(f"Saved to {save_run(result, RUNS_DIR)}\n")
        results.append(result)
    print(json.dumps(summarize(results), indent=2))
