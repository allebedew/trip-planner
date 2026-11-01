# Picos de Europa trip planner

Two LLM agents and two scripted humans plan and book a one-day hike in Picos de Europa.

```text
human_a <-> agent_a <-> agent_b <-> human_b
```

## Run

Requires [uv](https://docs.astral.sh/uv/) and an Anthropic API key in `.env`.

```bash
uv sync
```

```bash
cp .env.example .env
```

```bash
uv run trip
```

This runs the scenario with seeds 1 to 5, `--runs 1` runs it once. Each run is saved to `runs/seed-N.json`.

## How it works

Each agent reaches only its own environment, they share no state.

| Agent | Actions (`*` can fail) |
|---|---|
| A | `send_message`, `get_trip_options`*, `finish` |
| B | `send_message`, `check_trip`*, `book_trip`* |

On each tick the previous tick's messages are delivered and agents with new mail run concurrently. The run
ends when nobody has anything to say.

The LLM chooses the route and the date. Code filters unsafe routes, computes the gear and the bookings, and
validates the result.

## Results

5 of 5 runs successful, all `PR-PNPE-10 2026-11-01`. Example: [trajectory.json](examples/trajectory.json).

## Limitations

- The humans are scripted, and the guide always agrees.
- Weather and routes are snapshots.
- The LLM's choice is measured with runs, not guaranteed by code.
