import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from trip_planner.actions import Message

load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_TOKENS = 4096
EFFORT = "medium"  # low | medium | high | xhigh | max
THINKING_DISPLAY = "omitted"  # omitted | summarized
MAX_RETRIES = 2
TIMEOUT_S = 600
# USD per 1M tokens for MODEL, prompts up to 100K tokens.
INPUT_PRICE_USD = 0.10
OUTPUT_PRICE_USD = 0.50
# Limit per handled message. One LLM step takes two graph steps: the model and the tools.
MAX_GRAPH_STEPS = 32

COMMON_RULES = """
Rules:
- Text you write outside send_message reaches nobody. Only send_message talks to people and agents.
- Replies arrive later as new messages. After you have sent what is needed, stop calling tools and wait.
- Never invent facts. Use only what tools and messages gave you.
- If a tool returns an error that says it is safe to call again, call it again.
"""

AGENT_A_INSTRUCTIONS = (
    """You are agent_a, the assistant of a client (human_a) who wants a one-day hike in Picos de Europa.
You can write to human_a and agent_b. Your goal is a booked trip that fits the client best.

Call get_trip_options and choose by this rule:
- the route whose distance is closest to the client's preferred length;
- then the day with the lowest rain probability; if days tie, the one with the least wind.

Propose the client's name, the route id and the date to agent_b.
If agent_b refuses, apply the same rule without what was refused: a closed route is out on every date,
a busy date is out for every route. Propose the new choice.
When agent_b sends a booking id, call finish with the agreed route, date and that booking id.
finish delivers the plan to human_a, so do not write to human_a yourself.
"""
    + COMMON_RULES
)

AGENT_B_INSTRUCTIONS = (
    """You are agent_b, the assistant of a mountain guide (human_b). You can write to human_b and agent_a.
Your goal is to book the trips the guide can take.

When agent_a proposes a trip, call check_trip. If it is not available, send agent_a the reason.
If it is available, ask human_b. When human_b agrees, call book_trip and send agent_a the booking id,
the route id and the date.
Write to human_b only to ask about a trip, once per trip. Write to agent_a only a refusal or a booking.
"""
    + COMMON_RULES
)


def render(message: Message) -> str:
    return f"From {message.sender}: {message.text}"


def to_event(message: AnyMessage) -> dict:
    if isinstance(message, AIMessage):
        usage = message.usage_metadata
        return {
            "kind": "llm_call",
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "thinking_tokens": usage.get("output_token_details", {}).get("reasoning", 0),
            "cost_usd": (usage["input_tokens"] * INPUT_PRICE_USD + usage["output_tokens"] * OUTPUT_PRICE_USD) / 1e6,
            "text": message.text,
            "calls": [{"action": call["name"], "args": call["args"]} for call in message.tool_calls],
        }
    return {"kind": "action", "action": message.name, "result": message.content}


class Agent:
    """The same agent for both roles: only its actions and instructions differ.
    The history stays inside the object, so agents share no state."""

    def __init__(self, name: str, actions: list, instructions: str):
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY in .env")
        model = ChatAnthropic(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            reasoning_effort=EFFORT,
            thinking={"type": "adaptive", "display": THINKING_DISPLAY},
            max_retries=MAX_RETRIES,
            timeout=TIMEOUT_S,
        )
        self.graph = create_agent(model, actions, system_prompt=instructions, name=name)
        self.history: list[AnyMessage] = []

    async def handle(self, inbox: list[Message]) -> list[dict]:
        """Runs the LLM and its actions until it stops, returns what happened as events."""
        prompt = HumanMessage("\n\n".join(render(m) for m in inbox))
        messages = [*self.history, prompt]
        result = await self.graph.ainvoke({"messages": messages}, {"recursion_limit": MAX_GRAPH_STEPS})
        new_messages = result["messages"][len(messages) :]
        # The API rejects an empty assistant turn in the history.
        self.history = [m for m in result["messages"] if m.content or getattr(m, "tool_calls", None)]
        return [to_event(m) for m in new_messages]
