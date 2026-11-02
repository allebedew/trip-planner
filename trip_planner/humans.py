from trip_planner.actions import Message, load_json

CLIENT = load_json("client")


def human_a_opening() -> Message:
    return Message(
        sender="human_a",
        to="agent_a",
        text=(
            f"Hi, this is {CLIENT['name']}! I would like a one-day hike in Picos de Europa in the first half "
            f"of November, about {CLIENT['preferred_km']} km long, on the day with the best weather. "
            "My knee hurts on long descents."
        ),
    )


async def human_b(message: Message) -> list[Message]:
    """The guide agrees: agent_b only asks about trips it has already checked."""
    return [Message(sender="human_b", to="agent_b", text="Yes, I can take this trip.")]
