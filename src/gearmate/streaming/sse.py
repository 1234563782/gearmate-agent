import json

from gearmate.persistence.repositories import EventRecord


def encode_event(event: EventRecord) -> str:
    data = json.dumps(event.payload, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.sequence_no}\nevent: {event.event_type}\ndata: {data}\n\n"


def heartbeat() -> str:
    return ": heartbeat\n\n"
