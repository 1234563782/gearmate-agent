from secrets import randbits
from time import time_ns

CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid() -> str:
    timestamp_ms = time_ns() // 1_000_000
    value = (timestamp_ms << 80) | randbits(80)
    chars = ["0"] * 26
    for index in range(25, -1, -1):
        chars[index] = CROCKFORD[value & 31]
        value >>= 5
    return "".join(chars)
