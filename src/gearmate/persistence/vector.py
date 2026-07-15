from typing import Any

from sqlalchemy.types import UserDefinedType


class Vector1024(UserDefinedType[Any]):
    cache_ok = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        pass

    def get_col_spec(self, **kw: object) -> str:
        return "VECTOR(1024)"
