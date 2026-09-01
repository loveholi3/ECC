import json
from typing import Any


def parse_tool_arguments(raw_arguments: str | None) -> dict[str, Any]:
    if not raw_arguments:
        return {}

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"raw": raw_arguments}

    if isinstance(arguments, dict):
        return arguments
    return {"value": arguments}
