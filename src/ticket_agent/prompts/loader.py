import json
from pathlib import Path


def read_utf8_text(file_path: Path) -> str:
    content = file_path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(
            f"File '{file_path}' is empty or contains only whitespace."
        )

    return content


def load_prompts_from_file(file_path: Path) -> tuple[str, str]:
    content = read_utf8_text(file_path)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON format in '{file_path}': {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in '{file_path}'."
        )

    system_prompt = data.get("SYSTEM_PROMPT")
    finalize_prompt = data.get("FINALIZE_PROMPT")

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(
            f"'SYSTEM_PROMPT' in '{file_path}' must be a non-empty string."
        )

    if not isinstance(finalize_prompt, str) or not finalize_prompt.strip():
        raise ValueError(
            f"'FINALIZE_PROMPT' in '{file_path}' must be a non-empty string."
        )

    return system_prompt, finalize_prompt