import json
from pathlib import Path
from typing import Any


def extract_thinking(message: Any) -> Any:
    if isinstance(message, dict) and "message" in message:
        msg: Any = message["message"]  # pyright: ignore[reportUnknownVariableType]
        if isinstance(msg, dict) and "content" in msg:
            for content in msg["content"]:  # pyright: ignore[reportUnknownVariableType]
                if content.get("type") == "thinking":  # pyright: ignore[reportUnknownMemberType]
                    return content.get("thinking", "")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    return ""


def get_message_text(message: Any) -> Any:
    if isinstance(message, dict):
        if "message" in message:
            msg: Any = message["message"]  # pyright: ignore[reportUnknownVariableType]
            if isinstance(msg, dict) and "content" in msg:
                for content in msg["content"]:  # pyright: ignore[reportUnknownVariableType]
                    if content.get("type") == "text":  # pyright: ignore[reportUnknownMemberType]
                        return content.get("text", "")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        elif "text" in message:
            return message["text"]  # pyright: ignore[reportUnknownVariableType]
    return message.get("message", "")  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]


def format_chat(chat_data: list[dict[str, Any]]):
    output: list[str] = []

    for entry in chat_data:
        if "history" in entry:
            for msg_idx, msg in enumerate(entry["history"]):
                role = "USER" if msg_idx % 2 == 0 else "ASSISTANT"
                output.append(f"{'=' * 30} {role} {'=' * 30}")

                if role == "ASSISTANT":
                    thinking = extract_thinking(msg)
                    if thinking:
                        output.append("\n------- THINKING -------")
                        output.append(thinking)
                        output.append("------- END THINKING -------\n")

                text = get_message_text(msg)
                if text:
                    output.append(text)

                output.append(f"{'=' * 68}\n")

    return "\n".join(output)


def process_chat_file(json_path: str | Path, output_path: str | Path):
    chat_data = json.loads(Path(json_path).read_text())

    formatted_chat = format_chat(chat_data)

    Path(output_path).parent.mkdir(exist_ok=True)
    Path(output_path).write_text(formatted_chat)

    print(f"Chat history saved to {output_path}")


if __name__ == "__main__":
    process_chat_file(
        Path(".chat_cache/Reinforcement_Learning_Robustness_2025-04-16_11_0.json"),
        Path("examples/example_chat.txt"),
    )
