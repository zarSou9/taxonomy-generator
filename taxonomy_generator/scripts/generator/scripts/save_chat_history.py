import json
from pathlib import Path


def extract_thinking(message):
    if isinstance(message, dict) and "message" in message:
        msg = message["message"]
        if isinstance(msg, dict) and "content" in msg:
            for content in msg["content"]:
                if content.get("type") == "thinking":
                    return content.get("thinking", "")
    return ""


def get_message_text(message):
    if isinstance(message, dict):
        if "message" in message:
            msg = message["message"]
            if isinstance(msg, dict) and "content" in msg:
                for content in msg["content"]:
                    if content.get("type") == "text":
                        return content.get("text", "")
        elif "text" in message:
            return message["text"]
    return message.get("message", "")


def format_chat(chat_data):
    output = []

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


def process_chat_file(json_path, output_path):
    json_path = Path(json_path)
    chat_data = json.loads(json_path.read_text())

    formatted_chat = format_chat(chat_data)

    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    output_path.write_text(formatted_chat)

    print(f"Chat history saved to {output_path}")


if __name__ == "__main__":
    process_chat_file(
        Path(".chat_cache/Reinforcement_Learning_Robustness_2025-04-16_11_0.json"),
        Path("examples/example_chat.txt"),
    )
