import concurrent.futures
import concurrent.futures.thread
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, overload

from anthropic import Anthropic
from anthropic.types import Message as AntMessage
from google import genai
from google.genai.chats import Chat as GenaiChat
from google.genai.types import Content as GContent
from google.genai.types import GenerateContentConfig, GoogleSearch, Part, Tool
from tqdm import tqdm

from taxonomy_generator.utils.utils import cap_words, log

CHAT_CACHE_PATH = Path(".chat_cache")

AllModels = Literal[
    "claude-3-7-sonnet-latest",
    "claude-3-5-haiku-latest",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    "gemini-2.5-pro-exp-03-25",
]
History = str | list[str | AntMessage]


anthropic_client = Anthropic()
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def anthropic_message(
    role: Literal["user", "assistant"],
    prompt: str,
    use_cache=False,
    thinking_block: dict | None = None,
):
    return {
        "role": role,
        "content": [
            *([thinking_block] if thinking_block else []),
            {
                "text": prompt,
                "type": "text",
                **(
                    {"cache_control": {"type": "ephemeral"}}
                    if use_cache and len(prompt) > 4000
                    else {}
                ),
            },
        ],
    }


def convert_to_anthropic_messages(
    history: History,
    use_cache=False,
    max_cache_blocks=4,
    dont_cache_last=False,
) -> list[dict]:
    if isinstance(history, str):
        return [anthropic_message("user", history, use_cache)]
    messages = []
    cached_blocks = 0
    for i, m in enumerate(history):
        thinking_block = None
        if isinstance(m, AntMessage):
            thinking_block = next(
                (c for c in m.content if c.type == "thinking"),
                None,
            )
            thinking_block = thinking_block and thinking_block.model_dump()
            m = get_ant_message_text(m)

        role = "user" if i % 2 == 0 else "assistant"
        cache_this = (
            role == "user"
            and cached_blocks < max_cache_blocks
            and use_cache
            and not (dont_cache_last and i == len(history) - 1)
        )
        messages.append(anthropic_message(role, m, cache_this, thinking_block))

        if cache_this and len(m) > 4000:
            cached_blocks += 1

    return messages


def convert_to_google_messages(history: History) -> list[GContent]:
    if isinstance(history, str):
        return [GContent(role="user", parts=[Part(text=history)])]
    messages = []
    for i, m in enumerate(history):
        text = resolve_message_text(m)
        if i % 2 == 0:
            messages.append(GContent(role="user", parts=[Part(text=text)]))
        else:
            messages.append(GContent(role="model", parts=[Part(text=text)]))
    return messages


def is_google_model(model: str) -> bool:
    return model in [
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-pro-exp-03-25",
    ]


def is_anthropic_model(model: str) -> bool:
    return model in ["claude-3-7-sonnet-latest", "claude-3-5-haiku-latest"]


def get_ant_message_text(ant_message: AntMessage) -> str:
    return next(c for c in ant_message.content if c.type == "text").text


def resolve_message_text(message: str | AntMessage) -> str:
    return message if isinstance(message, str) else get_ant_message_text(message)


def get_last(history: History) -> str:
    return resolve_message_text(history[-1] if isinstance(history, list) else history)


def resolve_simple_history(history: History) -> list[str]:
    history = history if isinstance(history, list) else [history]
    return [resolve_message_text(m) for m in history]


@overload
def ask_llm(
    history: History,
    system: str | None = None,
    model: AllModels = "claude-3-7-sonnet-latest",
    temp: float | None = None,
    max_tokens: int = 8192,
    max_retries: int = 4,
    initial_retry_delay: int = 15,
    stop_sequences: list[str] = [],
    ground_with_google_search=False,
    use_cache=False,
    dont_cache_last=False,
    use_thinking: Literal[False] = False,  # <---
    thinking_budget: int = 7000,
    verbose=False,
) -> str: ...


@overload
def ask_llm(
    history: History,
    system: str | None = None,
    model: AllModels = "claude-3-7-sonnet-latest",
    temp: float | None = None,
    max_tokens: int = 8192,
    max_retries: int = 4,
    initial_retry_delay: int = 15,
    stop_sequences: list[str] = [],
    ground_with_google_search=False,
    use_cache=False,
    dont_cache_last=False,
    use_thinking: Literal[True] = True,  # <---
    thinking_budget: int = 7000,
    verbose=False,
) -> AntMessage: ...


def ask_llm(
    history: History,
    system: str | None = None,
    model: AllModels = "claude-3-7-sonnet-latest",
    temp: float | None = None,
    max_tokens: int = 8192,
    max_retries: int = 4,
    initial_retry_delay: int = 15,
    stop_sequences: list[str] = [],
    ground_with_google_search=False,
    use_cache=False,
    dont_cache_last=False,
    use_thinking=False,
    thinking_budget: int = 7000,
    verbose=False,
) -> str | AntMessage:
    if verbose:
        log(get_last(history))

    if not is_google_model(model) and ground_with_google_search:
        model = "gemini-1.5-pro"

    if use_thinking:
        model = "claude-3-7-sonnet-latest"

    response = None
    for attempt in range(max_retries):
        try:
            if is_anthropic_model(model):
                messages = convert_to_anthropic_messages(
                    history,
                    use_cache,
                    max_cache_blocks=(3 if system else 4),
                    dont_cache_last=dont_cache_last,
                )

                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "messages": messages,
                    "stop_sequences": stop_sequences,
                }

                if not use_thinking:
                    kwargs["temperature"] = temp or 0.6

                if system:
                    kwargs["system"] = [
                        {
                            "type": "text",
                            "text": system,
                            **(
                                {"cache_control": {"type": "ephemeral"}}
                                if use_cache
                                else {}
                            ),
                        }
                    ]

                if use_thinking:
                    kwargs["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": thinking_budget,
                    }

                ant_message: AntMessage = anthropic_client.messages.create(**kwargs)

                if verbose:
                    usage_strs = []
                    for name, value in ant_message.usage:
                        usage_strs.append(
                            f"{cap_words(name.replace('_', ' '))}: {value}"
                        )
                    log(
                        f"-------USAGE START------\n\n{'\n'.join(usage_strs)}\n\n-------USAGE END-------"
                    )

                if use_thinking:
                    if verbose:
                        output = "-------THINKING START------\n\n"

                        for m in ant_message.content:
                            if m.type == "thinking":
                                output += m.thinking + "\n"
                            elif m.type == "redacted_thinking":
                                output += (
                                    "------------\nREDACTED SECTION\n------------\n"
                                )

                        output += "\n-------THINKING END-------\n\n"

                        output += f"\n-------RESPONSE START-------\n\n{get_ant_message_text(ant_message)}\n\n-------RESPONSE END-------\n"

                        log(output)

                    return ant_message

                response = get_ant_message_text(ant_message)
            elif is_google_model(model):
                generation_config = GenerateContentConfig(
                    temperature=temp or 1,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                    system_instruction=system,
                )
                if ground_with_google_search:
                    generation_config.tools = [Tool(google_search=GoogleSearch())]
                    generation_config.response_modalities = ["TEXT"]

                chat: GenaiChat = genai_client.chats.create(
                    model=model,
                    config=generation_config,
                    history=(
                        None
                        if isinstance(history, str)
                        else convert_to_google_messages(history[:-1])
                    ),
                )

                response = chat.send_message(get_last(history)).text
            else:
                raise ValueError(f"Invalid model: {model}")

            if response is None:
                raise RuntimeError("No response from the API.")

            response = response.strip()
            if verbose:
                log(response)
            return response
        except Exception as e:
            if isinstance(e, ValueError):
                raise

            if attempt >= max_retries - 1:
                raise

            retry_delay = initial_retry_delay * (2**attempt)
            print(f"API error: {e!s}. Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)

    raise Exception("Max retries reached. Unable to get a response from the API.")


def run_through_convo(
    user_messages: list[str],
    verbose=False,
    stdin_chat=False,
    **kwargs,
) -> list[str | AntMessage]:
    chat_history = []
    assistant_messages = []
    i = 0
    while i < len(user_messages):
        chat_history.append(user_messages[i])
        response = ask_llm(chat_history, **kwargs)
        chat_history.append(response)
        assistant_messages.append(response)
        if verbose or stdin_chat:
            print(response + "\n\n\n")
        if stdin_chat and i == len(user_messages) - 1:
            user_messages.append(input())
            print("\n\n\n")
        i += 1
    return assistant_messages


def run_in_parallel(
    histories: Sequence[History],
    settingss: list[dict] | None = None,
    max_workers: int = 5,
    test: bool = False,
    **kwargs,
) -> list[str]:
    """Runs multiple ask_llm calls in parallel using ThreadPoolExecutor.

    Args:
        histories: A list of prompts or message histories to process
        settingss: Optional list of settings dictionaries to override kwargs for each history
        max_workers: Maximum number of concurrent workers (default: 5)
        test: If True, returns test responses instead of calling the LLM
        **kwargs: Additional arguments to pass to ask_llm

    Returns:
        list[str]: Responses in the same order as the input prompts
    """
    if test:
        return [
            f'TEST RESPONSE FROM PROMPT: "{h[:1000] + "..." if isinstance(h, str) else h}"'
            for h in histories
        ]

    results: list[str] = [""] * len(histories)

    if settingss:
        assert len(settingss) == len(histories)
    else:
        settingss = [{}] * len(histories)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt = {
            executor.submit(ask_llm, history, **kwargs, **settings): i
            for i, (history, settings) in enumerate(
                zip(histories, settingss, strict=False),
            )
        }

        for future in tqdm(
            concurrent.futures.as_completed(future_to_prompt),
            total=len(histories),
            desc="Processing prompts",
            unit="prompt",
        ):
            result = future.result()
            results[future_to_prompt[future]] = result

    return results


class ChatMessage(TypedDict):
    message: str | AntMessage
    settings_override: NotRequired[dict | None]


class ChatMessageJSON(TypedDict):
    message: str | dict
    settings_override: NotRequired[dict | None]


class Cache(TypedDict):
    settings: dict
    history: list[ChatMessageJSON]


def resolve_chat_json(chat_history: list[ChatMessage]) -> list[ChatMessageJSON]:
    return [
        ChatMessageJSON(
            message=(
                cm["message"]
                if isinstance(cm["message"], str)
                else cm["message"].model_dump(mode="json")
            ),
            settings_override=cm.get("settings_override"),
        )
        for cm in chat_history
    ]


def model_validate_chat(chat_json: list[ChatMessageJSON]) -> list[ChatMessage]:
    return [
        ChatMessage(
            message=(
                cmj["message"]
                if isinstance(cmj["message"], str)
                else AntMessage.model_validate(cmj["message"])
            ),
            settings_override=cmj.get("settings_override"),
        )
        for cmj in chat_json
    ]


def chat_to_history(chat_history: list[ChatMessage]):
    return [m["message"] for m in chat_history]


def history_to_chat(history: History | None):
    history = ([history] if isinstance(history, str) else history) or []
    return [
        ChatMessage(message=m, settings_override={})
        if i % 2 == 0
        else ChatMessage(message=m)
        for i, m in enumerate(history)
    ]


class Chat:
    def __init__(
        self,
        history: History | None = None,
        cache_file_name: str | None = "history",
        cache_limit: int = 30,
        dont_cache: bool = False,
        **kwargs,
    ):
        self.history = history_to_chat(history)
        self.settings = kwargs
        self.cache_file_name = cache_file_name
        self.cache_limit = cache_limit
        self.dont_cache = dont_cache

        CHAT_CACHE_PATH.mkdir(parents=True, exist_ok=True)
        self.cache_file = CHAT_CACHE_PATH / f"{self.cache_file_name}.json"

        self.cache_list: list[Cache] = []
        if self.cache_file.exists():
            self.cache_list = json.loads(self.cache_file.read_text())[
                -self.cache_limit :
            ]

    def ask(self, prompt: str, **kwargs) -> str:
        self.history.append(
            ChatMessage(message=prompt.strip(), settings_override=kwargs)
        )

        cached_history = self.handle_cache()

        if len(cached_history) > len(self.history):
            if kwargs.get("verbose"):
                print("Used cache!")
            response = cached_history[len(self.history)]
            self.history.append(response)
        else:
            response = ChatMessage(
                message=ask_llm(
                    chat_to_history(self.history),
                    **(self.settings | kwargs),
                ),
            )
            self.history.append(response)
            self.handle_cache()

        return resolve_message_text(response["message"])

    @property
    def simple_history(self):
        return resolve_simple_history([cm["message"] for cm in self.history])

    def handle_cache(self):
        if self.dont_cache:
            return self.history

        history_json = resolve_chat_json(self.history)

        possible_caches = [
            cache for cache in self.cache_list if cache["settings"] == self.settings
        ]

        valid_caches = [
            cache
            for cache in possible_caches
            if history_json == cache["history"][: len(history_json)]  # If chat in cache
        ]
        if valid_caches:
            longest_valid_cache = max(
                valid_caches,
                key=lambda cache: len(cache["history"]),
            )
            return model_validate_chat(longest_valid_cache["history"])

        sub_caches = [
            cache
            for cache in possible_caches
            if cache["history"] == history_json[: len(cache["history"])]
        ]
        if sub_caches:
            sub_caches[0]["history"] = history_json
            self.cache_list = [
                cache for cache in self.cache_list if cache not in sub_caches[1:]
            ]
        else:
            self.cache_list.append(Cache(settings=self.settings, history=history_json))

        self.save_cache()

        return self.history

    def save_cache(self):
        self.cache_file.write_text(
            json.dumps(self.cache_list[-self.cache_limit :], ensure_ascii=False)
        )
