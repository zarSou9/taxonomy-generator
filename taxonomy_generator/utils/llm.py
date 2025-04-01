import concurrent.futures
import concurrent.futures.thread
import os
import time
from typing import Literal, Optional

from anthropic import Anthropic
from google import genai
from google.genai.chats import Chat as GenaiChat
from google.genai.types import Content, GenerateContentConfig, GoogleSearch, Part, Tool

from taxonomy_generator.utils.utils import get_last, log_space

anthropic_client = Anthropic()
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def anthropic_user_message(prompt: str, use_cache=False):
    return {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt,
                **(
                    {"cache_control": {"type": "ephemeral"}}
                    if use_cache and len(prompt) > 4000
                    else {}
                ),
            }
        ],
    }


def anthropic_assistant_message(response: str):
    return {
        "role": "assistant",
        "content": response,
    }


def convert_to_anthropic_messages(
    prompt_or_messages: str | list[str],
    use_cache=False,
    max_cache_blocks=4,
    dont_cache_last=False,
):
    if isinstance(prompt_or_messages, str):
        return [anthropic_user_message(prompt_or_messages, use_cache)]
    else:
        messages = []
        cached_blocks = 0
        for i, m in enumerate(prompt_or_messages):
            if i % 2 == 0:
                messages.append(
                    anthropic_user_message(
                        m,
                        cached_blocks < max_cache_blocks
                        and use_cache
                        and (not dont_cache_last or i != len(prompt_or_messages) - 1),
                    )
                )
                if len(m) > 4000:
                    cached_blocks += 1
            else:
                messages.append(anthropic_assistant_message(m))
        return messages


def convert_to_google_messages(prompt_or_messages: str | list[str]) -> list[Content]:
    if isinstance(prompt_or_messages, str):
        return [Content(role="user", parts=[Part(text=prompt_or_messages)])]
    else:
        messages = []
        for i, m in enumerate(prompt_or_messages):
            if i % 2 == 0:
                messages.append(Content(role="user", parts=[Part(text=m)]))
            else:
                messages.append(Content(role="model", parts=[Part(text=m)]))
        return messages


def is_google_model(model):
    google_models = [
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-pro-exp-03-25",
    ]
    for m in google_models:
        if m == model:
            return True
    return False


def ask_llm(
    prompt_or_messages: str | list[str],
    system: Optional[str] = None,
    model: Literal[
        "claude-3-7-sonnet-latest",
        "gemini-1.5-pro",
        "gemini-2.0-flash",
        "gemini-2.5-pro-exp-03-25",
    ] = "claude-3-7-sonnet-latest",
    temp=None,
    max_tokens=8192,
    max_retries=4,
    initial_retry_delay=15,
    stop_sequences: list[str] = [],
    ground_with_google_search=False,
    use_cache=False,
    dont_cache_last=False,
    verbose=False,
) -> str | None:
    if verbose:
        log_space(get_last(prompt_or_messages))

    if not is_google_model(model) and ground_with_google_search:
        model = "gemini-1.5-pro"

    response = None
    for attempt in range(max_retries):
        try:
            if model == "claude-3-7-sonnet-latest":
                messages = convert_to_anthropic_messages(
                    prompt_or_messages,
                    use_cache,
                    max_cache_blocks=(3 if system else 4),
                    dont_cache_last=dont_cache_last,
                )

                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temp or 0.6,
                    "messages": messages,
                    "stop_sequences": stop_sequences,
                }

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

                response = anthropic_client.messages.create(**kwargs).content[0].text
            else:
                generation_config = GenerateContentConfig(
                    temperature=temp or 1,
                    max_output_tokens=max_tokens,
                    stop_sequences=stop_sequences,
                )
                if ground_with_google_search:
                    generation_config.tools = [Tool(google_search=GoogleSearch())]
                    generation_config.response_modalities = ["TEXT"]

                chat: GenaiChat = genai_client.chats.create(
                    model=model,
                    config=generation_config,
                    history=convert_to_google_messages(prompt_or_messages[:-1]),
                )

                try:
                    response = chat.send_message(get_last(prompt_or_messages)).text
                except Exception as e:
                    print(f"Gemini stopped generating a response: {str(e)}")
                    return None

            if response:
                if verbose:
                    log_space(response)
                return response
        except Exception as e:
            if attempt < max_retries - 1:
                retry_delay = initial_retry_delay * (2**attempt)
                print(f"API error: {str(e)}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                raise  # Re-raise the exception if we've exhausted our retries or it's a different error

    raise Exception("Max retries reached. Unable to get a response from the API.")


def run_through_convo(
    user_messages: list[str], verbose=False, stdin_chat=False, **kwargs
):
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
    prompts_or_messages_list: list[str | list[str]], max_workers: int = 5, **kwargs
) -> list[str | None]:
    """
    Runs multiple ask_llm calls in parallel using ThreadPoolExecutor.

    Args:
        prompts_or_messages_list: A list of prompts or message histories to process
        max_workers: Maximum number of concurrent workers (default: 5)
        **kwargs: Additional arguments to pass to ask_llm

    Returns:
        list[str | None]: Responses in the same order as the input prompts
    """

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_prompt = {
            executor.submit(ask_llm, prompt_or_messages, **kwargs): i
            for i, prompt_or_messages in enumerate(prompts_or_messages_list)
        }

        # Initialize results list with placeholders
        results = [None] * len(prompts_or_messages_list)

        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_prompt):
            prompt_index = future_to_prompt[future]
            try:
                result = future.result()
                results[prompt_index] = result
            except Exception as e:
                print(f"Error processing prompt at index {prompt_index}: {str(e)}")
                results[prompt_index] = None

    return results


class Chat:
    def __init__(self, history: list[str] | str | None = None, **kwargs):
        self.history = ([history] if isinstance(history, str) else history) or []
        self.settings = kwargs

    def ask(self, prompt: str | None = None, **kwargs) -> str:
        if prompt:
            self.history.append(prompt.strip())
        response = ask_llm(self.history, **(self.settings | kwargs))
        self.history.append(response)
        return response
