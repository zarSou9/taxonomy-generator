import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from taxonomy_generator.utils.llm import Chat


@pytest.fixture
def mock_cache_dir(tmp_path: Path):
    cache_dir = tmp_path / ".chat_cache"
    with patch("taxonomy_generator.utils.llm.CHAT_CACHE_PATH", cache_dir):
        yield cache_dir


@pytest.fixture
def mock_ask_llm():
    with patch("taxonomy_generator.utils.llm.ask_llm") as ask_llm:
        yield ask_llm


def test_cache_functionality(mock_ask_llm: Any, mock_cache_dir: Any):
    mock_ask_llm.return_value = "First cached response"

    chat1 = Chat(temp=2)
    assert chat1.ask("How are you?", model="other") == "First cached response"

    mock_ask_llm.reset_mock()
    mock_ask_llm.return_value = "Second cached response"

    assert chat1.ask("Second message?", temp=3) == "Second cached response"

    mock_ask_llm.reset_mock()
    mock_ask_llm.return_value = "Not cached"

    chat2 = Chat(temp=2)
    assert chat2.ask("How are you?", model="other") == "First cached response"
    assert chat2.ask("Second message?", temp=3) == "Second cached response"
    assert chat2.ask("Third message?") == "Not cached"

    cache_list = json.loads((mock_cache_dir / "history.json").read_text())
    assert len(cache_list) == 1

    mock_ask_llm.reset_mock()
    mock_ask_llm.return_value = "Third"

    chat3 = Chat(temp=2)
    assert chat3.ask("How are you?", model="other") == "First cached response"
    assert chat3.ask("Second message?", temp=3) == "Second cached response"
    assert chat3.ask("Third message?", other_setting=True, verbose=True) == "Third"

    cache_list = json.loads((mock_cache_dir / "history.json").read_text())
    assert len(cache_list) == 2
