import os
from unittest.mock import patch, mock_open
import pytest
from llm.cli.selector import save_config, Color

def test_save_config_default_no_persist(capsys):
    provider = "claude"
    model = "claude-sonnet-4-6"
    expected_content = f"LLM_PROVIDER={provider}\nLLM_MODEL={model}\n"

    m_open = mock_open()
    with patch("builtins.open", m_open), patch.dict(os.environ, {}, clear=True):
        save_config(provider, model)

        captured = capsys.readouterr()
        assert f"Config saved to {Color.CYAN}.llm.env{Color.RESET}" in captured.out
        assert "LLM_PROVIDER" not in os.environ
        assert "LLM_MODEL" not in os.environ

    m_open.assert_called_once_with(".llm.env", "w")
    m_open().write.assert_called_once_with(expected_content)


def test_save_config_with_persist(capsys):
    provider = "openai"
    model = "gpt-4o"
    expected_content = f"LLM_PROVIDER={provider}\nLLM_MODEL={model}\n"

    m_open = mock_open()
    with patch("builtins.open", m_open), patch.dict(os.environ, {}, clear=True):
        save_config(provider, model, persist=True)

        captured = capsys.readouterr()
        assert f"Config saved to {Color.CYAN}.llm.env{Color.RESET}" in captured.out
        assert f"Config loaded to current session" in captured.out

        assert os.environ.get("LLM_PROVIDER") == provider
        assert os.environ.get("LLM_MODEL") == model

    m_open.assert_called_once_with(".llm.env", "w")
    m_open().write.assert_called_once_with(expected_content)
