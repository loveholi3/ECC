import os
import sys
from unittest.mock import patch, mock_open, call
import pytest

from llm.cli.selector import (
    select_provider,
    select_model,
    save_config,
    interactive_select,
    main,
)


@pytest.fixture
def mock_providers():
    return [
        ("claude", "Anthropic Claude"),
        ("openai", "OpenAI GPT"),
    ]


@patch("builtins.input")
def test_select_provider_empty(mock_input, capsys):
    assert select_provider([]) is None
    captured = capsys.readouterr()
    assert "No providers available." in captured.out
    mock_input.assert_not_called()


@patch("builtins.input")
def test_select_provider_valid(mock_input, mock_providers):
    mock_input.return_value = "1"
    assert select_provider(mock_providers) == "claude"


@patch("builtins.input")
def test_select_provider_cancel(mock_input, mock_providers):
    mock_input.return_value = ""
    assert select_provider(mock_providers) is None


@patch("builtins.input")
def test_select_provider_invalid_then_valid(mock_input, mock_providers, capsys):
    # out of bounds, out of bounds, non-integer, valid
    mock_input.side_effect = ["5", "0", "abc", "2"]

    assert select_provider(mock_providers) == "openai"

    assert mock_input.call_count == 4
    captured = capsys.readouterr()
    assert "Invalid selection. Try again." in captured.out
    assert "Please enter a number." in captured.out


@pytest.fixture
def mock_models():
    return [
        ("gpt-4o", "GPT-4o"),
        ("gpt-3.5-turbo", "GPT-3.5"),
    ]


@patch("builtins.input")
def test_select_model_empty(mock_input, capsys):
    assert select_model([]) is None
    captured = capsys.readouterr()
    assert "No models available." in captured.out
    mock_input.assert_not_called()


@patch("builtins.input")
def test_select_model_valid(mock_input, mock_models):
    mock_input.return_value = "1"
    assert select_model(mock_models) == "gpt-4o"


@patch("builtins.input")
def test_select_model_cancel(mock_input, mock_models):
    mock_input.return_value = ""
    assert select_model(mock_models) is None


@patch("builtins.input")
def test_select_model_invalid_then_valid(mock_input, mock_models, capsys):
    # out of bounds, out of bounds, non-integer, valid
    mock_input.side_effect = ["5", "0", "abc", "2"]

    assert select_model(mock_models) == "gpt-3.5-turbo"

    assert mock_input.call_count == 4
    captured = capsys.readouterr()
    assert "Invalid selection. Try again." in captured.out
    assert "Please enter a number." in captured.out


@patch("builtins.open", new_callable=mock_open)
@patch.dict(os.environ, {}, clear=True)
def test_save_config_no_persist(mock_file):
    save_config("test-provider", "test-model")

    mock_file.assert_called_once_with(".llm.env", "w")
    mock_file().write.assert_called_once_with("LLM_PROVIDER=test-provider\nLLM_MODEL=test-model\n")
    assert "LLM_PROVIDER" not in os.environ


@patch("builtins.open", new_callable=mock_open)
@patch.dict(os.environ, {}, clear=True)
def test_save_config_persist(mock_file):
    save_config("test-provider", "test-model", persist=True)

    mock_file.assert_called_once_with(".llm.env", "w")
    mock_file().write.assert_called_once_with("LLM_PROVIDER=test-provider\nLLM_MODEL=test-model\n")
    assert os.environ["LLM_PROVIDER"] == "test-provider"
    assert os.environ["LLM_MODEL"] == "test-model"


@patch("llm.cli.selector.save_config")
@patch("llm.cli.selector.select_model")
@patch("llm.cli.selector.select_provider")
def test_interactive_select_success(mock_select_provider, mock_select_model, mock_save_config):
    mock_select_provider.return_value = "claude"
    mock_select_model.return_value = "claude-opus-4-8"

    result = interactive_select()

    assert result == ("claude", "claude-opus-4-8")
    mock_save_config.assert_called_once_with("claude", "claude-opus-4-8", False)


@patch("llm.cli.selector.select_provider")
def test_interactive_select_cancel_provider(mock_select_provider):
    mock_select_provider.return_value = None
    assert interactive_select() is None


@patch("llm.cli.selector.select_model")
@patch("llm.cli.selector.select_provider")
def test_interactive_select_cancel_model(mock_select_provider, mock_select_model):
    mock_select_provider.return_value = "claude"
    mock_select_model.return_value = None
    assert interactive_select() is None


@patch("llm.cli.selector.interactive_select")
def test_main_success(mock_interactive_select, capsys):
    mock_interactive_select.return_value = ("openai", "gpt-4o")
    main()

    captured = capsys.readouterr()
    assert "export LLM_PROVIDER=openai" in captured.out
    assert "export LLM_MODEL=gpt-4o" in captured.out


@patch("llm.cli.selector.interactive_select")
@patch("sys.exit")
def test_main_cancel(mock_exit, mock_interactive_select):
    mock_interactive_select.return_value = None
    main()

    mock_exit.assert_called_once_with(0)
