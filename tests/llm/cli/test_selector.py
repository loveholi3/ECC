import os
import sys
from unittest.mock import patch, mock_open, call
import pytest
from llm.cli.selector import (
    print_banner,
    print_providers,
    select_provider,
    select_model,
    save_config,
    interactive_select,
    main,
    Color
)

@pytest.fixture
def mock_providers():
    return [
        ("provider1", "First Provider"),
        ("provider2", "Second Provider"),
    ]

@pytest.fixture
def mock_models():
    return [
        ("model1", "First Model"),
        ("model2", "Second Model"),
    ]

def test_print_banner(capsys):
    print_banner()
    captured = capsys.readouterr()
    assert "LLM Provider Selector" in captured.out
    assert "Provider-agnostic AI interactions" in captured.out
    assert Color.CYAN.value in captured.out or str(Color.CYAN) in captured.out

def test_print_providers(capsys, mock_providers):
    print_providers(mock_providers)
    captured = capsys.readouterr()
    assert "Available Providers:" in captured.out
    assert "provider1" in captured.out
    assert "First Provider" in captured.out
    assert "provider2" in captured.out
    assert "Second Provider" in captured.out

def test_select_provider_empty(capsys):
    result = select_provider([])
    captured = capsys.readouterr()
    assert result is None
    assert "No providers available." in captured.out

@patch("builtins.input", return_value="1")
def test_select_provider_valid(mock_input, mock_providers):
    result = select_provider(mock_providers)
    assert result == "provider1"
    mock_input.assert_called_once()

@patch("builtins.input", side_effect=["invalid", "3", "2"])
def test_select_provider_invalid_then_valid(mock_input, mock_providers, capsys):
    result = select_provider(mock_providers)
    captured = capsys.readouterr()
    assert result == "provider2"
    assert mock_input.call_count == 3
    assert "Please enter a number." in captured.out
    assert "Invalid selection. Try again." in captured.out

@patch("builtins.input", return_value="")
def test_select_provider_empty_input(mock_input, mock_providers):
    result = select_provider(mock_providers)
    assert result is None

def test_select_model_empty(capsys):
    result = select_model([])
    captured = capsys.readouterr()
    assert result is None
    assert "No models available." in captured.out

@patch("builtins.input", return_value="2")
def test_select_model_valid(mock_input, mock_models):
    result = select_model(mock_models)
    assert result == "model2"
    mock_input.assert_called_once()

@patch("builtins.input", side_effect=["", ])
def test_select_model_empty_input(mock_input, mock_models):
    result = select_model(mock_models)
    assert result is None

@patch("builtins.input", side_effect=["a", "0", "1"])
def test_select_model_invalid_then_valid(mock_input, mock_models, capsys):
    result = select_model(mock_models)
    captured = capsys.readouterr()
    assert result == "model1"
    assert mock_input.call_count == 3
    assert "Please enter a number." in captured.out
    assert "Invalid selection. Try again." in captured.out

@patch("builtins.open", new_callable=mock_open)
def test_save_config_no_persist(mock_file, capsys):
    save_config("myprovider", "mymodel", persist=False)
    mock_file.assert_called_once_with(".llm.env", "w")
    mock_file().write.assert_called_once_with("LLM_PROVIDER=myprovider\nLLM_MODEL=mymodel\n")
    captured = capsys.readouterr()
    assert "Config saved to" in captured.out
    assert "Config loaded to current session" not in captured.out

@patch("os.environ", {})
@patch("builtins.open", new_callable=mock_open)
def test_save_config_persist(mock_file, capsys):
    save_config("myprovider", "mymodel", persist=True)
    mock_file.assert_called_once_with(".llm.env", "w")
    mock_file().write.assert_called_once_with("LLM_PROVIDER=myprovider\nLLM_MODEL=mymodel\n")
    captured = capsys.readouterr()
    assert "Config saved to" in captured.out
    assert "Config loaded to current session" in captured.out
    assert os.environ.get("LLM_PROVIDER") == "myprovider"
    assert os.environ.get("LLM_MODEL") == "mymodel"

@patch("llm.cli.selector.save_config")
@patch("llm.cli.selector.select_model")
@patch("llm.cli.selector.select_provider")
def test_interactive_select_success(mock_select_provider, mock_select_model, mock_save_config, capsys):
    mock_select_provider.return_value = "claude"
    mock_select_model.return_value = "claude-haiku-4-5"

    result = interactive_select(persist=True)

    assert result == ("claude", "claude-haiku-4-5")
    mock_select_provider.assert_called_once()
    mock_select_model.assert_called_once()
    mock_save_config.assert_called_once_with("claude", "claude-haiku-4-5", True)

    captured = capsys.readouterr()
    assert "Selected:" in captured.out
    assert "claude" in captured.out
    assert "claude-haiku-4-5" in captured.out

@patch("llm.cli.selector.select_provider")
def test_interactive_select_cancel_provider(mock_select_provider):
    mock_select_provider.return_value = None
    result = interactive_select()
    assert result is None

@patch("llm.cli.selector.select_model")
@patch("llm.cli.selector.select_provider")
def test_interactive_select_cancel_model(mock_select_provider, mock_select_model):
    mock_select_provider.return_value = "openai"
    mock_select_model.return_value = None
    result = interactive_select()
    assert result is None

@patch("sys.exit")
@patch("llm.cli.selector.interactive_select")
def test_main_success(mock_interactive_select, mock_exit, capsys):
    mock_interactive_select.return_value = ("my_provider", "my_model")
    main()

    captured = capsys.readouterr()
    assert "Ready to use!" in captured.out
    assert "export LLM_PROVIDER=my_provider" in captured.out
    assert "export LLM_MODEL=my_model" in captured.out
    mock_exit.assert_not_called()

@patch("sys.exit")
@patch("llm.cli.selector.interactive_select")
def test_main_cancel(mock_interactive_select, mock_exit, capsys):
    mock_interactive_select.return_value = None
    main()

    captured = capsys.readouterr()
    assert "Selection cancelled." in captured.out
    mock_exit.assert_called_once_with(0)
