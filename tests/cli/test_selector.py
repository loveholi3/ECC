import pytest
from llm.cli.selector import print_providers, Color


def test_print_providers_normal(capsys):
    providers = [
        ("claude", "Anthropic Claude"),
        ("openai", "OpenAI GPT"),
    ]
    print_providers(providers)
    captured = capsys.readouterr()

    # Check header
    assert "Available Providers:" in captured.out

    # Check first provider
    assert "1. " not in captured.out # because color escape codes are between 1 and .
    assert f"  {Color.GREEN}1{Color.RESET}. {Color.BOLD}claude{Color.RESET} - Anthropic Claude\n" in captured.out

    # Check second provider
    assert f"  {Color.GREEN}2{Color.RESET}. {Color.BOLD}openai{Color.RESET} - OpenAI GPT\n" in captured.out


def test_print_providers_empty(capsys):
    print_providers([])
    captured = capsys.readouterr()

    assert f"\n{Color.BOLD}Available Providers:{Color.RESET}\n\n" == captured.out
