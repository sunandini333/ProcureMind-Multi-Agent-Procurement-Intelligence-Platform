"""
agents/llm_client.py
Shared LLM wrapper for all Procurement Copilot agents.

Supports two providers, toggled via LLM_PROVIDER in config / .env:
  - "anthropic"  → direct Anthropic API (dev/local)
  - "bedrock"    → AWS Bedrock (Dell enterprise env)

Usage:
    from agents.llm_client import call_llm
    response = call_llm(system="You are...", user="What are the top suppliers?")
    print(response)  # plain string
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    LLM_PROVIDER,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    AWS_REGION,
    BEDROCK_MODEL_ID,
)
from utils.logger import logger


# ── Provider implementations ──────────────────────────────────────────────────

def _call_anthropic(system: str, user: str, max_tokens: int = 2048) -> str:
    """Direct Anthropic API call."""
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example → .env and add your key."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _call_bedrock(system: str, user: str, max_tokens: int = 2048) -> str:
    """AWS Bedrock call — drop-in replacement for Dell environment."""
    import boto3, json

    client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    })
    response = client.invoke_model(modelId=BEDROCK_MODEL_ID, body=body)
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


# ── Public interface ──────────────────────────────────────────────────────────

def call_llm(
    system: str,
    user: str,
    max_tokens: int = 2048,
    provider: str | None = None,
) -> str:
    """
    Call the configured LLM and return the response as a plain string.

    Args:
        system:     System prompt.
        user:       User message.
        max_tokens: Max tokens in the response.
        provider:   Override LLM_PROVIDER for this call ("anthropic" | "bedrock").

    Returns:
        Response text as a plain string.
    """
    active_provider = (provider or LLM_PROVIDER).lower()
    logger.debug(f"LLM call → provider={active_provider}, ~{len(user)} user chars")

    if active_provider == "anthropic":
        return _call_anthropic(system, user, max_tokens)
    elif active_provider == "bedrock":
        return _call_bedrock(system, user, max_tokens)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: '{active_provider}'. Use 'anthropic' or 'bedrock'.")
