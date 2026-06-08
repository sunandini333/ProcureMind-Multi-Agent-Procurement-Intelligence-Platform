"""
agents/llm_client.py

Shared LLM wrapper for Procurement Copilot.

Supported providers:
    - gemini
    - anthropic
    - bedrock

Usage:
    from agents.llm_client import call_llm

    response = call_llm(
        system="You are a procurement analyst",
        user="Who are the top suppliers by spend?"
    )

    print(response)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import (
    LLM_PROVIDER,

    # Gemini
    GEMINI_API_KEY,
    GEMINI_MODEL,

    # Anthropic
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,

    # Bedrock
    AWS_REGION,
    BEDROCK_MODEL_ID,
)

from utils.logger import logger


# ============================================================
# Gemini
# ============================================================

def _call_gemini(
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> str:
    """Google Gemini API."""

    from google import genai

    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set in .env"
        )

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    prompt = f"""
SYSTEM:
{system}

USER:
{user}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
    )

    return response.text.strip()


# ============================================================
# Anthropic
# ============================================================

def _call_anthropic(
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> str:
    """Direct Anthropic API."""

    import anthropic

    if not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set."
        )

    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY
    )

    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[
            {
                "role": "user",
                "content": user,
            }
        ],
    )

    return message.content[0].text.strip()


# ============================================================
# AWS Bedrock
# ============================================================

def _call_bedrock(
    system: str,
    user: str,
    max_tokens: int = 2048,
) -> str:
    """AWS Bedrock wrapper."""

    import boto3
    import json

    client = boto3.client(
        "bedrock-runtime",
        region_name=AWS_REGION,
    )

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "system": system,
            "messages": [
                {
                    "role": "user",
                    "content": user,
                }
            ],
        }
    )

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
    )

    result = json.loads(
        response["body"].read()
    )

    return result["content"][0]["text"].strip()


# ============================================================
# Public API
# ============================================================

def call_llm(
    system: str,
    user: str,
    max_tokens: int = 2048,
    provider: str | None = None,
) -> str:
    """
    Unified LLM interface.

    Args:
        system: System prompt
        user: User prompt
        max_tokens: Max response tokens
        provider: Override provider

    Returns:
        Plain text response
    """

    active_provider = (
        provider or LLM_PROVIDER
    ).lower()

    logger.debug(
        f"LLM call -> provider={active_provider}"
    )

    if active_provider == "gemini":
        return _call_gemini(
            system,
            user,
            max_tokens,
        )

    if active_provider == "anthropic":
        return _call_anthropic(
            system,
            user,
            max_tokens,
        )

    if active_provider == "bedrock":
        return _call_bedrock(
            system,
            user,
            max_tokens,
        )

    raise ValueError(
        f"Unknown provider: {active_provider}"
    )