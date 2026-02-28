"""
LLM client wrappers for the wargame simulation.
Supports Groq, Mistral, and OpenRouter APIs (all OpenAI-compatible).

API keys are loaded from a .env file. See .env.example for the required variables.
"""

import os
import time
import re
import openai
import requests
from dataclasses import dataclass
from typing import Optional, Tuple
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for a specific model endpoint."""
    model_id: str
    display_name: str
    service: str
    api_key: str
    base_url: str
    max_tokens: int = 4096
    temperature: float = 0.7
    rpm_limit: int = 30


# ── Model configurations ──────────────────────────────────────────────────

CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")

MODELS = {
    "clemson-qwen3-30b": ModelConfig(
        model_id="qwen3-30b-a3b-instruct-fp8",
        display_name="Qwen3 30B (Clemson)",
        service="clemson",
        api_key=os.getenv("CLEMSON_API_KEY", ""),
        base_url="https://llm.rcd.clemson.edu/v1",
        max_tokens=1024,
        rpm_limit=20,  # Unlimited, but be polite
    ),
    "groq-qwen3-32b": ModelConfig(
        model_id="qwen/qwen3-32b",
        display_name="Qwen3 32B (Groq)",
        service="groq",
        api_key=os.getenv("GROQ_API_KEY", ""),
        base_url="https://api.groq.com/openai/v1",
        max_tokens=1024,
        rpm_limit=2,
    ),
    "or-gpt-oss-120b": ModelConfig(
        model_id="openai/gpt-oss-120b:free",
        display_name="GPT-OSS 120B (OpenRouter)",
        service="openrouter",
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=1024,
        rpm_limit=5,
    ),
    "cf-llama-70b": ModelConfig(
        model_id="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        display_name="Llama 3.3 70B (CF)",
        service="cloudflare",
        api_key=os.getenv("CLOUDFLARE_API_KEY", ""),
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run",
        max_tokens=1024,
        rpm_limit=10,
    ),
}


class LLMClient:
    """Wrapper for making LLM API calls with rate limiting and parsing."""

    def __init__(self, model_key: str):
        self.config = MODELS[model_key]
        self._is_cloudflare = self.config.service == 'cloudflare'
        if not self._is_cloudflare:
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
        else:
            self.client = None
        self._last_call_time = 0.0
        self._min_interval = 60.0 / self.config.rpm_limit

    def _rate_limit(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_call_time = time.time()

    def _call_cloudflare(self, system_prompt: str, user_prompt: str) -> str:
        """Make a Cloudflare Workers AI API call."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        r = requests.post(
            f"{self.config.base_url}/{self.config.model_id}",
            headers=headers,
            json={
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": self.config.max_tokens,
            },
            timeout=120,
        )
        data = r.json()
        if data.get("success"):
            return data["result"]["response"]
        else:
            raise RuntimeError(f"Cloudflare API error: {data.get('errors', data)}")

    def call(self, system_prompt: str, user_prompt: str,
             max_retries: int = 5) -> str:
        """Make an API call and return the response text."""
        self._rate_limit()

        for attempt in range(max_retries):
            try:
                if self._is_cloudflare:
                    return self._call_cloudflare(system_prompt, user_prompt)

                resp = self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
                return resp.choices[0].message.content
            except openai.RateLimitError:
                wait = min(5 * (attempt + 1), 30)
                print(f"  [rate-limit, wait {wait}s]", end="", flush=True)
                time.sleep(wait)
            except openai.APIError as e:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    print(f"  [API err, wait {wait}s]", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 3 * (attempt + 1)
                    print(f"  [err: {type(e).__name__}, wait {wait}s]", end="", flush=True)
                    time.sleep(wait)
                else:
                    raise

        raise RuntimeError(f"Failed after {max_retries} retries for {self.config.display_name}")


def parse_response(text: str) -> Tuple[str, str, str]:
    """Parse a model response into (reasoning, signal, action).

    Returns (reasoning, signal_code, action_code).
    If parsing fails, returns best-effort extraction.
    """
    reasoning = ""
    signal = ""
    action = ""

    # Strip <think>...</think> tags (Qwen3 thinking mode)
    clean_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    if not clean_text:
        clean_text = text  # Fallback to original if stripping removed everything

    text = clean_text

    # Try to extract REASONING
    m = re.search(r'REASONING:\s*(.*?)(?=\nSIGNAL:|\Z)', text, re.DOTALL)
    if m:
        reasoning = m.group(1).strip()

    # Try to extract SIGNAL
    m = re.search(r'SIGNAL:\s*(.*?)(?=\nACTION:|\Z)', text, re.DOTALL)
    if m:
        signal_text = m.group(1).strip()
        # Extract action code from signal text
        code_match = re.search(r'((?:ESC|DEESC)_\d+)', signal_text)
        if code_match:
            signal = code_match.group(1)
        else:
            signal = signal_text[:50]

    # Try to extract ACTION
    m = re.search(r'ACTION:\s*(.*?)(?:\n|\Z)', text, re.DOTALL)
    if m:
        action_text = m.group(1).strip()
        code_match = re.search(r'((?:ESC|DEESC)_\d+)', action_text)
        if code_match:
            action = code_match.group(1)
        else:
            action = action_text[:50]

    # Fallback: try to find any action code in the text
    if not action:
        codes = re.findall(r'((?:ESC|DEESC)_\d+)', text)
        if codes:
            action = codes[-1]  # Take the last one mentioned
        else:
            action = "ESC_00"  # Default to status quo

    if not signal:
        signal = action  # Default signal to match action

    if not reasoning:
        reasoning = text[:500]

    return reasoning, signal, action
