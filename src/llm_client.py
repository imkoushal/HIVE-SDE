"""
llm_client.py — Unified LLM client abstraction.

Supports OpenAI, Groq, and Google Gemini APIs with a simple interface.
Handles rate limiting, retries, and structured JSON output.
"""

import os
import re
import json
import time
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ─── Base Client ────────────────────────────────────────────────────────────

class LLMClient:
    """Unified LLM client with retry logic."""

    def __init__(self, provider: str = None, model: str = None, temperature: float = 0.3):
        """
        Initialize the LLM client.

        Args:
            provider: 'openai', 'groq', or 'gemini'. Auto-detected from env if None.
            model: Model name. Defaults based on provider.
            temperature: Sampling temperature.
        """
        self.temperature = temperature
        self.max_retries = 8
        self.retry_delay = 2.0
        self._call_delay = 1.0 if provider == "groq" else 0.0  # Proactive rate limiting (API calls take ~3.7s, so ~14 RPM)

        # Auto-detect provider from available API keys
        if provider is None:
            if os.getenv("GROQ_API_KEY"):
                provider = "groq"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
                provider = "gemini"
            else:
                raise ValueError(
                    "No API key found. Set GROQ_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in .env file.\n"
                    "Create a .env file in the project root with:\n"
                    "  GROQ_API_KEY=gsk_...\n"
                    "  or\n"
                    "  OPENAI_API_KEY=sk-...\n"
                    "  or\n"
                    "  GEMINI_API_KEY=AI...\n"
                )

        self.provider = provider

        if provider == "openai":
            self._init_openai(model or "gpt-4o-mini")
        elif provider == "groq":
            self._init_groq(model or "allam-2-7b")
        elif provider == "gemini":
            self._init_gemini(model or "gemini-3.5-flash-lite")
        else:
            raise ValueError(f"Unknown provider: {provider}")

        print(f"[llm] Initialized {self.provider} client with model: {self.model}")

    def _init_openai(self, model: str):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def _init_groq(self, model: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1",
        )
        self.model = model

    def _init_gemini(self, model: str):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, system_prompt: str = None, json_mode: bool = False) -> str:
        """
        Generate a completion from the LLM.

        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            json_mode: If True, request JSON output mode.

        Returns:
            The generated text response.
        """
        # Proactive delay to stay under rate limits (Groq free tier: 30 RPM)
        if self._call_delay > 0:
            time.sleep(self._call_delay)

        for attempt in range(self.max_retries):
            try:
                if self.provider in ("openai", "groq"):
                    result = self._generate_openai(prompt, system_prompt, json_mode)
                elif self.provider == "gemini":
                    result = self._generate_gemini(prompt, system_prompt, json_mode)
                else:
                    raise ValueError(f"Unknown provider: {self.provider}")
                return result or ""
            except Exception as e:
                err_str = str(e).lower()
                # Rate limit — wait longer
                if "rate_limit" in err_str or "429" in err_str or "rate limit" in err_str:
                    wait = max(self.retry_delay * (2 ** attempt), 10.0)
                    print(f"[llm] Rate limited. Waiting {wait:.0f}s before retry {attempt + 1}/{self.max_retries}...")
                    time.sleep(wait)
                elif attempt < self.max_retries - 1:
                    wait = self.retry_delay * (2 ** attempt)
                    print(f"[llm] Retry {attempt + 1}/{self.max_retries} after error: {e}. Waiting {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    print(f"[llm] All retries exhausted: {e}")
                    return ""  # Return empty string instead of None to prevent crashes

    def _generate_openai(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _generate_gemini(self, prompt: str, system_prompt: str, json_mode: bool) -> str:
        from google.genai import types

        full_prompt = ""
        if system_prompt:
            full_prompt = f"System instructions: {system_prompt}\n\n"
        full_prompt += prompt

        if json_mode:
            full_prompt += "\n\nIMPORTANT: Respond ONLY with valid JSON. No markdown, no code fences."

        config = types.GenerateContentConfig(
            temperature=self.temperature,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=full_prompt,
            config=config,
        )
        return response.text

    def generate_json(self, prompt: str, system_prompt: str = None) -> dict:
        """Generate a response and parse it as JSON."""
        text = self.generate(prompt, system_prompt, json_mode=True)
        # Try to extract JSON from the response
        text = text.strip()
        # Remove markdown code fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        # Remove thinking tags if present (Qwen models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Find JSON object or array
        match = re.search(r'[\[{].*[\]}]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from response: {text[:200]}")


# ─── Convenience ────────────────────────────────────────────────────────────

_default_client = None

def get_client(**kwargs) -> LLMClient:
    """Get or create the default LLM client."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient(**kwargs)
    return _default_client

