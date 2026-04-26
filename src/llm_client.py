import os
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError


BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


# Verified free models on OpenRouter in priority order
FREE_MODEL_FALLBACKS = [
    "mistralai/mistral-7b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-3-1b-it:free",
    "google/gemma-3-4b-it:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "openchat/openchat-7b:free",
    "huggingfaceh4/zephyr-7b-beta:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
    "openai/gpt-oss-120b:free"
]


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Talent Scout Agent"
    }
)


def chat_with_fallback(
    messages: list,
    response_format: dict = None,
    temperature: float = 0.0,
    max_retries: int = 1,
    retry_delay: float = 10.0
) -> str:
    """
    Tries each model in FREE_MODEL_FALLBACKS in order.
    Retries on RateLimitError before falling to next model.
    Skips models that return None or empty content.
    Returns raw content string.
    Raises RuntimeError if all models fail.
    """
    last_error = None

    for model in FREE_MODEL_FALLBACKS:
        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                }
                if response_format:
                    kwargs["response_format"] = response_format

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                # Some free models return None when they produce
                # a tool_call or malformed completion instead of text
                if content is None:
                    print(f"[LLM] Model {model} returned None content. Skipping to next model.")
                    break

                # Guard against whitespace-only responses
                if not content.strip():
                    print(f"[LLM] Model {model} returned empty string. Skipping to next model.")
                    break

                print(f"[LLM] Success with model: {model}")
                return content

            except RateLimitError as e:
                last_error = e
                print(f"[LLM] Rate limit on {model}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)

            except Exception as e:
                last_error = e
                print(f"[LLM] Non-rate-limit error on {model}: {e}")
                break  # Skip to next model immediately

        print(f"[LLM] Moving to next fallback model...")

    raise RuntimeError(
        f"All LLM models failed or are rate-limited. Last error: {last_error}"
    )