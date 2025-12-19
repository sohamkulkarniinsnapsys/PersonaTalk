import os
import logging
import json
import time
import asyncio
from typing import List, Dict, Any

import requests

logger = logging.getLogger(__name__)


class GroqLLM:
    """Minimal Groq LLM adapter implementing the project's expected interface.

    Expects env vars:
      - GROQ_API_KEY
      - GROQ_API_URL (optional; default is groq's public endpoint placeholder)
      - GROQ_MODEL (optional; default chosen below)

    This adapter is intentionally small: it posts a single prompt string and
    returns the model output. The ConversationController/Orchestrator builds
    structured prompts (including evaluation instructions) before calling this.
    """

    def __init__(self):
        self.api_key = os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is required for GroqLLM in live mode")

        # Use a base host without a path so we can try multiple /v1 shapes safely.
        # Allow overriding via GROQ_API_URL if set (should be like https://api.groq.com or https://api.groq.ai)
        self.api_url = os.environ.get("GROQ_API_URL", "https://api.groq.com")
        # Optional: allow callers to provide the exact model endpoint (including path)
        # Example: https://api.groq.com/v1/models/mixtral-8x7b:predict
        self.model_endpoint = os.environ.get('GROQ_MODEL_ENDPOINT')
        # Default model—override via env (updated from decommissioned mixtral default)
        self.model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

        # Basic session
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        # Track how many failure responses we've logged in full to avoid log noise
        self._full_failure_logs = 0
        self._max_full_failure_logs = int(os.environ.get('GROQ_FULL_FAILURE_LOGS', 3))

    def _build_prompt_text(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        # Flatten messages to a single instruction text. The Orchestrator uses
        # a messages list shaped like [{'role':'user'|'assistant','content': '...'}]
        pieces = []
        if system_prompt:
            pieces.append(f"SYSTEM: {system_prompt}\n")
        for m in messages:
            role = m.get('role', 'user').upper()
            pieces.append(f"{role}: {m.get('content','')}\n")
        return "\n".join(pieces)

    async def _post(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Try several common Groq endpoint shapes to be tolerant of API variations.
        # Now async: wraps all requests.post calls in asyncio.to_thread to avoid event loop blocking.
        candidate_paths = [
            # Prefer the OpenAI-compatible endpoint Groq documents
            "/openai/v1/chat/completions",
            "/openai/v1/completions",
            "/chat/completions",
            # Groq model endpoints (may 404 depending on deployment)
            f"/v1/models/{self.model}/generate",
            f"/v1/models/{self.model}:predict",
            f"/v1/models/{self.model}:generate",
            f"/models/{self.model}/generate",
            f"/models/{self.model}:predict",
            # Fallback generic endpoints
            "/v1/generate",
            "/v1/completions",
        ]

        # Payload templates in order of preference for Groq's OpenAI-compatible API
        # Most common first: chat/completions format (used by GROQ_MODEL_ENDPOINT)
        payload_templates = [
            {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature},  # chat/completions
            {"model": self.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": temperature},  # completions
            {"model": self.model, "input": prompt, "max_tokens": max_tokens, "temperature": temperature},  # generate
        ]

        last_err = None
        # If a specific model endpoint is provided, try it first and return early.
        if self.model_endpoint:
            try:
                logger.debug(f"Groq using explicit model endpoint: {self.model_endpoint}")
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                r = await asyncio.to_thread(
                    self.session.post,
                    self.model_endpoint,
                    json=payload,
                    timeout=30,
                )
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except Exception:
                        data = r.text
                    return self._extract_text_from_response(data)
                else:
                    # Log response and body; limit verbose body logs to first few failures
                    logger.warning(f"Groq explicit endpoint returned {r.status_code}: {r.text}")
                    if self._full_failure_logs < self._max_full_failure_logs:
                        logger.warning(f"Groq explicit endpoint full response headers: {r.headers}")
                        logger.warning(f"Groq explicit endpoint full response body: {r.text}")
                        self._full_failure_logs += 1
            except Exception as e:
                logger.warning(f"Groq request to explicit endpoint failed: {e}")
                if self._full_failure_logs < self._max_full_failure_logs:
                    logger.exception("Exception contacting explicit Groq endpoint")
                    self._full_failure_logs += 1

        # Try multiple base hosts: configured one first, then known alternate
        alternate_base = os.environ.get('GROQ_API_URL_ALT')
        base_urls = [self.api_url.rstrip('/')]
        if alternate_base:
            base_urls.append(alternate_base.rstrip('/'))

        for base in base_urls:
            for path in candidate_paths:
                url = base + path
                for payload in payload_templates:
                    # Basic retry per endpoint/payload
                    for attempt in range(2):
                        try:
                            # Adapt payload shape for endpoints that expect a 'model' key
                            payload_to_send = payload.copy()
                            if '/openai/v1/chat/completions' in url or url.endswith('/chat/completions'):
                                payload_to_send = {
                                    "model": self.model,
                                    "messages": [
                                        {"role": "user", "content": prompt}
                                    ],
                                    "temperature": temperature,
                                    # map max tokens if API supports
                                    "max_tokens": max_tokens,
                                    # Ensure no premature stop sequences
                                    "stop": None,
                                }
                            elif '/openai/v1/completions' in url:
                                payload_to_send = {
                                    "model": self.model,
                                    "prompt": prompt,
                                    "temperature": temperature,
                                    "max_tokens": max_tokens,
                                    "stop": None,
                                }
                            elif '/v1/generate' in url or url.endswith('/v1/generate'):
                                payload_to_send = {
                                    "model": self.model,
                                    "input": prompt,
                                    "max_tokens": max_tokens,
                                    "temperature": temperature,
                                    "stop": None,
                                }
                            elif '/v1/completions' in url or url.endswith('/v1/completions'):
                                payload_to_send = {
                                    "model": self.model,
                                    "prompt": prompt,
                                    "max_tokens": max_tokens,
                                    "temperature": temperature,
                                    "stop": None,
                                }
                            else:
                                # ensure payload includes prompt when possible
                                if 'prompt' not in payload_to_send and 'input' not in payload_to_send and 'messages' not in payload_to_send:
                                    payload_to_send['prompt'] = prompt
                                if 'max_tokens' not in payload_to_send:
                                    payload_to_send['max_tokens'] = max_tokens
                                if 'stop' not in payload_to_send:
                                    payload_to_send['stop'] = None

                            logger.debug(f"Groq trying {url} max_tokens={max_tokens} payload keys: {list(payload_to_send.keys())} (attempt {attempt+1})")
                            # Run blocking network call in worker thread to avoid blocking event loop
                            r = await asyncio.to_thread(
                                self.session.post,
                                url,
                                json=payload_to_send,
                                timeout=30,
                            )
                            if r.status_code == 200:
                                try:
                                    data = r.json()
                                except Exception:
                                    data = r.text
                                text = self._extract_text_from_response(data)
                                logger.info(f"Groq endpoint succeeded: {url}, response length: {len(text)}")
                                return text
                            elif r.status_code == 400 and "model_decommissioned" in r.text:
                                # Stop early on explicit decommission notice to avoid noisy retries
                                logger.warning(f"Groq model appears decommissioned: {self.model}. Body: {r.text}")
                                raise RuntimeError(f"Groq model {self.model} is decommissioned; set GROQ_MODEL to a supported model.")
                            else:
                                # Log more visible warning to aid debugging and capture full body for first N failures
                                logger.warning(f"Groq attempt to {url} returned {r.status_code}: {r.text}")
                                if self._full_failure_logs < self._max_full_failure_logs:
                                    logger.warning(f"Groq attempt full response headers for {url}: {r.headers}")
                                    logger.warning(f"Groq attempt full response body for {url}: {r.text}")
                                    self._full_failure_logs += 1
                        except Exception as e:
                            last_err = e
                            logger.warning(f"Groq request failed to {url} (attempt {attempt+1}): {e}")
                            if self._full_failure_logs < self._max_full_failure_logs:
                                logger.exception(f"Exception contacting Groq at {url}")
                                self._full_failure_logs += 1
                        await asyncio.sleep(0.2)

        # If we get here, none of the endpoint/payload combos worked
        raise RuntimeError(f"Groq API failed after trying endpoints; last error: {last_err}")

    def _extract_text_from_response(self, data: Dict[str, Any]) -> str:
        # Common Groq-like shapes: check several possible keys
        if not data:
            return ""
        # 1) direct 'text'
        if isinstance(data, str):
            return data
        if 'text' in data and isinstance(data['text'], str):
            return data['text']
        # 2) outputs -> list -> content
        if 'outputs' in data and isinstance(data['outputs'], list):
            out = data['outputs'][0]
            if isinstance(out, dict):
                # Common Groq v1: outputs[0].content[0].text or outputs[0].content[0]
                content = out.get('content')
                if isinstance(content, list) and len(content) > 0:
                    first = content[0]
                    if isinstance(first, dict) and 'text' in first:
                        return first['text']
                    if isinstance(first, str):
                        return first
        # 3) OpenAI-compatible chat completions
        if 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
            first = data['choices'][0]
            if isinstance(first, dict):
                message = first.get('message')
                if isinstance(message, dict) and isinstance(message.get('content'), str):
                    return message['content']
                if 'text' in first and isinstance(first['text'], str):
                    return first['text']
        # 4) other completions/generations
        for key in ('choices', 'generations', 'generation'):
            if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                first = data[key][0]
                if isinstance(first, dict) and 'text' in first:
                    return first['text']
                if isinstance(first, str):
                    return first
        # 5) output field
        if 'output' in data:
            out = data['output']
            if isinstance(out, dict) and 'text' in out:
                return out['text']
            try:
                return json.dumps(out)
            except Exception:
                return str(out)
        # fallback: stringify
        try:
            return json.dumps(data)
        except Exception:
            return str(data)

    async def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> Dict[str, Any]:
        """Primary method used by the orchestrator.

        Returns a dict: { 'text': str, 'should_tts': bool }
        """
        try:
            prompt = self._build_prompt_text(messages, system_prompt)
            # Choose conservative temperature for deterministic behavior
            # Use HIGHER max_tokens (4096) to ensure full responses aren't truncated mid-sentence
            # Groq models support up to 4096 tokens and we need enough budget for detailed explanations
            logger.info(f"🧠 GroqLLM: Calling _post with max_tokens=4096 (for complete answers)")
            text = await self._post(prompt, max_tokens=4096, temperature=float(os.environ.get('GROQ_TEMP', 0.2)))
            logger.info(f"🧠 GroqLLM: Received response ({len(text)} chars): '{text[:150]}'...")
            # Groq output may already be a JSON string if evaluation prompt asked for JSON.
            # We return raw string or parsed JSON to the orchestrator which handles it.
            # Try to parse JSON if it looks like JSON
            try:
                parsed = json.loads(text)
                return parsed if isinstance(parsed, dict) else {"text": str(parsed), "should_tts": True}
            except Exception:
                return {"text": text, "should_tts": True}
        except Exception as e:
            logger.error(f"GroqLLM.generate_response error: {e}")
            return {"text": "", "should_tts": False}
