"""Generate questions from LLMs.

Supports Ollama (local), OpenAI, Google Gemini, and Together.ai backends.
Adapted from CogBench V1 with passage-based generation for V2.
"""

import os
import json
import time
import requests
from datetime import datetime
from typing import Optional

from cogbenchv2.config import (
    BLOOM_LEVELS, SUBJECTS, LOCAL_MODELS, API_MODELS, TOGETHER_MODELS,
    ALL_MODELS, TEMPERATURE, MAX_TOKENS, OLLAMA_URL, OLLAMA_TIMEOUT,
    RESULTS_DIR, ADVERSARIAL_PAIRINGS,
    GOOGLE_API_KEY, TOGETHER_API_KEY, OPENAI_API_KEY,
)
from cogbenchv2.generation.prompts import (
    build_standard_prompt, build_adversarial_prompt, get_adversarial_pairing,
)
from cogbenchv2.generation.extract import extract_qa
from cogbenchv2.passages.processor import load_all_passages


# ─── LLM backends ─────────────────────────────────────────────────────────────

def generate_ollama(model: str, prompt: str, temperature: float = TEMPERATURE,
                    max_tokens: int = MAX_TOKENS, retries: int = 3) -> dict:
    """Generate from Ollama. Returns {text, latency_ms, error}."""
    t0 = time.time()
    last_error = None

    # Reasoning models need reduced context to fit in VRAM and longer timeout
    is_reasoning = "deepseek-r1" in model or "r1" in model.split(":")[-1]
    options = {"temperature": temperature, "num_predict": max_tokens}
    if is_reasoning:
        options["num_ctx"] = 2048  # Cap context to prevent OOM on 11GB GPUs
    timeout = OLLAMA_TIMEOUT * 2 if is_reasoning else OLLAMA_TIMEOUT

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": options,
            }, timeout=timeout)
            resp.raise_for_status()
            text = resp.json()["response"].strip()
            # Strip <think>...</think> blocks from reasoning models
            if is_reasoning and "<think>" in text:
                import re
                text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
            return {"text": text, "latency_ms": (time.time() - t0) * 1000, "error": None}
        except Exception as e:
            last_error = e
            if attempt < retries - 1:
                # Check if Ollama crashed (connection refused) and wait for auto-restart
                if "Connection" in str(e) or "Max retries" in str(e):
                    import subprocess
                    print(f"  [!] Ollama connection lost, waiting for restart... (attempt {attempt+1})")
                    time.sleep(15)  # systemd RestartSec=3, give it time to reload model
                    # Verify Ollama is back
                    for _ in range(6):
                        try:
                            requests.get("http://localhost:11434/api/tags", timeout=5)
                            print(f"  [+] Ollama back online")
                            break
                        except Exception:
                            time.sleep(10)
                else:
                    time.sleep(5 * (attempt + 1))
    return {"text": None, "latency_ms": (time.time() - t0) * 1000, "error": str(last_error)}


def generate_openai(model: str, prompt: str, temperature: float = TEMPERATURE,
                    max_tokens: int = MAX_TOKENS) -> dict:
    """Generate from OpenAI API."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY", "") or OPENAI_API_KEY
    client = OpenAI(api_key=api_key)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content.strip()
        return {"text": text, "latency_ms": (time.time() - t0) * 1000, "error": None}
    except Exception as e:
        return {"text": None, "latency_ms": (time.time() - t0) * 1000, "error": str(e)}


def generate_gemini(model: str, prompt: str, temperature: float = TEMPERATURE,
                    max_tokens: int = MAX_TOKENS) -> dict:
    """Generate from Google Gemini API."""
    import google.generativeai as genai
    import warnings
    warnings.filterwarnings("ignore")

    api_key = os.environ.get("GOOGLE_API_KEY", "") or GOOGLE_API_KEY
    genai.configure(api_key=api_key)

    t0 = time.time()
    try:
        gen_model = genai.GenerativeModel(model)
        response = gen_model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        text = response.text.strip()
        return {"text": text, "latency_ms": (time.time() - t0) * 1000, "error": None}
    except Exception as e:
        return {"text": None, "latency_ms": (time.time() - t0) * 1000, "error": str(e)}


def generate_together(model: str, prompt: str, temperature: float = TEMPERATURE,
                      max_tokens: int = MAX_TOKENS) -> dict:
    """Generate from Together.ai API."""
    from together import Together
    api_key = os.environ.get("TOGETHER_API_KEY", "") or TOGETHER_API_KEY
    client = Together(api_key=api_key)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.choices[0].message.content.strip()
        return {"text": text, "latency_ms": (time.time() - t0) * 1000, "error": None}
    except Exception as e:
        return {"text": None, "latency_ms": (time.time() - t0) * 1000, "error": str(e)}


def unload_ollama_model(model: str):
    """Unload Ollama model from GPU memory."""
    try:
        requests.post(OLLAMA_URL, json={
            "model": model, "prompt": "", "keep_alive": 0,
        }, timeout=10)
    except Exception:
        pass


def _get_backend(model: str):
    """Determine which backend to use for a model."""
    if model.startswith("gemini-"):
        return "google"
    if model in TOGETHER_MODELS:
        return "together"
    if model in API_MODELS:
        return API_MODELS[model]["backend"]
    return "ollama"


def _call_llm(model: str, prompt: str) -> dict:
    """Route to the correct backend."""
    backend = _get_backend(model)
    if backend == "google":
        return generate_gemini(model, prompt)
    elif backend == "together":
        return generate_together(model, prompt)
    elif backend == "openai":
        return generate_openai(model, prompt)
    else:
        return generate_ollama(model, prompt)


# ─── Main generation loop ─────────────────────────────────────────────────────

def generate_for_model(model: str, passages: list, mode: str = "standard",
                       output_dir: str = None) -> list:
    """Generate questions for all (passage, level) combos with one model.

    Args:
        model: Model identifier
        passages: List of passage dicts
        mode: "standard" or "adversarial"
        output_dir: Where to save results

    Returns:
        List of generation records
    """
    output_dir = output_dir or RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    model_safe = model.replace(":", "_").replace(".", "_").replace("/", "_")
    save_path = os.path.join(output_dir, f"gen_{model_safe}_{mode}.json")

    # Resume from existing (skip errored records so they get retried)
    existing = {}
    if os.path.exists(save_path):
        with open(save_path) as f:
            data = json.load(f)
            for r in data.get("generations", []):
                if r.get("error"):
                    continue  # retry errored records
                key = f"{r['passage_id']}_{r['level']}_{r.get('mode', 'standard')}"
                existing[key] = r
        print(f"  Resuming from {len(existing)} successful generations (skipping errors)")

    if mode == "standard":
        levels = sorted(BLOOM_LEVELS.keys())  # 1-6
    else:
        # Adversarial: use ADVERSARIAL_PAIRINGS
        levels = [tgt for tgt, _ in ADVERSARIAL_PAIRINGS]

    total = len(passages) * len(levels)
    generations = list(existing.values())
    done = len(existing)
    n_errors = 0
    t_start = time.time()

    print(f"\n  Generating: {model} ({mode} mode)")
    print(f"  {len(passages)} passages x {len(levels)} levels = {total} prompts")

    # Warm up Ollama
    backend = _get_backend(model)
    if done == 0 and backend == "ollama":
        print(f"  Warming up {model}...")
        generate_ollama(model, "Hello", temperature=0)

    for passage in passages:
        for level in levels:
            key = f"{passage['passage_id']}_{level}_{mode}"
            if key in existing:
                continue

            # Build prompt
            if mode == "standard":
                prompt = build_standard_prompt(passage["text"], level)
                vocab_level = None
            else:
                vocab_level = get_adversarial_pairing(level)
                if vocab_level is None:
                    continue
                prompt = build_adversarial_prompt(passage["text"], level, vocab_level)

            # Generate
            result = _call_llm(model, prompt)
            question, answer = extract_qa(result["text"])

            record = {
                "model": model,
                "mode": mode,
                "passage_id": passage["passage_id"],
                "subject": passage["subject"],
                "level": level,
                "level_name": BLOOM_LEVELS[level],
                "vocab_level": vocab_level,
                "prompt": prompt,
                "raw_response": result["text"],
                "question": question,
                "answer": answer,
                "latency_ms": result["latency_ms"],
                "error": result["error"],
                "timestamp": datetime.now().isoformat(),
            }
            generations.append(record)

            if result["error"]:
                n_errors += 1

            done += 1

            # Progress
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{model}] {done}/{total} "
                      f"({done/total*100:.0f}%) | "
                      f"{rate:.1f} gen/s | ETA: {eta:.0f}s | "
                      f"Errors: {n_errors}")

                # Save incrementally
                _save(save_path, model, mode, generations)

    # Final save
    _save(save_path, model, mode, generations)

    elapsed = time.time() - t_start
    print(f"  [{model}] Done: {done} generations in {elapsed:.0f}s, {n_errors} errors")
    print(f"  Saved to: {save_path}")

    # Unload Ollama model
    if backend == "ollama":
        print(f"  Unloading {model}...")
        unload_ollama_model(model)
        time.sleep(2)

    return generations


def _save(path, model, mode, generations):
    """Save generation results."""
    with open(path, "w") as f:
        json.dump({
            "model": model,
            "mode": mode,
            "n_generations": len(generations),
            "timestamp": datetime.now().isoformat(),
            "generations": generations,
        }, f, indent=2)
