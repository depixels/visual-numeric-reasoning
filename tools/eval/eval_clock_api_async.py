#!/usr/bin/env python3
"""Async high-concurrency version of eval_clock_api.py.

Uses asyncio + httpx to call multiple samples concurrently.
Concurrency is controlled by --concurrency (default 16).
"""

import argparse
import asyncio
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from eval_common import (
    compute_metrics,
    extract_gt_label,
    extract_image_relpath,
    finalize_prediction_row,
    infer_split,
    load_jsonl,
    safe_div,
    write_csv,
    write_json,
    write_jsonl,
)
from parse_time import parse_hhmm, parse_hhmmss


DEFAULT_PROMPT = (
    "Read the exact time shown on this analog clock. "
    "Answer in HH:MM:SS format." 
)

# ─────────────────────────── Provider Configs ────────────────────────────────

PROVIDER_BASE_URLS = {
    "vllm_qwen": None,          # user-specified
    "azure_gpt": None,          # user-specified
    "qwen_dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "gemini_3_pro": "https://api.novita.ai/openai",
}

PROVIDER_MODELS = {
    "gemini_3_pro": "pa/gemini-3-pro-preview",
}

# ─────────────────────────── Argument Parsing ────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Async eval of analog-clock API models")
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--images_root", default=None)
    p.add_argument(
        "--provider",
        choices=["vllm_qwen", "gemini_3_pro", "azure_gpt", "qwen_dashscope"],
        required=True,
    )
    p.add_argument("--model", default="/data/hyz/workspace/hf/Qwen3-VL-4B-Instruct")
    p.add_argument("--base_url", default=None)
    p.add_argument("--api_key", required=True)
    p.add_argument("--timeout", type=int, default=120)

    p.add_argument("--developer_prompt", default=None)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--user_prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_sleep", type=float, default=1.0)

    # ── Concurrency ──
    p.add_argument(
        "--concurrency", type=int, default=16,
        help="Max parallel API calls. Tune to stay under RPM/TPM limits."
    )
    p.add_argument(
        "--rpm_limit", type=int, default=0,
        help="Requests-per-minute hard cap (0 = disabled). "
             "When set, adds inter-request delay to respect the limit."
    )

    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=50)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--pred_json", default="predictions.json")
    p.add_argument("--pred_jsonl", default="per_sample_results.jsonl")
    p.add_argument("--pred_csv", default="per_sample_results.csv")
    p.add_argument("--metrics_json", default="metrics.json")
    return p.parse_args()


# ─────────────────────────── Image Helpers ───────────────────────────────────

def _image_to_data_url(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = "image/png"
    if ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    elif ext == ".webp":
        mime = "image/webp"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ─────────────────────────── Async API Caller ────────────────────────────────

def _build_messages(
    image_data_url: str,
    user_prompt: str,
    developer_prompt: Optional[str],
    system_prompt: Optional[str],
    qwen_style: bool = False,   # DashScope needs list-wrapped system content
) -> List[Dict[str, Any]]:
    user_content: List[Dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": user_prompt},
    ]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]
    if developer_prompt:
        dev_content = (
            [{"type": "text", "text": developer_prompt}] if qwen_style else developer_prompt
        )
        messages = [{"role": "developer", "content": dev_content}] + messages
    if system_prompt:
        sys_content = (
            [{"type": "text", "text": system_prompt}] if qwen_style else system_prompt
        )
        messages = [{"role": "system", "content": sys_content}] + messages
    return messages


async def _async_chat(
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    extra_body: Optional[Dict] = None,
) -> str:
    """Raw async POST to OpenAI-compatible /chat/completions."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    if extra_body:
        payload.update(extra_body)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            out = data["choices"][0]["message"]["content"]
            if isinstance(out, str):
                return out
            if isinstance(out, list):
                return "".join(p.get("text", "") for p in out if isinstance(p, dict))
            return str(out)
        except Exception as e:
            last_exc = e
            if attempt == max_retries - 1:
                raise
            wait = retry_sleep * (2 ** attempt)   # exponential back-off
            print(f"[warn] attempt {attempt + 1}/{max_retries} failed: {e}. retry in {wait:.1f}s")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable") from last_exc


# ─────────────────────────── Per-Sample Worker ───────────────────────────────

async def _process_one(
    *,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    args: argparse.Namespace,
    base_url: str,
    model: str,
    row: Dict[str, Any],
    idx: int,
    images_root: str,
    results: List[Optional[Dict[str, Any]]],
    counter: Dict[str, int],
    lock: asyncio.Lock,
    total: int,
) -> None:
    sample_id = row.get("id", f"sample_{idx:06d}")
    image_rel = extract_image_relpath(row)
    image_abs = image_rel if os.path.isabs(image_rel) else os.path.join(images_root, image_rel)
    gt_label = extract_gt_label(row)
    split = infer_split(row, args.gt_jsonl)

    call_error = None
    response_text = ""
    t0 = time.time()

    async with sem:   # ← concurrency gate
        try:
            # I/O-bound file read — run in thread pool to avoid blocking event loop
            image_data_url = await asyncio.get_event_loop().run_in_executor(
                None, _image_to_data_url, image_abs
            )

            qwen_style = args.provider == "qwen_dashscope"
            messages = _build_messages(
                image_data_url,
                args.user_prompt,
                args.developer_prompt,
                args.system_prompt,
                qwen_style=qwen_style,
            )

            extra_body = None
            if args.provider == "gemini_3_pro":
                extra_body = {
                    "extra_body": {
                        "google": {
                            "thinkingConfig": {"thinking_level": "high"}
                        }
                    }
                }

            response_text = await _async_chat(
                client=client,
                base_url=base_url,
                api_key=args.api_key,
                model=model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retries=args.max_retries,
                retry_sleep=args.retry_sleep,
                extra_body=extra_body,
            )
        except Exception as e:
            call_error = str(e)

    elapsed = time.time() - t0

    pred_seconds_total = parse_hhmmss(response_text) if response_text else None
    pred_minutes = (
        pred_seconds_total // 60
        if pred_seconds_total is not None
        else parse_hhmm(response_text) if response_text else None
    )

    pred_row = finalize_prediction_row(
        sample_id=sample_id,
        split=split,
        image_rel=image_rel,
        gt_label=gt_label,
        raw_output=response_text,
        pred_minutes=pred_minutes,
        pred_seconds_total=pred_seconds_total,
        latency_sec=elapsed,
        error=call_error,
    )

    async with lock:
        results[idx] = pred_row
        counter["done"] += 1
        done = counter["done"]
        if done % 10 == 0 or done == total:
            print(f"[progress] {done}/{total} done")


# ─────────────────────────── Main ────────────────────────────────────────────

async def _async_main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    def _out(name: str) -> str:
        return name if os.path.isabs(name) else os.path.join(args.output_dir, name)

    pred_json_path = _out(args.pred_json)
    pred_jsonl_path = _out(args.pred_jsonl)
    pred_csv_path = _out(args.pred_csv)
    metrics_json_path = _out(args.metrics_json)

    rows = load_jsonl(args.gt_jsonl)
    end = None if args.limit is None else args.start_index + args.limit
    rows = rows[args.start_index:end]
    total = len(rows)

    images_root = args.images_root or os.path.dirname(args.gt_jsonl)

    # Resolve base_url and model
    base_url = args.base_url or PROVIDER_BASE_URLS.get(args.provider) or "http://127.0.0.1:8001/v1"
    model = PROVIDER_MODELS.get(args.provider, args.model)

    print(f"Provider : {args.provider}")
    print(f"Model    : {model}")
    print(f"Base URL : {base_url}")
    print(f"Samples  : {total}")
    print(f"Concurrency: {args.concurrency}")

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    results: List[Optional[Dict[str, Any]]] = [None] * total
    counter: Dict[str, int] = {"done": 0}

    # RPM throttle: optional inter-task delay
    rpm_delay = (60.0 / args.rpm_limit) if args.rpm_limit > 0 else 0.0

    start_t = time.time()

    # httpx async client — shared across all coroutines
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = []
        for i, row in enumerate(rows):
            coro = _process_one(
                sem=sem,
                client=client,
                args=args,
                base_url=base_url,
                model=model,
                row=row,
                idx=i,
                images_root=images_root,
                results=results,
                counter=counter,
                lock=lock,
                total=total,
            )
            tasks.append(asyncio.ensure_future(coro))
            if rpm_delay > 0:
                await asyncio.sleep(rpm_delay)   # gentle rate limiting

        # Periodic save while tasks run
        save_task = asyncio.ensure_future(
            _periodic_save(
                results=results,
                counter=counter,
                lock=lock,
                total=total,
                save_every=args.save_every,
                pred_json_path=pred_json_path,
                pred_jsonl_path=pred_jsonl_path,
                pred_csv_path=pred_csv_path,
            )
        )

        await asyncio.gather(*tasks)
        save_task.cancel()

    # Final save
    preds = [r for r in results if r is not None]
    write_json(pred_json_path, preds)
    write_jsonl(pred_jsonl_path, preds)
    write_csv(pred_csv_path, preds)

    metrics = compute_metrics(preds)
    metrics["provider"] = args.provider
    metrics["model"] = model
    metrics["num_samples"] = total
    metrics["elapsed_sec_total"] = time.time() - start_t
    metrics["avg_latency_sec"] = safe_div(
        sum(float(r["latency_sec"]) for r in preds if isinstance(r.get("latency_sec"), (int, float))),
        len(preds),
    )
    metrics["pred_json"] = pred_json_path
    metrics["pred_jsonl"] = pred_jsonl_path
    metrics["pred_csv"] = pred_csv_path
    metrics["gt_jsonl"] = args.gt_jsonl
    metrics["eval_protocol"] = {
        "developer_prompt": args.developer_prompt,
        "system_prompt": args.system_prompt,
        "user_prompt": args.user_prompt,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
    }

    write_json(metrics_json_path, metrics)

    print("\nEvaluation done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


async def _periodic_save(
    *,
    results: List[Optional[Dict]],
    counter: Dict[str, int],
    lock: asyncio.Lock,
    total: int,
    save_every: int,
    pred_json_path: str,
    pred_jsonl_path: str,
    pred_csv_path: str,
) -> None:
    last_saved = 0
    while True:
        await asyncio.sleep(5)
        async with lock:
            done = counter["done"]
        if done - last_saved >= save_every or done == total:
            preds = [r for r in results if r is not None]
            write_json(pred_json_path, preds)
            write_jsonl(pred_jsonl_path, preds)
            write_csv(pred_csv_path, preds)
            last_saved = done
            print(f"[checkpoint] saved {len(preds)} results")


def main() -> None:
    args = _parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()