#!/usr/bin/env python3
"""Evaluate analog-clock readout via API models and save predictions/metrics.

Examples:
  # vLLM served Qwen3-VL-4B (OpenAI-compatible API)
  python tools/eval/eval_clock_api.py \
    --gt_jsonl /data/rege_ood/clean/samples.jsonl \
    --images_root /data/rege_ood/clean \
    --provider vllm_qwen \
    --model Qwen/Qwen3-VL-4B-Instruct \
    --base_url http://127.0.0.1:8000/v1 \
    --api_key EMPTY \
    --output_dir /data/rege_ood_eval/qwen_clean

  # Gemini 3 Pro via Novita API
  python tools/eval/eval_clock_api.py \
    --gt_jsonl /data/rege_ood/clean/samples.jsonl \
    --images_root /data/rege_ood/clean \
    --provider gemini_3_pro \
    --api_key xxx \
    --output_dir /data/rege_ood_eval/gemini_clean

  # Azure OpenAI (OpenAI-compatible endpoint)
  python tools/eval/eval_clock_api.py \
    --gt_jsonl /data/rege_ood/clean/samples.jsonl \
    --images_root /data/rege_ood/clean \
    --provider azure_gpt \
    --base_url https://<resource>.openai.azure.com/openai/v1/ \
    --api_key <AZURE_OPENAI_API_KEY> \
    --model <deployment_name> \
    --output_dir /data/rege_ood_eval/azure_gpt_clean

  # Qwen VL via DashScope OpenAI-compatible API
  python tools/eval/eval_clock_api.py \
    --gt_jsonl /data/rege_ood/clean/samples.jsonl \
    --images_root /data/rege_ood/clean \
    --provider qwen_dashscope \
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api_key <DASHSCOPE_API_KEY> \
    --model qwen-vl-max-latest \
    --output_dir /data/rege_ood_eval/qwen_api_clean
"""

import argparse
import base64
import json
import os
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI

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

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate analog-clock API model outputs")
    p.add_argument("--gt_jsonl", required=True)
    p.add_argument("--images_root", default=None)
    p.add_argument(
        "--provider",
        choices=["vllm_qwen", "gemini_3_pro", "azure_gpt", "qwen_dashscope"],
        required=True,
    )

    p.add_argument("--model", default="/data/hyz/workspace/hf/Qwen3-VL-4B-Instruct")
    p.add_argument("--base_url", default="http://127.0.0.1:8001/v1")
    p.add_argument("--api_key", required=True)
    p.add_argument("--timeout", type=int, default=3600)

    p.add_argument("--developer_prompt", default=None)
    p.add_argument("--system_prompt", default=None)
    p.add_argument("--user_prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--max_retries", type=int, default=5)
    p.add_argument("--retry_sleep", type=float, default=0.8)

    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--save_every", type=int, default=20)

    p.add_argument("--output_dir", required=True)
    p.add_argument("--pred_json", default="predictions.json")
    p.add_argument("--pred_jsonl", default="per_sample_results.jsonl")
    p.add_argument("--pred_csv", default="per_sample_results.csv")
    p.add_argument("--metrics_json", default="metrics.json")
    return p.parse_args()


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


def _new_client(base_url: str, api_key: str, timeout: int) -> OpenAI:
    # Disable SDK internal retries; we handle retries explicitly.
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=0)


def _call_openai_vision(
    client: OpenAI,
    model: str,
    image_data_url: str,
    user_prompt: str,
    developer_prompt: Optional[str],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
    provider_name: str,
) -> str:
    content = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": user_prompt},
    ]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
    if developer_prompt:
        messages = [{"role": "developer", "content": developer_prompt}] + messages
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            out = rsp.choices[0].message.content
            if isinstance(out, str):
                return out
            if isinstance(out, list):
                return "".join(part.get("text", "") for part in out if isinstance(part, dict))
            return str(out)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"[warn] {provider_name} call failed attempt={attempt + 1}/{max_retries}: {e}")
            time.sleep(retry_sleep)
    return ""


def _call_gemini_3_pro(
    api_key: str,
    image_data_url: str,
    user_prompt: str,
    developer_prompt: Optional[str],
    system_prompt: Optional[str],
    timeout: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.novita.ai/openai",
        timeout=timeout,
        max_retries=0,
    )
    content = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": user_prompt},
    ]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": content}]
    if developer_prompt:
        messages = [{"role": "developer", "content": developer_prompt}] + messages
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model="pa/gemini-3-pro-preview",
                messages=messages,
                stream=False,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body={
                    "extra_body": {
                        "google": {
                            "thinkingConfig": {
                                "thinking_level": "high",
                            }
                        }
                    }
                },
                response_format={"type": "text"},
            )
            out = rsp.choices[0].message.content
            if isinstance(out, str):
                return out
            if isinstance(out, list):
                return "".join(part.get("text", "") for part in out if isinstance(part, dict))
            return str(out)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"[warn] gemini call failed attempt={attempt + 1}/{max_retries}: {e}")
            time.sleep(retry_sleep)
    return ""


def _call_qwen_dashscope(
    client: OpenAI,
    model: str,
    image_data_url: str,
    user_prompt: str,
    developer_prompt: Optional[str],
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> str:
    user_content = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": user_prompt},
    ]
    messages: List[Dict[str, Any]] = [{"role": "user", "content": user_content}]
    if developer_prompt:
        messages = [
            {"role": "developer", "content": [{"type": "text", "text": developer_prompt}]}
        ] + messages
    if system_prompt:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
        ] + messages

    for attempt in range(max_retries):
        try:
            rsp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            out = rsp.choices[0].message.content
            if isinstance(out, str):
                return out
            if isinstance(out, list):
                return "".join(part.get("text", "") for part in out if isinstance(part, dict))
            return str(out)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"[warn] qwen_dashscope call failed attempt={attempt + 1}/{max_retries}: {e}")
            time.sleep(retry_sleep)
    return ""


def main() -> None:
    args = _parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    pred_json_path = args.pred_json
    if not os.path.isabs(pred_json_path):
        pred_json_path = os.path.join(args.output_dir, pred_json_path)

    pred_jsonl_path = args.pred_jsonl
    if not os.path.isabs(pred_jsonl_path):
        pred_jsonl_path = os.path.join(args.output_dir, pred_jsonl_path)

    pred_csv_path = args.pred_csv
    if not os.path.isabs(pred_csv_path):
        pred_csv_path = os.path.join(args.output_dir, pred_csv_path)

    metrics_json_path = args.metrics_json
    if not os.path.isabs(metrics_json_path):
        metrics_json_path = os.path.join(args.output_dir, metrics_json_path)

    rows = load_jsonl(args.gt_jsonl)
    end = None if args.limit is None else args.start_index + args.limit
    rows = rows[args.start_index:end]

    images_root = args.images_root or os.path.dirname(args.gt_jsonl)
    openai_like_client = None
    if args.provider in {"vllm_qwen", "azure_gpt", "qwen_dashscope"}:
        openai_like_client = _new_client(args.base_url, args.api_key, args.timeout)
        if args.provider in {"vllm_qwen", "azure_gpt"}:
            try:
                _ = openai_like_client.models.list()
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to API server at {args.base_url}. "
                    "Please make sure endpoint/base_url/api_key are correct."
                ) from e

    preds: List[Dict[str, Any]] = []
    start_t = time.time()

    for i, row in enumerate(rows, 1):
        sample_id = row.get("id", f"sample_{i:06d}")
        image_rel = extract_image_relpath(row)
        image_abs = image_rel if os.path.isabs(image_rel) else os.path.join(images_root, image_rel)
        gt_label = extract_gt_label(row)
        split = infer_split(row, args.gt_jsonl)

        call_error = None
        response_text = ""
        t0 = time.time()
        try:
            image_data_url = _image_to_data_url(image_abs)
            if args.provider in {"vllm_qwen", "azure_gpt"}:
                assert openai_like_client is not None
                response_text = _call_openai_vision(
                    client=openai_like_client,
                    model=args.model,
                    image_data_url=image_data_url,
                    user_prompt=args.user_prompt,
                    developer_prompt=args.developer_prompt,
                    system_prompt=args.system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                    provider_name=args.provider,
                )
            elif args.provider == "qwen_dashscope":
                assert openai_like_client is not None
                response_text = _call_qwen_dashscope(
                    client=openai_like_client,
                    model=args.model,
                    image_data_url=image_data_url,
                    user_prompt=args.user_prompt,
                    developer_prompt=args.developer_prompt,
                    system_prompt=args.system_prompt,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )
            else:
                response_text = _call_gemini_3_pro(
                    api_key=args.api_key,
                    image_data_url=image_data_url,
                    user_prompt=args.user_prompt,
                    developer_prompt=args.developer_prompt,
                    system_prompt=args.system_prompt,
                    timeout=args.timeout,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    retry_sleep=args.retry_sleep,
                )
        except Exception as e:
            call_error = str(e)

        pred_seconds_total = parse_hhmmss(response_text) if response_text else None
        pred_minutes = (
            pred_seconds_total // 60
            if pred_seconds_total is not None
            else parse_hhmm(response_text) if response_text else None
        )
        elapsed = time.time() - t0

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
        preds.append(pred_row)

        if i % args.save_every == 0 or i == len(rows):
            write_json(pred_json_path, preds)
            write_jsonl(pred_jsonl_path, preds)
            write_csv(pred_csv_path, preds)

        if i % 10 == 0 or i == len(rows):
            print(f"[progress] {i}/{len(rows)} done")

    metrics = compute_metrics(preds)
    metrics["provider"] = args.provider
    metrics["model"] = (
        args.model
        if args.provider in {"vllm_qwen", "azure_gpt", "qwen_dashscope"}
        else "pa/gemini-3-pro-preview"
    )
    metrics["num_samples"] = len(rows)
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
    }

    write_json(metrics_json_path, metrics)

    print("\nEvaluation done.")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nSaved predictions: {pred_json_path}")
    print(f"Saved per-sample jsonl: {pred_jsonl_path}")
    print(f"Saved per-sample csv: {pred_csv_path}")
    print(f"Saved metrics: {metrics_json_path}")


if __name__ == "__main__":
    main()
