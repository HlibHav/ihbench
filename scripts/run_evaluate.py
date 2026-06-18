#!/usr/bin/env python3
"""Evaluate pre-generated model responses on IHBench.

Takes model responses (from run_inference_api.py or run_inference_hf.py) and
baseline responses, runs TF and RQ judges, and outputs scored results.

Usage:
    python scripts/run_evaluate.py \
        --dataset /path/to/verified_conversations.jsonl \
        --responses responses/gpt-realtime-1.5.jsonl \
        --baseline-responses responses/gpt-4o-audio-preview.jsonl \
        --judge-model qwen3.5-397b \
        --judge-base-url http://localhost:8000/v1 \
        --output results/gpt-realtime-1.5.jsonl

Or with a YAML config:
    python scripts/run_evaluate.py \
        --dataset /path/to/verified_conversations.jsonl \
        --responses responses/gpt-realtime-1.5.jsonl \
        --baseline-responses responses/gpt-4o-audio-preview.jsonl \
        --judge-config configs/judge.yaml \
        --output results/gpt-realtime-1.5.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from ihbench.data import load_eval_samples, load_eval_samples_from_hf
from ihbench.judge import call_rq_judge, call_tf_judge
from ihbench.metrics import compute_all_metrics, format_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
_log = logging.getLogger(__name__)


def load_responses(path: str | Path) -> dict[str, dict]:
    """Load sample_id -> {response, epoch, ...} mapping from a responses JSONL file."""
    responses = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            responses[r["sample_id"]] = r
    return responses


def load_baseline_from_hf(repo_id: str) -> dict[str, dict]:
    """Load the GPT-4o Audio baseline (TF anchor) from the HF dataset's
    `baseline` config, keyed by sample_id (`<conversation_id>_<message_index>`)
    to match `load_responses`."""
    from datasets import load_dataset
    ds = load_dataset(repo_id, "baseline", split="test")
    out = {}
    for r in ds:
        sid = f"{r['conversation_id']}_{r['interrupting_user_message_index']}"
        out[sid] = {"sample_id": sid, "response": r["response"]}
    return out


def load_judge_config(path: str | Path) -> dict:
    """Load judge config from YAML."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


async def evaluate_one(
    client: AsyncOpenAI,
    model_name: str,
    sample_data: dict,
    prediction: str,
    baseline: str,
    judge_temperature: float | None,
    reasoning_effort: str | None = None,
) -> dict:
    """Evaluate a single sample on both TF and RQ axes."""
    conv_history = sample_data["conversation_history"]
    msg_idx = sample_data["interrupting_user_message_index"]
    rubric = sample_data["rubric"]
    sample_id = sample_data["id"]

    result = {
        "sample_id": sample_id,
        "conversation_id": sample_data["conversation_id"],
        "interruption_type": sample_data["interruption_type"],
        "turn_depth": sample_data["turn_depth"],
    }
    if "epoch" in sample_data:
        result["epoch"] = sample_data["epoch"]

    # Task Fulfillment (comparative)
    try:
        tf_judgment, resp_letter = await call_tf_judge(
            client=client,
            model_name=model_name,
            conversation_history=conv_history,
            msg_idx=msg_idx,
            criterion=rubric["task_fulfillment"]["criterion"],
            prediction=prediction,
            gt_response=baseline,
            sample_id=sample_id,
            judge_temperature=judge_temperature,
            reasoning_effort=reasoning_effort,
        )
        if tf_judgment:
            result["tf_win_score"] = int(tf_judgment.choice == resp_letter)
            result["tf_choice"] = tf_judgment.choice
            result["tf_model_position"] = resp_letter
            result["tf_comparison_rationale"] = tf_judgment.comparison_rationale
    except Exception as e:
        _log.error("TF judge error for %s: %s", sample_id, e)
        result["tf_win_score"] = None

    # Recovery Quality (absolute)
    try:
        rq_judgment = await call_rq_judge(
            client=client,
            model_name=model_name,
            conversation_history=conv_history,
            msg_idx=msg_idx,
            criteria=rubric["recovery_quality"]["criteria"],
            response=prediction,
            judge_temperature=judge_temperature,
            reasoning_effort=reasoning_effort,
        )
        if rq_judgment:
            n_criteria = len(rq_judgment.assessments)
            n_met = sum(a.met for a in rq_judgment.assessments)
            result["rq_pass"] = rq_judgment.verdict == "pass"
            result["rq_score"] = n_met / n_criteria if n_criteria > 0 else 0.0
            result["rq_assessments"] = [a.model_dump() for a in rq_judgment.assessments]
    except Exception as e:
        _log.error("RQ judge error for %s: %s", sample_id, e)
        result["rq_pass"] = None
        result["rq_score"] = None

    return result


async def run_evaluation(
    client: AsyncOpenAI,
    model_name: str,
    eval_pairs: list[tuple[dict, str, str]],
    output_path: Path,
    judge_temperature: float | None,
    reasoning_effort: str | None = None,
    max_concurrent: int = 40,
) -> list[dict]:
    """Run evaluation on all pairs with bounded concurrency."""
    sem = asyncio.Semaphore(max_concurrent)
    results = []
    completed = 0
    total = len(eval_pairs)

    async def _bounded(sample_data, prediction, baseline):
        async with sem:
            return await evaluate_one(client, model_name, sample_data, prediction, baseline, judge_temperature, reasoning_effort)

    tasks = [_bounded(sd, pred, bl) for sd, pred, bl in eval_pairs]

    with open(output_path, "w") as fout:
        pbar = tqdm(total=total, desc="Judging", unit="sample")
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                fout.write(json.dumps(result) + "\n")
                fout.flush()
                completed += 1
            except Exception as e:
                tqdm.write(f"Evaluation failed: {e}")
            pbar.update(1)
        pbar.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate IHBench responses with LLM judges")
    parser.add_argument("--dataset", default=None,
                        help="Local conversations JSONL. If omitted, the HF dataset (--hf-repo) is used.")
    parser.add_argument("--hf-repo", default="bosonai/ihbench",
                        help="HuggingFace dataset repo (used when --dataset / --baseline-responses omitted)")
    parser.add_argument("--responses", required=True, help="Model responses JSONL")
    parser.add_argument("--baseline-responses", default=None,
                        help="Baseline responses JSONL. If omitted, the HF dataset's `baseline` config is used.")
    parser.add_argument("--output", required=True, help="Output JSONL path for scored results")
    parser.add_argument("--judge-config", default=None, help="YAML config for judge model")
    parser.add_argument("--judge-model", default=None, help="Judge model name")
    parser.add_argument("--judge-base-url", default=None, help="Judge API base URL")
    parser.add_argument("--judge-api-key", default=None, help="Judge API key")
    parser.add_argument("--judge-temperature", type=float, default=None)
    parser.add_argument("--judge-reasoning-effort", default=None, help="Reasoning effort for judge (e.g., high, xhigh)")
    parser.add_argument("--max-concurrent", type=int, default=40, help="Max concurrent judge calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples (for testing)")
    args = parser.parse_args()

    # Resolve judge config: YAML -> CLI -> defaults
    judge_model = args.judge_model or "gpt-5.4-mini"
    judge_base_url = args.judge_base_url
    judge_api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "none")
    judge_temperature = args.judge_temperature
    reasoning_effort = args.judge_reasoning_effort
    max_concurrent = args.max_concurrent

    if args.judge_config:
        cfg = load_judge_config(args.judge_config)
        judge_model = args.judge_model or cfg.get("model", judge_model)
        judge_base_url = args.judge_base_url or cfg.get("base_url", judge_base_url)
        judge_api_key = args.judge_api_key or cfg.get("api_key", judge_api_key)
        if args.judge_temperature is None and "temperature" in cfg:
            judge_temperature = cfg["temperature"]
        if args.judge_reasoning_effort is None and "reasoning_effort" in cfg:
            reasoning_effort = cfg["reasoning_effort"]
        max_concurrent = args.max_concurrent or cfg.get("max_concurrent", max_concurrent)

    # Expand env vars in api_key (handles ${VAR} and $VAR patterns)
    if judge_api_key:
        judge_api_key = re.sub(
            r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z_0-9]*)",
            lambda m: os.environ.get(m.group(1) or m.group(2), m.group(0)),
            judge_api_key,
        )

    _log.info("Judge model: %s at %s", judge_model, judge_base_url or "(default)")

    # Load data
    if args.dataset:
        _log.info("Loading dataset from %s", args.dataset)
        samples = load_eval_samples(args.dataset)
    else:
        _log.info("Loading HuggingFace dataset %s", args.hf_repo)
        samples = load_eval_samples_from_hf(args.hf_repo)
    sample_map = {s.id: s for s in samples}

    _log.info("Loading model responses from %s", args.responses)
    model_responses = load_responses(args.responses)

    if args.baseline_responses:
        _log.info("Loading baseline responses from %s", args.baseline_responses)
        baseline_responses = load_responses(args.baseline_responses)
    else:
        _log.info("Loading baseline responses from HF dataset %s (baseline config)", args.hf_repo)
        baseline_responses = load_baseline_from_hf(args.hf_repo)

    # Build eval pairs: (sample_data_dict, model_response_text, baseline_response_text)
    eval_pairs = []
    for sid, model_rec in model_responses.items():
        # Strip _epochN suffix to find the base sample
        base_sid = re.sub(r"_epoch\d+$", "", sid)
        if base_sid not in sample_map:
            _log.warning("%s not in dataset, skipping", sid)
            continue
        # Baseline may or may not have epoch suffix; try both
        baseline_rec = baseline_responses.get(sid) or baseline_responses.get(base_sid)
        if baseline_rec is None:
            _log.warning("%s not in baseline responses, skipping", sid)
            continue
        sample = sample_map[base_sid]
        raw_history = [
            {
                "role": m.role,
                "content": m.content,
                **({"original_content": m.original_content} if m.original_content else {}),
                **({"interruption_spec": m.interruption_spec} if m.interruption_spec else {}),
            }
            for m in sample.conversation_history
        ]
        sample_data = {
            "id": sample.id,
            "conversation_id": sample.conversation_id,
            "conversation_history": raw_history,
            "interrupting_user_message_index": sample.interrupting_user_message_index,
            "interruption_type": sample.interruption_type,
            "turn_depth": sample.turn_depth,
            "rubric": {
                "task_fulfillment": {"criterion": sample.rubric.task_fulfillment_criterion},
                "recovery_quality": {"criteria": sample.rubric.recovery_quality_criteria},
            },
        }
        if "epoch" in model_rec:
            sample_data["epoch"] = model_rec["epoch"]
        eval_pairs.append((sample_data, model_rec["response"], baseline_rec["response"]))

    if args.limit:
        eval_pairs = eval_pairs[: args.limit]
    _log.info("Evaluating %d samples", len(eval_pairs))

    # Run
    client = AsyncOpenAI(api_key=judge_api_key, base_url=judge_base_url)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = asyncio.run(
        run_evaluation(client, judge_model, eval_pairs, output_path, judge_temperature, reasoning_effort, max_concurrent)
    )

    _log.info("%d samples evaluated. Results saved to %s", len(results), output_path)
    has_epochs = any(r.get("epoch") is not None and r["epoch"] > 0 for r in results)
    print(format_report(results, with_ci=has_epochs))


if __name__ == "__main__":
    main()
