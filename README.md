# IHBench Evaluation Toolkit

Evaluation harness for **IHBench (Interruption Handling Benchmark)**, which
measures post-interruption recovery in voice agents executing structured
workflows. This repo runs a model on the benchmark and scores it on two axes,
**task fulfillment** and **recovery quality**, with LLM judges.

The benchmark data (conversations, embedded audio, rubrics, and the baseline
responses) lives in the companion HuggingFace dataset and is loaded
automatically; you do not need to download anything separately.

**Links**
- Code: <https://github.com/boson-ai/ihbench>
- Dataset: <https://huggingface.co/datasets/bosonai/ihbench> (`bosonai/ihbench`)
- Paper: IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows

## Install

```bash
git clone https://github.com/boson-ai/ihbench
cd ihbench
pip install -e ".[api]"     # OpenAI + Gemini API models and the judge
```

Pick the extra that matches what you want to run:

```bash
pip install -e .            # judging + OpenAI Chat models only
pip install -e ".[api]"     # all API models (adds OpenAI Realtime + Gemini)
pip install -e ".[hf]"      # add open-weight (HuggingFace) inference
pip install -e ".[all]"     # everything (API + HuggingFace)
```

Set credentials (copy `.env.example` to `.env`):

```bash
OPENAI_API_KEY=...     # for OpenAI models and the default judge
GEMINI_API_KEY=...     # for Gemini models / the secondary judge
```

## TL;DR: evaluate a model end to end

```bash
# 1. generate responses on the benchmark (auto-loads bosonai/ihbench)
python scripts/run_inference_api.py --config configs/gpt-realtime-2.yaml \
    --output responses/gpt-realtime-2.jsonl --epochs 3

# 2. judge them (TF vs. the GPT-4o Audio baseline + RQ rubric pass/fail)
python scripts/run_evaluate.py --responses responses/gpt-realtime-2.jsonl \
    --output results/gpt-realtime-2.jsonl --judge-config configs/judge.yaml

# 3. print the scores
python scripts/summarize_results.py results/
```

That is the whole pipeline. The sections below explain each step and the
open-weight / text-only variants.

## Run the benchmark

The pipeline is three steps: **infer → judge → summarize**. By default every
script loads the benchmark from `bosonai/ihbench` (override with `--dataset` to
use a local JSONL).

### 1. Generate model responses

API model (OpenAI / Gemini / Realtime):

```bash
python scripts/run_inference_api.py \
    --config configs/gpt-realtime-2.yaml \
    --output responses/gpt-realtime-2.jsonl \
    --epochs 3
```

Open-weight model (HuggingFace), single GPU:

```bash
python scripts/run_inference_hf.py \
    --config configs/qwen3-omni-30b.yaml \
    --output responses/qwen3-omni-30b.jsonl \
    --epochs 3
```

Text-only ablation (transcripts instead of audio): add `--text-only`.

On first run the embedded audio is materialized once to `.ihbench_audio_cache/`
(byte-identical to the released WAVs) and reused thereafter.

### 2. Judge the responses

```bash
python scripts/run_evaluate.py \
    --responses responses/gpt-realtime-2.jsonl \
    --output results/gpt-realtime-2.jsonl \
    --judge-config configs/judge.yaml
```

The task-fulfillment baseline (GPT-4o Audio) is read from the dataset's
`baseline` config automatically; override with `--baseline-responses`. The
default judge is GPT-5.4-mini (`configs/judge.yaml`); a Gemini secondary judge
is in `configs/judge-gemini.yaml`.

### 3. Summarize

```bash
python scripts/summarize_results.py results/
```

Reports per-model task-fulfillment win rate and recovery-quality pass rate with
95% bootstrap confidence intervals, plus per-interruption-type breakdowns.

## Adding a model

Each model is two files and zero edits to the runner: a client under
`ihbench/clients/<name>_client.py` that self-registers via
`@register_client("<HF-id-glob>")`, and a YAML in `configs/`. See the existing
open-weight clients for the pattern.

## Layout

```
ihbench/            core library
  data.py           dataset loading (HF + local JSONL) -> EvalSample objects
  judge.py          TF (comparative) and RQ (rubric) judges
  metrics.py        scoring
  config.py         YAML -> client/generation config
  clients/          one client per model family (OpenAI, Gemini, Realtime,
                    vLLM, + open-weight: Qwen, Phi-4, Voxtral, MiMo, Kimi, Gemma)
scripts/            run_inference_api, run_inference_hf, run_evaluate, summarize_results
configs/            one YAML per evaluated model + the judge configs
```

## Citation

```bibtex
@misc{ihbench2026,
  title         = {IHBench: Evaluating Post-Interruption Recovery in Voice Agents with Structured Workflows},
  author        = {Salimi, Ahmad and Ma, Wentao and Tang, Yuzhi and Shen, Dongming and Li, Mu and Smola, Alex},
  year          = {2026},
  eprint        = {arXiv:TODO},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
}
```

## License

Copyright 2026 Boson AI. This evaluation toolkit is licensed under the
Apache License, Version 2.0 (see `LICENSE`). The benchmark dataset is released
separately at <https://huggingface.co/datasets/bosonai/ihbench>.
