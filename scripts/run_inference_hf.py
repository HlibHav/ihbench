#!/usr/bin/env python3
"""Run inference on IHBench samples using a local HuggingFace model.

Uses Accelerate for multi-GPU data sharding. Each process handles a shard of
the dataset, and results are gathered at the end.

Usage:
    # Single GPU:
    python scripts/run_inference_hf.py \
        --dataset /path/to/verified_conversations.jsonl \
        --model Qwen/Qwen2-Audio-7B-Instruct \
        --output responses/qwen2-audio-7b.jsonl

    # Multi-GPU with accelerate:
    accelerate launch --num_processes 4 scripts/run_inference_hf.py \
        --dataset /path/to/verified_conversations.jsonl \
        --model Qwen/Qwen2-Audio-7B-Instruct \
        --output responses/qwen2-audio-7b.jsonl

    # With YAML config and multiple epochs:
    accelerate launch --num_processes 4 scripts/run_inference_hf.py \
        --config configs/qwen2-audio.yaml \
        --dataset /path/to/verified_conversations.jsonl \
        --output responses/qwen2-audio-7b.jsonl \
        --epochs 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from ihbench.clients import GenerationConfig, GenericMessage, get_client
from ihbench.clients.hf_client import HFLocalClient
from ihbench.data import EvalSample, load_eval_samples, load_eval_samples_from_hf


def run_sample(client: HFLocalClient, sample: EvalSample, config: GenerationConfig, audio_dir: str | None = None) -> dict:
    """Generate a response for a single sample.

    When `audio_dir` is None we never tag any message as audio, so audio-capable
    clients fall back to their text path — that's how the --text-only modality
    ablation is plumbed end-to-end: runner drops audio_dir, every message
    becomes type="text", client uses text-only inputs.
    """
    messages = []
    for m in sample.format_history_for_model(audio_dir=audio_dir):
        if "audio_path" in m:
            messages.append(GenericMessage(role=m["role"], type="audio", content=m["audio_path"]))
        else:
            messages.append(GenericMessage(role=m["role"], type="text", content=m["content"]))
    response = client.generate(sample.system_message, messages, config)
    result = {
        "sample_id": sample.id,
        "conversation_id": sample.conversation_id,
        "interruption_type": sample.interruption_type,
        "turn_depth": sample.turn_depth,
        "response": response,
    }
    if hasattr(sample, "epoch"):
        result["epoch"] = sample.epoch
    return result


def main():
    parser = argparse.ArgumentParser(description="Run IHBench inference with a local HF model")
    parser.add_argument("--dataset", default=None,
                        help="Local conversations JSONL. If omitted, the HF dataset (--hf-repo) is used.")
    parser.add_argument("--hf-repo", default="bosonai/ihbench",
                        help="HuggingFace dataset repo (used when --dataset is omitted)")
    parser.add_argument("--audio-cache", default=".ihbench_audio_cache",
                        help="Local dir where HF-embedded audio is materialized")
    parser.add_argument("--output", required=True, help="Output JSONL path for responses")
    parser.add_argument("--config", default=None, help="YAML model config file")
    parser.add_argument("--model", default=None, help="HF model name or local path")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1, help="Number of inference epochs (for CI)")
    parser.add_argument("--audio-dir", default=None, help="Directory with extracted WAV files (data/audio/)")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--text-only", action="store_true",
                        help="Run the text-only modality ablation: skip audio input even for audio-capable models.")
    args = parser.parse_args()

    # Resolve config
    model = args.model
    device_map = args.device_map or "auto"
    torch_dtype = args.torch_dtype or "bfloat16"
    temperature = args.temperature or 1.0
    max_new_tokens = args.max_new_tokens or 4096

    client_kwargs: dict = {}
    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        model = args.model or cfg.get("model", model)
        device_map = args.device_map or cfg.get("device_map", device_map)
        torch_dtype = args.torch_dtype or cfg.get("torch_dtype", torch_dtype)
        if args.temperature is None and "temperature" in cfg:
            temperature = cfg["temperature"]
        if args.max_new_tokens is None and "max_new_tokens" in cfg:
            max_new_tokens = cfg["max_new_tokens"]
        # Arbitrary per-client constructor kwargs (e.g. {"thinking": true}).
        # These are forwarded verbatim to the client's __init__.
        client_kwargs = cfg.get("client_kwargs") or {}

    if model is None:
        parser.error("--model is required (via CLI or YAML config)")

    # Hybrid data-parallel + model-parallel launch.
    #
    # When launched via torchrun --nproc-per-node=N, each rank inherits
    # RANK, LOCAL_RANK, WORLD_SIZE env vars. We use LOCAL_RANK to pin each
    # process to a disjoint GPU slice (gpus_per_replica GPUs each) BEFORE
    # any torch / transformers code touches CUDA. Inside that slice,
    # device_map="auto" then naturally tensor-parallels the model across
    # only the rank's GPUs.
    #
    # The gather collective uses GLOO (CPU-side) rather than NCCL because
    # each rank has different visible GPUs, so NCCL's device-coordinated
    # all_gather can't align the ranks ("invalid device ordinal"). For
    # passing Python object lists between ranks, gloo is plenty fast.
    #
    # Plain `python run_inference_hf.py` (no torchrun env) falls through
    # to single-process mode.
    if "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        # gpus_per_replica is set by the SLURM launcher from the YAML.
        gpus_per_replica = int(os.environ.get("IHB_GPUS_PER_REPLICA", "1"))
        start = local_rank * gpus_per_replica
        end = start + gpus_per_replica
        # CUDA_VISIBLE_DEVICES may already be set (limiting the whole job's
        # GPU pool); slice that, not the absolute device list.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            ids = visible.split(",")
        else:
            import torch
            ids = [str(i) for i in range(torch.cuda.device_count())]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(ids[start:end])

        # Initialize the cross-rank gather group on GLOO so it doesn't
        # depend on GPU device alignment. Use a large timeout: the final
        # all_gather_object is a barrier where fast ranks wait for the slowest.
        # GLOO's default 30-min timeout trips for slow models (e.g. thinking
        # variants that emit long reasoning traces) when one rank's shard is
        # generation-heavy -- the straggler misses the barrier and the whole
        # job dies with 0 output (gemma4-12b-thinking-audio hit this). 6 h is
        # comfortably above any single-shard wall time and harmless for fast
        # models (the barrier returns as soon as all ranks arrive).
        import datetime as _dt
        import torch.distributed as dist
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo", timeout=_dt.timedelta(hours=6)
            )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        is_main = rank == 0

        # Override device_map for single-GPU replicas. With a GLOO process
        # group initialized and only one visible GPU per rank, newer
        # transformers ('device_map="auto"') tries to tensor-parallelize the
        # model across "all visible devices" — but for models without a
        # tensor parallel plan (most multimodal LMs: Qwen2.5-Omni, Phi-4,
        # ...) this raises "This model does not have a tensor parallel
        # plan." Pin the device explicitly when each rank owns 1 GPU.
        if gpus_per_replica == 1:
            device_map = "cuda:0"
    else:
        rank = 0
        world_size = 1
        is_main = True

    if args.dataset:
        if is_main:
            print(f"Loading dataset from {args.dataset} (epochs={args.epochs})...", file=sys.stderr)
        samples = load_eval_samples(args.dataset, epochs=args.epochs)
        hf_audio_cache = None
    else:
        if is_main:
            print(f"Loading HuggingFace dataset {args.hf_repo} (epochs={args.epochs})...", file=sys.stderr)
        samples = load_eval_samples_from_hf(args.hf_repo, args.audio_cache, epochs=args.epochs)
        hf_audio_cache = args.audio_cache  # HF loader materialized audio here
    if args.limit:
        samples = samples[: args.limit]

    # Shard across processes
    shard = [s for i, s in enumerate(samples) if i % world_size == rank]
    if is_main:
        print(f"Total samples: {len(samples)}, this shard: {len(shard)} (rank {rank}/{world_size})", file=sys.stderr)

    client = get_client(
        model,
        device_map=device_map,
        torch_dtype=torch_dtype,
        text_only=args.text_only,
        **client_kwargs,
    )
    config = GenerationConfig(temperature=temperature, max_new_tokens=max_new_tokens)

    # Write shard results to temp file, then gather on main process
    shard_results = []
    for sample in tqdm(shard, desc=f"Inference (rank {rank})", unit="sample", disable=not is_main):
        try:
            effective_audio_dir = None if args.text_only else (args.audio_dir or hf_audio_cache)
            result = run_sample(client, sample, config, effective_audio_dir)
            shard_results.append(result)
        except Exception as e:
            tqdm.write(f"  [rank {rank}] Failed {sample.id}: {e}", file=sys.stderr)

    # Gather results from all processes in memory via torch.distributed
    # all_gather_object on the GLOO group set up earlier. No temp shard
    # files, no NCCL device alignment issues.
    if world_size > 1:
        import torch.distributed as dist
        gathered: list = [None] * world_size
        dist.all_gather_object(gathered, shard_results)
        all_results = [r for shard in gathered for r in shard]
        if is_main:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as fout:
                for r in all_results:
                    fout.write(json.dumps(r) + "\n")
            print(f"\nDone. {len(all_results)} responses saved to {output_path}", file=sys.stderr)
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fout:
            for r in shard_results:
                fout.write(json.dumps(r) + "\n")
        print(f"\nDone. {len(shard_results)} responses saved to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
