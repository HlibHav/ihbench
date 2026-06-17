"""Data loader for IHBench evaluation.

Two sources are supported:
  - the public HuggingFace dataset (``load_eval_samples_from_hf``), the default
    for the released benchmark; audio is embedded in the dataset and materialized
    to a local cache so the existing clients consume it via ``audio_dir``.
  - verified conversation JSONL files (``load_eval_samples``), used internally
    during data construction.

Both expand into per-interruption evaluation samples.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "assistant" or "user"
    content: str
    original_content: str | None = None  # Full utterance before truncation (interrupted assistant msgs)
    interruption_spec: dict[str, Any] | None = None  # Present on interrupting user messages
    message_index: int | None = None  # Index in the original conversation (for audio lookup)


@dataclass
class Rubric:
    """Evaluation rubric for a single interruption point."""

    task_fulfillment_criterion: str
    recovery_quality_criteria: list[str]


@dataclass
class EvalSample:
    """A single evaluation sample: one interruption point in one conversation."""

    id: str
    conversation_id: str
    system_message: str
    conversation_history: list[Message]  # Messages up to and including the interruption
    interrupting_user_message_index: int
    interruption_type: str
    turn_depth: int  # Index of the interrupting user message among all user messages
    rubric: Rubric

    def format_history_for_model(
        self,
        audio_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Format conversation history as model input messages.

        Interrupted assistant messages use their truncated content (what was
        actually delivered before the interruption).

        Args:
            audio_dir: Directory containing audio WAVs as
                       ``{audio_dir}/{conversation_id}/{message_index}.wav``.
                       When provided, user messages include an ``audio_path``
                       field if the corresponding WAV exists.

        Returns:
            List of dicts with keys: role, content, and optionally audio_path.
        """
        messages = []
        for i, msg in enumerate(self.conversation_history):
            content = msg.content
            entry: dict[str, Any] = {"role": msg.role, "content": content}

            # Attach audio path for user messages
            if audio_dir is not None and msg.role == "user" and msg.message_index is not None:
                wav_path = Path(audio_dir) / self.conversation_id / f"{msg.message_index}.wav"
                if wav_path.exists():
                    entry["audio_path"] = str(wav_path)

            messages.append(entry)
        return messages


@dataclass
class Conversation:
    """A full benchmark conversation with all metadata."""

    id: str
    domain: dict[str, str]
    goal: str
    system_message: str
    knowledge_base: dict[str, Any]
    user_intent: dict[str, Any]
    messages: list[Message]
    rubrics: dict[int, dict[str, Any]]  # Keyed by interrupting user message index
    naturally_terminated: bool = True


def _parse_message(raw: dict[str, Any], index: int) -> Message:
    return Message(
        role=raw["role"],
        content=raw["content"],
        original_content=raw.get("original_content"),
        interruption_spec=raw.get("interruption_spec"),
        message_index=index,
    )


def load_conversations(path: str | Path) -> list[Conversation]:
    """Load conversations from a verified JSONL file."""
    path = Path(path)
    conversations = []
    with open(path) as f:
        for line in f:
            raw = json.loads(line)
            messages = [_parse_message(m, i) for i, m in enumerate(raw["conversation_history"])]

            # Parse rubrics: keyed by the user message index that triggered the interruption
            rubrics = {}
            for r in raw.get("rubrics", []):
                idx = r.get("interrupting_user_message_index", r.get("message_index"))
                if idx is not None:
                    rubrics[idx] = r

            conversations.append(Conversation(
                id=raw["id"],
                domain=raw["domain"] if isinstance(raw["domain"], dict) else {"name": raw["domain"]},
                goal=raw["goal"],
                system_message=raw["system_message"],
                knowledge_base=raw.get("knowledge_base", {}),
                user_intent=raw.get("user_intent", {}),
                messages=messages,
                rubrics=rubrics,
                naturally_terminated=raw.get("naturally_terminated", True),
            ))
    return conversations


def expand_to_eval_samples(conversations: list[Conversation]) -> list[EvalSample]:
    """Expand conversations into per-interruption evaluation samples.

    Each interrupting user message (one with an interruption_spec) becomes
    a separate evaluation sample. The conversation history is truncated to
    include messages up to and including the interrupting user message.
    """
    samples = []
    for conv in conversations:
        user_msg_count = 0
        for i, msg in enumerate(conv.messages):
            if msg.role == "user":
                if msg.interruption_spec:
                    # Find the rubric for this interruption
                    rubric_raw = conv.rubrics.get(i)
                    if rubric_raw is None:
                        continue  # Skip interruptions without rubrics

                    # Rubric may be nested: {"rubric": {"task_fulfillment": ..., "recovery_quality": ...}}
                    # or flat: {"task_fulfillment": ..., "recovery_quality": ...}
                    inner = rubric_raw.get("rubric", rubric_raw)
                    tf_raw = inner.get("task_fulfillment", {})
                    rq_raw = inner.get("recovery_quality", {})

                    rubric = Rubric(
                        task_fulfillment_criterion=tf_raw.get("criterion", ""),
                        recovery_quality_criteria=rq_raw.get("criteria", []),
                    )

                    sample = EvalSample(
                        id=f"{conv.id}_{i}",
                        conversation_id=conv.id,
                        system_message=conv.system_message,
                        conversation_history=conv.messages[: i + 1],
                        interrupting_user_message_index=i,
                        interruption_type=msg.interruption_spec.get("interruption_type", "unknown"),
                        turn_depth=user_msg_count,
                        rubric=rubric,
                    )
                    samples.append(sample)
                user_msg_count += 1
    return samples


def load_eval_samples(path: str | Path, epochs: int = 1) -> list[EvalSample]:
    """Convenience: load conversations and expand to eval samples in one call.

    Args:
        path: Path to verified conversations JSONL.
        epochs: Number of times to duplicate the samples. Each duplicate gets
                a unique ``epoch`` attribute (0-indexed) and a modified ``id``
                (``{original_id}_epoch{N}``). This enables multi-epoch evaluation
                for computing confidence intervals.
    """
    samples = expand_to_eval_samples(load_conversations(path))
    if epochs <= 1:
        for s in samples:
            s.epoch = 0
        return samples

    expanded = []
    for epoch in range(epochs):
        for s in samples:
            dup = EvalSample(
                id=f"{s.id}_epoch{epoch}",
                conversation_id=s.conversation_id,
                system_message=s.system_message,
                conversation_history=s.conversation_history,
                interrupting_user_message_index=s.interrupting_user_message_index,
                interruption_type=s.interruption_type,
                turn_depth=s.turn_depth,
                rubric=s.rubric,
            )
            dup.epoch = epoch
            expanded.append(dup)
    return expanded


# ---------------------------------------------------------------------------
# Public HuggingFace dataset loader
# ---------------------------------------------------------------------------

def _conversations_from_hf(repo_id, audio_cache, max_turns=20):
    """Reconstruct Conversation objects from the released HF dataset.

    The dataset stores one row per conversation with flat per-turn columns
    (assistant_turn_t_*, user_turn_t_*, turn_t_*_rubric) and embedded audio.
    We rebuild the message list (assistant at even index, user at odd) and
    materialize each user turn's audio to ``{audio_cache}/{conv_id}/{idx}.wav``
    so the existing clients can consume it via ``audio_dir``.
    """
    from datasets import Audio, load_dataset

    ds = load_dataset(repo_id, "conversations", split="test")
    # Disable decoding so we get the raw (byte-identical) WAV bytes back.
    for c in [col for col in ds.column_names if col.endswith("_audio")]:
        ds = ds.cast_column(c, Audio(decode=False))

    audio_cache = Path(audio_cache)
    conversations = []
    for row in ds:
        cid = row["conversation_id"]
        n = row["num_turns"]
        messages = []
        rubrics = {}
        for t in range(1, n + 1):
            ai, ui = 2 * (t - 1), 2 * t - 1
            a_orig = row.get(f"assistant_turn_{t}_original")
            messages.append(_parse_message({
                "role": "assistant",
                "content": row[f"assistant_turn_{t}_transcript"],
                "original_content": a_orig,
            }, ai))

            itype = row.get(f"user_turn_{t}_interruption_type")
            interruption_spec = {"interruption_type": itype} if itype else None
            messages.append(_parse_message({
                "role": "user",
                "content": row[f"user_turn_{t}_transcript"],
                "interruption_spec": interruption_spec,
            }, ui))

            # Materialize the user turn's audio bytes to the cache (byte-identical).
            audio = row.get(f"user_turn_{t}_audio")
            if audio and audio.get("bytes"):
                wav = audio_cache / cid / f"{ui}.wav"
                wav.parent.mkdir(parents=True, exist_ok=True)
                if not wav.exists():
                    wav.write_bytes(audio["bytes"])

            if itype:
                rubrics[ui] = {
                    "interrupting_user_message_index": ui,
                    "interruption_type": itype,
                    "rubric": {
                        "task_fulfillment": {"criterion": row.get(f"turn_{t}_tf_rubric") or ""},
                        "recovery_quality": {"criteria": list(row.get(f"turn_{t}_rq_rubrics") or [])},
                    },
                }

        conversations.append(Conversation(
            id=cid,
            domain=row["domain"] if isinstance(row["domain"], dict) else {"name": row["domain"]},
            goal=row["goal"],
            system_message=row["system_message"],
            knowledge_base=row.get("knowledge_base", {}),
            user_intent=row.get("user_intent", {}),
            messages=messages,
            rubrics=rubrics,
        ))
    return conversations


def load_eval_samples_from_hf(
    repo_id: str = "bosonai/ihbench",
    audio_cache: str | Path = ".ihbench_audio_cache",
    epochs: int = 1,
) -> list[EvalSample]:
    """Load the public IHBench benchmark from HuggingFace and expand to samples.

    Args:
        repo_id: HF dataset repo (default ``bosonai/ihbench``).
        audio_cache: local dir where embedded user-turn audio is written as
                     ``{conversation_id}/{message_index}.wav`` (byte-identical to
                     the released audio), then read back via ``audio_dir``.
        epochs: number of epochs (duplicates samples with distinct ``epoch``/id).

    Returns the same ``EvalSample`` objects as ``load_eval_samples`` so the rest
    of the harness (inference, judging) is identical regardless of source.
    """
    samples = expand_to_eval_samples(_conversations_from_hf(repo_id, audio_cache))
    if epochs <= 1:
        for s in samples:
            s.epoch = 0
        return samples
    expanded = []
    for epoch in range(epochs):
        for s in samples:
            dup = EvalSample(
                id=f"{s.id}_epoch{epoch}",
                conversation_id=s.conversation_id,
                system_message=s.system_message,
                conversation_history=s.conversation_history,
                interrupting_user_message_index=s.interrupting_user_message_index,
                interruption_type=s.interruption_type,
                turn_depth=s.turn_depth,
                rubric=s.rubric,
            )
            dup.epoch = epoch
            expanded.append(dup)
    return expanded
