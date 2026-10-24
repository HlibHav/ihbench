#!/usr/bin/env python3
"""LIVE harness for AC-14 (F16/F24). Not collected by pytest; NEVER run in CI.

It talks to the real Boson Realtime API and spends credits. Run it by hand, from
the fork root, with BOSON_API_KEY in the environment (or the fork's .env):

    .venv/bin/python tests/live_replay_order.py

It picks the first sample whose history has at least two user turns with a WAV,
replays it once through HiggsRealtimeClient, records every sent and received
message to results/ac14-<utc>.jsonl (audio payloads replaced by their length,
headers never recorded) and prints three lines:

1. each committed.previous_item_id against the id of the item replayed just before it
2. whether any response.* event came before the client's response.create
3. the event types received after each conversation.item.create
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import websockets  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from ihbench.clients.base import GenerationConfig, GenericMessage  # noqa: E402
from ihbench.clients.higgs_realtime_client import HiggsRealtimeClient  # noqa: E402
from ihbench.data import load_eval_samples_from_hf  # noqa: E402

AUDIO_CACHE = ".ihbench_audio_cache"


def scrub(msg):
    """Replace audio payloads by their length so the log stays small and key-free."""
    if not isinstance(msg, dict):
        return msg
    out = dict(msg)
    for key in ("audio", "delta"):
        if isinstance(out.get(key), str) and (key == "audio" or str(out.get("type", "")).startswith("response.output_audio")):
            out[key] = f"<{len(out[key])} chars>"
    return out


class RecordingWS:
    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    async def send(self, data):
        self._log.append(("sent", time.monotonic(), json.loads(data)))
        return await self._inner.send(data)

    async def recv(self):
        raw = await self._inner.recv()
        try:
            self._log.append(("recv", time.monotonic(), json.loads(raw)))
        except (TypeError, ValueError):
            self._log.append(("recv", time.monotonic(), {"type": "<non-json>"}))
        return raw

    def __getattr__(self, name):
        return getattr(self._inner, name)


class RecordingConnect:
    def __init__(self, real_connect, log):
        self._real = real_connect
        self._log = log

    def __call__(self, url, **kwargs):  # kwargs (headers) deliberately not recorded
        return _RecordingCM(self._real(url, **kwargs), self._log)


class _RecordingCM:
    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    async def __aenter__(self):
        return RecordingWS(await self._inner.__aenter__(), self._log)

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


def pick_sample(samples):
    for sample in samples:
        history = sample.format_history_for_model(audio_dir=AUDIO_CACHE)
        audio_users = [m for m in history if m["role"] == "user" and "audio_path" in m and Path(m["audio_path"]).exists()]
        if len(audio_users) >= 2:
            return sample, history
    raise SystemExit("no sample with >= 2 user audio turns found")


def to_messages(history):
    out = []
    for m in history:
        if "audio_path" in m:
            out.append(GenericMessage(role=m["role"], type="audio", content=m["audio_path"]))
        else:
            out.append(GenericMessage(role=m["role"], type="text", content=m["content"]))
    return out


def analyse(log):
    # replay timeline in send order: ("audio"|"text", id filled from server events)
    timeline: list[dict] = []
    pending_text: list[dict] = []
    committed_seen = 0
    report_prev = []
    sent_response_create_at = None
    early_response = []
    after_item_create: list[list[str]] = []
    current_after: list[str] | None = None

    for direction, t, msg in log:
        mtype = msg.get("type", "")
        if direction == "sent":
            if mtype == "conversation.item.create":
                entry = {"kind": "text", "id": None}
                timeline.append(entry)
                pending_text.append(entry)
                current_after = []
                after_item_create.append(current_after)
            elif mtype == "input_audio_buffer.commit":
                timeline.append({"kind": "audio", "id": None})
            elif mtype == "response.create":
                sent_response_create_at = t
        else:
            if current_after is not None:
                current_after.append(mtype)
            audio_ids = {e["id"] for e in timeline if e["kind"] == "audio"}
            if mtype == "conversation.item.added" and pending_text:
                item = msg.get("item") or {}
                # A committed audio item is also announced with conversation.item.added,
                # after its input_audio_buffer.committed; it is not a text-item ack.
                if item.get("id") not in audio_ids:
                    pending_text.pop(0)["id"] = item.get("id")
            elif mtype == "input_audio_buffer.committed":
                audio_entries = [e for e in timeline if e["kind"] == "audio"]
                audio_entries[committed_seen]["id"] = msg.get("item_id")
                idx = timeline.index(audio_entries[committed_seen])
                expected = timeline[idx - 1]["id"] if idx > 0 else None
                report_prev.append((msg.get("previous_item_id"), expected, timeline[idx - 1]["kind"] if idx > 0 else "none"))
                committed_seen += 1
            if mtype.startswith("response.") and sent_response_create_at is None:
                early_response.append(mtype)
    return report_prev, early_response, after_item_create


def main():
    samples = load_eval_samples_from_hf("bosonai/ihbench", AUDIO_CACHE)
    sample, history = pick_sample(samples)
    messages = to_messages(history)

    log: list[tuple[str, float, dict]] = []
    real_connect = websockets.connect
    websockets.connect = RecordingConnect(real_connect, log)
    try:
        client = HiggsRealtimeClient("higgs-realtime", api_key=None)
        text = client.generate(sample.system_message, messages, GenerationConfig(max_new_tokens=4096))
    finally:
        websockets.connect = real_connect

    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"ac14-{stamp}.jsonl"
    with open(out_path, "w") as f:
        for direction, t, msg in log:
            f.write(json.dumps({"dir": direction, "t": t, "msg": scrub(msg)}) + "\n")

    prev, early, after = analyse(log)
    print(f"sample={sample.id} response_chars={len(text)} log={out_path}")
    print("1. committed.previous_item_id vs item replayed just before:")
    for i, (got, want, kind) in enumerate(prev, 1):
        print(f"   commit {i}: previous_item_id={got!r} expected={want!r} (prior item: {kind}) -> {'MATCH' if got == want else 'MISMATCH'}")
    print(f"2. response.* events before response.create: {early if early else 'none'}")
    print("3. event types received after each conversation.item.create:")
    for i, types in enumerate(after, 1):
        print(f"   item {i}: {sorted(set(types))}")


if __name__ == "__main__":
    main()
