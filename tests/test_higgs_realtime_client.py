"""Offline tests for the Higgs Realtime client (no network, sentinel keys only).

Wire facts (F-ids) come from the Boson protocol notes; the fake server below
only reacts to what a client sends, like the real one (F1: silent until the
first ``session.update``).
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import logging
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import websockets
import websockets.exceptions
import websockets.frames
import yaml

from ihbench.clients.base import GenerationConfig, GenericMessage

FORK = Path(__file__).resolve().parent.parent
SENTINEL = "sk-SENTINEL-123"
PREFIX = "higgs-realtime attempt failed:"
URL = "wss://api.boson.ai/v1/realtime"


# --------------------------------------------------------------------------- helpers


@pytest.fixture
def client_module():
    return importlib.import_module("ihbench.clients.higgs_realtime_client")


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("BOSON_API_KEY", raising=False)


def write_wav(path: Path, seconds: float, rate: int, channels: int = 1) -> Path:
    n = int(seconds * rate)
    t = np.arange(n) / rate
    mono = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    data = np.repeat(mono[:, None], channels, axis=1) if channels > 1 else mono
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(data.tobytes())
    return path


def write_empty_wav(path: Path) -> Path:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"")
    return path


def close_error(code: int | None) -> websockets.exceptions.ConnectionClosedError:
    rcvd = None if code is None else websockets.frames.Close(code, "")
    return websockets.exceptions.ConnectionClosedError(rcvd, None)


def done(status="completed"):
    resp = {"id": "resp_1"}
    if status is not None:
        resp["status"] = status
    return {"type": "response.done", "response": resp}


def delta(text):
    return {"type": "response.output_text.delta", "delta": text}


class Script:
    """Scripted server reactions. Each hook returns a list of (delay_s, event)."""

    def __init__(self, session_created_delay=0.0, committed_delay=0.0, commit_events=None,
                 response_events=None, extra_on_session=()):
        self.session_created_delay = session_created_delay
        self.committed_delay = committed_delay
        self.commit_events = commit_events  # replaces `committed` when set
        self.response_events = (
            [delta("Hel"), delta("lo"), done()] if response_events is None else response_events
        )
        self.extra_on_session = list(extra_on_session)
        self.commits = 0
        self.session_updates = 0

    def __call__(self, msg):
        t = msg.get("type")
        if t == "session.update":
            self.session_updates += 1
            if self.session_updates == 1:
                return [(self.session_created_delay, {"type": "session.created"})] + [
                    (0.0, e) for e in self.extra_on_session
                ]
            return [(0.0, {"type": "session.updated"})]
        if t == "input_audio_buffer.commit":
            self.commits += 1
            if self.commit_events is not None:
                return [(0.0, e) for e in self.commit_events]
            return [(self.committed_delay, {
                "type": "input_audio_buffer.committed",
                "item_id": f"item_{self.commits}",
                "previous_item_id": None if self.commits == 1 else f"item_{self.commits - 1}",
            })]
        if t == "response.create":
            return [(0.0, e) for e in self.response_events]
        return []


class FakeConnection:
    def __init__(self, react, enter_exc=None):
        self.react = react
        self.enter_exc = enter_exc
        self.sent: list[tuple[float, dict]] = []
        self.deliveries: list[tuple[float, str]] = []
        self.closed = False
        self.queue: asyncio.Queue | None = None

    async def __aenter__(self):
        if self.enter_exc is not None:
            raise self.enter_exc
        self.queue = asyncio.Queue()
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    def _deliver(self, delay, event):
        def put():
            kind = event.get("type", "?") if isinstance(event, dict) else "raw"
            self.deliveries.append((time.monotonic(), kind))
            self.queue.put_nowait(event)

        if delay:
            asyncio.get_running_loop().call_later(delay, put)
        else:
            put()

    async def send(self, data):
        msg = json.loads(data)
        self.sent.append((time.monotonic(), msg))
        for delay, event in self.react(msg):
            self._deliver(delay, event)

    async def recv(self):
        event = await self.queue.get()
        if isinstance(event, BaseException):
            raise event
        return event if isinstance(event, str) else json.dumps(event)

    async def close(self):
        self.closed = True

    # convenience views
    @property
    def types(self):
        return [m["type"] for _, m in self.sent]

    def of_type(self, t):
        return [m for _, m in self.sent if m["type"] == t]


class ConnectRecorder:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []
        self.conns: list[FakeConnection] = []
        self.react = Script()
        self.enter_exc = None

    @property
    def conn(self) -> FakeConnection:
        return self.conns[-1]

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        conn = FakeConnection(self.react, self.enter_exc)
        self.conns.append(conn)
        return conn


@pytest.fixture
def rec(monkeypatch):
    r = ConnectRecorder()
    monkeypatch.setattr(websockets, "connect", r)
    return r


def user_audio(path):
    return GenericMessage(role="user", type="audio", content=str(path))


def user_text(text):
    return GenericMessage(role="user", type="text", content=text)


def asst_text(text):
    return GenericMessage(role="assistant", type="text", content=text)


def generate(mod, messages, system="SYS", config=None, key="k", **client_kw):
    client = mod.HiggsRealtimeClient("higgs-realtime", api_key=key, **client_kw)
    return client.generate(system, messages, config or GenerationConfig(max_new_tokens=4096))


def walk(obj, path=()):
    """Yield (path, key, value) for every dict key in a nested structure."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, v
            yield from walk(v, path + (k,))
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, path)


def failed_records(caplog):
    return [r for r in caplog.records if r.getMessage().startswith(PREFIX)]


@pytest.fixture(autouse=True)
def _warn_level(caplog):
    caplog.set_level(logging.WARNING)


# --------------------------------------------------------------------------- config (step 2)


def test_config_has_five_values():
    cfg = yaml.safe_load((FORK / "configs" / "higgs-realtime.yaml").read_text())
    assert cfg == {
        "backend": "higgs-realtime",
        "model": "higgs-realtime",
        "api_key": "${BOSON_API_KEY}",
        "max_new_tokens": 4096,
        "max_workers": 1,
    }


# --------------------------------------------------------------------------- session start (step 3)


def test_first_message_is_session_update(client_module, rec):
    generate(client_module, [user_text("hi")])
    assert rec.conn.types[0] == "session.update"
    assert rec.conn.types.count("session.update") == 1


def test_session_update_model_and_instructions(client_module, rec):
    generate(client_module, [user_text("hi")], system="You are SYS")
    session = rec.conn.sent[0][1]["session"]
    assert session["model"] == "higgs-realtime"
    assert session["instructions"] == "You are SYS"


def test_session_update_text_only(client_module, rec):
    generate(client_module, [user_text("hi")])
    assert rec.conn.sent[0][1]["session"]["output_modalities"] == ["text"]


def test_turn_detection_null_at_f24_path_only(client_module, rec):
    generate(client_module, [user_text("hi")])
    first = rec.conn.sent[0][1]
    hits = [(p, k, v) for p, k, v in walk(first) if k == "turn_detection"]
    assert len(hits) == 1
    path, _, value = hits[0]
    assert path == ("session", "audio", "input")
    assert value is None


def test_max_output_tokens_from_max_new_tokens(client_module, rec):
    generate(client_module, [user_text("hi")], config=GenerationConfig(max_new_tokens=1234))
    assert rec.conn.sent[0][1]["session"]["max_output_tokens"] == 1234


def test_no_temperature_sent(client_module, rec, tmp_path):
    wav = write_wav(tmp_path / "a.wav", 0.5, 16000)
    generate(client_module, [user_audio(wav), asst_text("x"), user_text("y")],
             config=GenerationConfig(temperature=1.0, max_new_tokens=100))
    for _, msg in rec.conn.sent:
        assert not [k for _, k, _ in walk(msg) if k == "temperature"]


def test_handshake_bearer_header_no_key_in_url(client_module, rec):
    generate(client_module, [user_text("hi")], key=SENTINEL)
    assert len(rec.calls) == 1
    url, kwargs = rec.calls[0]
    assert url == URL
    assert SENTINEL not in url
    kwarg = client_module._headers_kwarg(importlib.metadata.version("websockets"))
    assert kwargs == {kwarg: {"Authorization": f"Bearer {SENTINEL}"}, "close_timeout": 1.0}


@pytest.mark.parametrize("key", [None, "", "   ", "${BOSON_API_KEY}"], ids=["unset", "empty", "blank", "placeholder"])
def test_bad_key_fails_before_connect(client_module, rec, clean_env, key):
    with pytest.raises(ValueError, match="BOSON_API_KEY"):
        client = client_module.HiggsRealtimeClient("higgs-realtime", api_key=key)
        client.generate("SYS", [user_text("hi")], GenerationConfig())
    assert rec.calls == []


def test_key_falls_back_to_env(client_module, rec, monkeypatch):
    monkeypatch.setenv("BOSON_API_KEY", SENTINEL)
    client = client_module.HiggsRealtimeClient("higgs-realtime")
    client.generate("SYS", [user_text("hi")], GenerationConfig())
    kwarg = client_module._headers_kwarg(importlib.metadata.version("websockets"))
    assert rec.calls[0][1][kwarg] == {"Authorization": f"Bearer {SENTINEL}"}


def test_headers_kwarg_by_version(client_module):
    assert client_module._headers_kwarg("12.0") == "extra_headers"
    assert client_module._headers_kwarg("13.1") == "extra_headers"
    assert client_module._headers_kwarg("14.0") == "additional_headers"
    assert client_module._headers_kwarg("17.3") == "additional_headers"


def test_connect_uses_kwarg_of_installed_version(client_module):
    version = importlib.metadata.version("websockets")
    expected = "additional_headers" if int(version.split(".")[0]) >= 14 else "extra_headers"
    assert client_module._headers_kwarg(version) == expected
    # the real library accepts that keyword
    assert expected in inspect.signature(websockets.connect).parameters


def test_default_attempt_timeout_is_60s(client_module):
    assert client_module.HiggsRealtimeClient("higgs-realtime", api_key="k").attempt_timeout_s == 60.0


# --------------------------------------------------------------------------- replay (step 4)


def test_chunk_b64_boundaries(client_module):
    data = "A" * (2 * 1_048_576 + 8)
    chunks = client_module._chunk_b64(data)
    assert len(chunks) == 3
    assert all(len(c) <= 1_048_576 and len(c) % 4 == 0 for c in chunks)
    assert "".join(chunks) == data


def test_replay_starts_after_session_created_only(client_module, rec):
    rec.react = Script(session_created_delay=0.2)  # never sends session.updated
    assert generate(client_module, [user_text("hi")]) == "Hello"
    conn = rec.conn
    created_at = next(t for t, kind in conn.deliveries if kind == "session.created")
    second_at = conn.sent[1][0]
    assert conn.sent[1][1]["type"] == "conversation.item.create"
    assert second_at >= created_at
    assert "session.updated" not in [k for _, k in conn.deliveries]


def test_no_audio_in_item_create(client_module, rec, tmp_path):
    wav = write_wav(tmp_path / "a.wav", 0.5, 16000)
    generate(client_module, [user_audio(wav), asst_text("ok"), user_text("again")])
    items = rec.conn.of_type("conversation.item.create")
    assert items
    for item in items:
        dumped = json.dumps(item)
        assert "input_audio" not in dumped
        assert '"audio"' not in dumped


def test_long_wav_split_into_appends_then_one_commit(client_module, rec, tmp_path):
    wav = write_wav(tmp_path / "long.wav", 20, 24000)  # 960000 PCM bytes -> 1.28M b64 chars
    generate(client_module, [user_audio(wav)])
    types = rec.conn.types
    assert types[1:-2] == ["input_audio_buffer.append"] * (len(types) - 3)
    assert types.count("input_audio_buffer.append") >= 2
    assert types.count("input_audio_buffer.commit") == 1
    assert types[-2:] == ["input_audio_buffer.commit", "response.create"]
    appends = rec.conn.of_type("input_audio_buffer.append")
    assert all(len(a["audio"]) <= 1_048_576 for a in appends)
    pcm = base64.b64decode("".join(a["audio"] for a in appends))
    assert len(pcm) == 960000


@pytest.mark.parametrize("rate,channels", [(48000, 2), (16000, 1), (24000, 1), (8000, 2)])
def test_audio_resampled_to_24k_mono_pcm16(client_module, rec, tmp_path, rate, channels):
    wav = write_wav(tmp_path / "a.wav", 2, rate, channels)
    generate(client_module, [user_audio(wav)])
    appends = rec.conn.of_type("input_audio_buffer.append")
    pcm = base64.b64decode("".join(a["audio"] for a in appends))
    assert abs(len(pcm) - 96000) <= 4  # 2 s * 24000 samples * 2 bytes


def test_text_user_turn_is_input_text_item(client_module, rec):
    generate(client_module, [user_text("hello there")])
    items = rec.conn.of_type("conversation.item.create")
    assert [i["item"] for i in items] == [{
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": "hello there"}],
    }]


def test_zero_sample_wav_fails_before_sending(client_module, rec, tmp_path, caplog):
    empty = write_empty_wav(tmp_path / "empty.wav")
    with pytest.raises(client_module.HiggsRealtimeError):
        generate(client_module, [user_text("first"), user_audio(empty)])
    assert rec.calls == []  # nothing sent, no connection
    records = failed_records(caplog)
    assert len(records) == 1
    assert "user turn 1" in records[0].getMessage()


def test_assistant_turn_is_text_item(client_module, rec):
    generate(client_module, [user_text("q"), asst_text("an answer")])
    items = [i["item"] for i in rec.conn.of_type("conversation.item.create")]
    assert items[1] == {
        "type": "message", "role": "assistant",
        "content": [{"type": "text", "text": "an answer"}],
    }


def test_empty_assistant_turn_sends_nothing(client_module, rec):
    generate(client_module, [user_text("q"), asst_text("")])
    assert rec.conn.types == ["session.update", "conversation.item.create", "response.create"]


def test_next_item_waits_for_committed(client_module, rec, tmp_path):
    rec.react = Script(committed_delay=0.2)
    w1 = write_wav(tmp_path / "1.wav", 0.5, 16000)
    w2 = write_wav(tmp_path / "2.wav", 0.5, 16000)
    generate(client_module, [user_audio(w1), asst_text("mid"), user_audio(w2)])
    conn = rec.conn
    assert conn.types == [
        "session.update",
        "input_audio_buffer.append", "input_audio_buffer.commit",
        "conversation.item.create",
        "input_audio_buffer.append", "input_audio_buffer.commit",
        "response.create",
    ]
    committed_at = [t for t, kind in conn.deliveries if kind == "input_audio_buffer.committed"]
    commit_idx = [i for i, ty in enumerate(conn.types) if ty == "input_audio_buffer.commit"]
    assert len(committed_at) == 2
    for delivered, idx in zip(committed_at, commit_idx):
        assert conn.sent[idx + 1][0] >= delivered


def test_completes_without_item_done(client_module, rec):
    # the script never emits conversation.item.done (F5)
    assert generate(client_module, [user_text("a"), asst_text("b"), user_text("c")]) == "Hello"
    assert "conversation.item.done" not in [k for _, k in rec.conn.deliveries]


def test_one_response_create_after_last_item(client_module, rec):
    generate(client_module, [user_text("a"), asst_text("b"), user_text("c")])
    types = rec.conn.types
    assert types.count("response.create") == 1
    assert types[-1] == "response.create"
    assert rec.conn.of_type("response.create") == [{"type": "response.create"}]


def test_no_system_role_item(client_module, rec):
    msgs = [GenericMessage(role="system", type="text", content="sneaky"), user_text("a"), asst_text("b")]
    generate(client_module, msgs, system="the system prompt")
    for item in rec.conn.of_type("conversation.item.create"):
        assert item["item"]["role"] in {"user", "assistant"}
    assert "sneaky" not in json.dumps([m for _, m in rec.conn.sent[1:]])


# --------------------------------------------------------------------------- response and failures (step 5)


def test_returns_concatenated_deltas_on_completed(client_module, rec):
    rec.react = Script(response_events=[delta("A"), delta("B"), delta(" C"), done("completed")])
    assert generate(client_module, [user_text("q")]) == "AB C"


def test_cancelled_response_fails(client_module, rec, caplog):
    rec.react = Script(response_events=[delta("partial"), done("cancelled")])
    with pytest.raises(client_module.HiggsRealtimeError) as ei:
        generate(client_module, [user_text("q")])
    assert "cancelled" in str(ei.value)
    assert "partial" not in str(ei.value)
    records = failed_records(caplog)
    assert len(records) == 1 and "partial" not in records[0].getMessage()


def test_unexpected_done_status_fails(client_module, rec, caplog):
    rec.react = Script(response_events=[delta("x"), done("failed")])
    with pytest.raises(client_module.HiggsRealtimeError):
        generate(client_module, [user_text("q")])
    records = failed_records(caplog)
    assert len(records) == 1 and "failed" in records[0].getMessage()


def test_missing_done_status_fails(client_module, rec, caplog):
    rec.react = Script(response_events=[delta("x"), done(None)])
    with pytest.raises(client_module.HiggsRealtimeError):
        generate(client_module, [user_text("q")])
    assert len(failed_records(caplog)) == 1


def test_unknown_events_ignored(client_module, rec):
    noise = [
        {"type": "totally.unknown", "x": 1},
        {"type": "rate_limits.updated"},
        {"type": "response.output_audio_transcript.length"},
        {"type": "conversation.item.added"},
        "not json at all",
        "[1, 2]",
    ]
    rec.react = Script(
        extra_on_session=noise,
        response_events=noise + [delta("ok")] + noise + [done()],
    )
    assert generate(client_module, [user_text("q")]) == "ok"


def test_error_event_fails_attempt(client_module, rec, caplog):
    err = {"type": "error", "error": {"type": "insufficient_quota", "code": "quota_x", "message": "SECRET-MSG"}}
    rec.react = Script(response_events=[err])
    with pytest.raises(client_module.HiggsRealtimeError) as ei:
        generate(client_module, [user_text("q")])
    assert "insufficient_quota" in str(ei.value)
    assert "SECRET-MSG" not in str(ei.value)
    records = failed_records(caplog)
    assert len(records) == 1 and "SECRET-MSG" not in records[0].getMessage()


def test_error_during_replay_fails(client_module, rec, tmp_path, caplog):
    wav = write_wav(tmp_path / "a.wav", 0.5, 16000)
    rec.react = Script(commit_events=[{"type": "error", "error": {"type": "voice_output_task_ongoing"}}])
    with pytest.raises(client_module.HiggsRealtimeError, match="voice_output_task_ongoing"):
        generate(client_module, [user_audio(wav)])
    assert "response.create" not in rec.conn.types
    assert len(failed_records(caplog)) == 1


@pytest.mark.parametrize("close,expect", [
    (("close", 4429), "4429"),
    (("close", 3000), "3000"),
    (("close", 1013), "1013"),
    (("idle", 1000), "session.idle_timeout"),
    (("close", None), "no close frame"),
], ids=["4429", "3000", "1013", "idle_timeout", "no_frame"])
def test_close_fails_attempt_naming_cause(client_module, rec, caplog, close, expect):
    kind, code = close
    events = [close_error(code)]
    if kind == "idle":
        events = [{"type": "session.idle_timeout"}, close_error(code)]
    rec.react = Script(response_events=events)
    with pytest.raises(client_module.HiggsRealtimeError) as ei:
        generate(client_module, [user_text("q")])
    assert expect in str(ei.value)
    records = failed_records(caplog)
    assert len(records) == 1 and expect in records[0].getMessage()


def test_close_after_done_returns_text(client_module, rec, caplog):
    rec.react = Script(response_events=[delta("fine"), done(), close_error(1000)])
    assert generate(client_module, [user_text("q")]) == "fine"
    assert failed_records(caplog) == []


def test_attempt_timeout_closes_and_fails(client_module, rec, caplog):
    rec.react = Script(response_events=[delta("never done")])
    with pytest.raises(client_module.HiggsRealtimeError):
        generate(client_module, [user_text("q")], attempt_timeout_s=0.2)
    assert rec.conn.closed is True
    records = failed_records(caplog)
    assert len(records) == 1 and "timeout" in records[0].getMessage()


class SendClosesScript(Script):
    """Server closes the socket while the client is still sending (reason text is attacker-shaped)."""

    def __init__(self, on_type, reason):
        super().__init__()
        self.on_type = on_type
        self.reason = reason

    def __call__(self, msg):
        if msg.get("type") == self.on_type:
            rcvd = websockets.frames.Close(1013, self.reason)
            raise websockets.exceptions.ConnectionClosedError(rcvd, None)
        return super().__call__(msg)


@pytest.mark.parametrize("on_type", ["conversation.item.create", "response.create"])
def test_close_during_send_is_a_logged_attempt_failure(client_module, rec, caplog, capsys, on_type):
    rec.react = SendClosesScript(on_type, f"reason {SENTINEL}")
    with pytest.raises(client_module.HiggsRealtimeError) as ei:
        generate(client_module, [user_text("q")])
    assert "1013" in str(ei.value)
    assert SENTINEL not in str(ei.value)
    records = failed_records(caplog)
    assert len(records) == 1 and "1013" in records[0].getMessage()
    out = capsys.readouterr()
    assert SENTINEL not in caplog.text + out.out + out.err


def test_connect_bounds_the_close_handshake(client_module, rec):
    generate(client_module, [user_text("q")])
    assert rec.calls[0][1].get("close_timeout") == 1.0


def test_timeout_attempt_is_bounded_when_close_is_slow(client_module, monkeypatch, caplog):
    """A peer that never answers the close frame must not stretch the attempt past timeout + ~2 s."""
    class SlowClose(FakeConnection):
        async def __aexit__(self, *exc):
            await asyncio.sleep(self.close_timeout)  # websockets waits this long for the peer
            return await super().__aexit__(*exc)

    def connect(url, **kwargs):
        conn = SlowClose(Script(response_events=[delta("never done")]))
        conn.close_timeout = kwargs.get("close_timeout", 10.0)  # websockets default is 10 s
        return conn

    monkeypatch.setattr(websockets, "connect", connect)
    t0 = time.monotonic()
    with pytest.raises(client_module.HiggsRealtimeError):
        generate(client_module, [user_text("q")], attempt_timeout_s=0.5)
    assert time.monotonic() - t0 < 0.5 + 2.0


def test_handshake_failure_is_logged_without_key(client_module, rec, caplog):
    rec.enter_exc = OSError(f"boom {SENTINEL}")
    with pytest.raises(client_module.HiggsRealtimeError) as ei:
        generate(client_module, [user_text("q")], key=SENTINEL)
    assert "OSError" in str(ei.value)
    assert SENTINEL not in str(ei.value)
    records = failed_records(caplog)
    assert len(records) == 1 and "OSError" in records[0].getMessage()
    assert SENTINEL not in caplog.text


def test_key_never_in_failure_output(client_module, rec, caplog, capsys):
    scenarios = [
        [{"type": "error", "error": {"type": "bad_thing", "message": SENTINEL}}],
        [close_error(4429)],
        [delta("x")],  # timeout
    ]
    for events in scenarios:
        rec.react = Script(response_events=events)
        with pytest.raises(client_module.HiggsRealtimeError) as ei:
            generate(client_module, [user_text("q")], key=SENTINEL, attempt_timeout_s=0.2)
        assert SENTINEL not in str(ei.value)
        assert SENTINEL not in repr(ei.value)
    client = client_module.HiggsRealtimeClient("higgs-realtime", api_key=SENTINEL)
    assert SENTINEL not in repr(client)
    out = capsys.readouterr()
    assert SENTINEL not in caplog.text + out.out + out.err


# --------------------------------------------------------------------------- runner (step 6)


@pytest.fixture
def runner(client_module, clean_env):
    spec = importlib.util.spec_from_file_location("run_inference_api_under_test", FORK / "scripts" / "run_inference_api.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_runner_builds_higgs_client(runner, client_module, monkeypatch):
    calls = []

    class Recorder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(client_module, "HiggsRealtimeClient", Recorder)
    args = SimpleNamespace(backend="higgs-realtime", model="higgs-realtime", api_key="k")
    built = runner.make_client(args)
    assert isinstance(built, Recorder)
    assert calls == [{"model_name": "higgs-realtime", "api_key": "k"}]


def test_runner_openai_realtime_unchanged(runner, monkeypatch):
    import ihbench.clients.openai_realtime_client as oai

    calls = []

    class Recorder:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(oai, "OpenAIRealtimeClient", Recorder)
    args = SimpleNamespace(backend="openai-realtime", model="gpt-realtime", api_key="k")
    built = runner.make_client(args)
    assert isinstance(built, Recorder)
    assert calls == [{"model_name": "gpt-realtime", "api_key": "k", "reasoning_effort": None}]
