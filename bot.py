#!/usr/bin/env python3
"""
Pipecat SmallWebRTC voice bot (STT -> LLM -> buffered TTS) for your local stack.

This keeps the *working* voice pipeline behavior from the known-good version, and adds
server-owned UI/state plumbing *without changing what triggers the LLM*.

Adds (optional, only if AppState is passed in from server.py):
- Clean transcript export (user + assistant) via AppState.events (SSE) + transcript store
- Runtime phase indicator: listening / waiting_llm / speaking / disconnected
- Uses AppState.config as *defaults* for endpoints/models/voice, with safe fallbacks to .env
  (prevents "blank UI field wiped my config" from breaking the pipeline)

Still includes your existing fixes:
- Conversation progression: writes assistant chunks back into LLMContext
- Dedupe: only consumes one stream of LLM fragments (ACCEPT_MESSAGE_TYPE)
- Chunking: buffers token fragments into sentence-ish chunks for natural TTS
- Greeting: optional one-time kickoff on connect (LLMRunFrame) WITHOUT sticky "greet me" user msg

Env (same as before):
  LLM_BASE_URL=http://192.168.188.5:11434/v1
  LLM_API_KEY=local
  LLM_MODEL=gpt-oss:latest

  STT_BASE_URL=http://192.168.188.50:8000/v1
  STT_API_KEY=local
  STT_MODEL=Systran/faster-whisper-small

  TTS_BASE_URL=http://192.168.188.50:8004/v1
  TTS_API_KEY=local
  TTS_MODEL=gpt-4o-mini-tts
  TTS_VOICE=Frieren.wav

Optional tuning:
  ACCEPT_MESSAGE_TYPE=bot-llm-text
  TTS_MIN_CHARS=120
  TTS_MAX_CHARS=320
  TTS_FLUSH_TIMEOUT_MS=900
  TTS_FLUSH_ON_PUNCT=1

  GREET_ON_CONNECT=1   # default 1: send LLMRunFrame() once on connect
"""

from __future__ import annotations

import asyncio
import io
import os
import re
import time
import wave
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import httpx
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    LLMRunFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

try:
    # Optional: only present in the "UI state" version of the server
    from app_state import AppState, now_ms
except Exception:  # pragma: no cover
    AppState = None  # type: ignore
    def now_ms() -> int:  # type: ignore
        return int(time.time() * 1000)

load_dotenv(override=True)


def must_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()


def clean_env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


def _nonempty(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _normalize_v1_base(url: str) -> str:
    """
    Ensures base URL ends with /v1 (no trailing slash after v1).
    Prevents common breakages when UI saves 'http://host:11434' or '.../v1/'.
    """
    u = url.strip().rstrip("/")
    if not u:
        return u
    if u.endswith("/v1"):
        return u
    # If someone passed .../v1/..., leave it alone (rare)
    if re.search(r"/v1($|/)", u):
        return u.rstrip("/")
    return u + "/v1"


SYSTEM_INSTRUCTION = clean_env_str(
    "SYSTEM_INSTRUCTION",
    "You are a low-latency voice assistant. Keep responses short and conversational. "
    "Avoid long lists unless asked. "
    "Greet the user once when the session starts. Do not introduce yourself again unless asked.",
)

# Chunking knobs
TTS_MIN_CHARS = int(clean_env_str("TTS_MIN_CHARS", "90") or "90")
TTS_MAX_CHARS = int(clean_env_str("TTS_MAX_CHARS", "280") or "280")
TTS_FLUSH_TIMEOUT_MS = int(clean_env_str("TTS_FLUSH_TIMEOUT_MS", "850") or "850")
TTS_FLUSH_ON_PUNCT = clean_env_str("TTS_FLUSH_ON_PUNCT", "1") not in ("0", "false", "False")

# DEDUPE: only accept one transport stream type
ACCEPT_MESSAGE_TYPE = clean_env_str("ACCEPT_MESSAGE_TYPE", "bot-llm-text")

# Greet on connect by forcing a first LLM run
GREET_ON_CONNECT = clean_env_str("GREET_ON_CONNECT", "1") not in ("0", "false", "False")


class RuntimePhase:
    IDLE = "idle"
    LISTENING = "listening"
    WAITING_LLM = "waiting_llm"
    SPEAKING = "speaking"
    DISCONNECTED = "disconnected"


async def _set_phase(app_state: Optional["AppState"], phase: str) -> None:
    if not app_state:
        return
    app_state.runtime.phase = phase
    app_state.runtime.updated_ms = now_ms()
    await app_state.events.publish({"type": "phase", "phase": phase, "updated_ms": app_state.runtime.updated_ms})


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int, int]:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        nframes = wf.getnframes()
        pcm = wf.readframes(nframes)
    if sampwidth != 2:
        raise ValueError(f"Expected 16-bit PCM WAV. Got sampwidth={sampwidth}")
    return pcm, sample_rate, channels


def _dig(d: Any, path: list[Any]) -> Any:
    cur = d
    for p in path:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and isinstance(p, int) and 0 <= p < len(cur):
            cur = cur[p]
        else:
            return None
    return cur


def _type_matches_accept(mtype: Optional[str]) -> bool:
    """Return True if a message type should be treated as LLM text output."""
    if not isinstance(mtype, str):
        return False
    if not ACCEPT_MESSAGE_TYPE:
        return True
    # Pipecat variants sometimes suffix stream types (e.g. bot-llm-text-delta)
    if mtype == ACCEPT_MESSAGE_TYPE:
        return True
    if mtype.startswith(ACCEPT_MESSAGE_TYPE + "-"):
        return True
    return False


def extract_llm_text_fragment(msg: Any) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Returns (fragment, is_final, msg_type).

    IMPORTANT: We only accept message.type == ACCEPT_MESSAGE_TYPE by default,
    to avoid duplicates from bot-transcription or other streams.
    """
    if not isinstance(msg, dict):
        return None, False, None

    mtype = msg.get("type")
    if ACCEPT_MESSAGE_TYPE and not _type_matches_accept(mtype):
        # Filter mismatch: caller may still want msg_type for debugging.
        return None, False, mtype

    # text candidates
    candidates = [
        ["data", "text"],
        ["data", "content"],
        ["data", "delta"],
        ["data", "choices", 0, "delta", "content"],
        ["data", "choices", 0, "message", "content"],
    ]
    text = None
    for p in candidates:
        got = _dig(msg, p)
        if isinstance(got, str) and got != "":
            text = got
            break

    # final candidates
    final = False
    for p in (["data", "final"], ["data", "is_final"], ["data", "done"], ["data", "completed"]):
        got = _dig(msg, p)
        if isinstance(got, bool):
            final = got
            break

    # fallback: infer final from type string (rare here)
    if isinstance(mtype, str) and re.search(r"(end|stop|done|final|complete)$", mtype, re.IGNORECASE):
        final = True

    if isinstance(text, str):
        return text, final, mtype
    return None, final, mtype


def extract_any_text(msg: Any) -> Tuple[Optional[str], Optional[str]]:
    """Best-effort extraction of any text payload from a transport message (no type filter)."""
    if not isinstance(msg, dict):
        return None, None
    mtype = msg.get("type")
    candidates = [
        ["data", "text"],
        ["data", "content"],
        ["data", "delta"],
        ["data", "choices", 0, "delta", "content"],
        ["data", "choices", 0, "message", "content"],
    ]
    for p in candidates:
        got = _dig(msg, p)
        if isinstance(got, str) and got.strip() != "":
            return got, mtype if isinstance(mtype, str) else None
    return None, mtype if isinstance(mtype, str) else None


@dataclass
class _BufState:
    buf: str = ""
    last_update_ms: int = 0
    last_flush_ms: int = 0


class TextBufferAndChunker(FrameProcessor):
    """
    Buffers token fragments into readable chunks and emits them as:
      OutputTransportMessageUrgentFrame(message={"label":"assistant.tts_text", ...})
    """

    def __init__(self):
        super().__init__()
        self._st = _BufState()

    def _check_started(self, frame: Frame) -> bool:
        return True

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _append(self, cur: str, frag: str) -> str:
        if not cur:
            return frag
        frag = frag.replace("–", "-").replace("—", "—")
        if frag.startswith(("'", "’")):
            return cur + frag
        if frag in (".", ",", "!", "?", ";", ":", ")", "]", "}", "…"):
            return cur + frag
        if frag == "-":
            return cur.rstrip() + "-"
        if cur.endswith((" ", "\n", "\t")):
            return cur + frag
        if frag.startswith(" "):
            return cur + frag
        return cur + " " + frag

    def _should_flush_on_punct(self, s: str) -> bool:
        if not TTS_FLUSH_ON_PUNCT:
            return False
        return bool(re.search(r"([.!?]\s|[.!?]$|\n)", s))

    def _normalize_chunk(self, s: str) -> str:
        s = re.sub(r"\s*-\s*", "-", s)
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    async def _emit_chunk(self, direction: FrameDirection, chunk: str, reason: str):
        chunk = self._normalize_chunk(chunk)
        if not chunk:
            return
        logger.info(f"[CHUNK] reason={reason} chars={len(chunk)} preview={chunk[:140]!r}")
        msg = {"label": "assistant.tts_text", "type": "assistant_text", "data": {"text": chunk, "final": True}}
        await self.push_frame(OutputTransportMessageUrgentFrame(message=msg), direction)

    async def _flush(self, direction: FrameDirection, reason: str):
        text = self._st.buf
        self._st.buf = ""
        self._st.last_flush_ms = self._now_ms()

        text = self._normalize_chunk(text)
        if not text:
            return

        while len(text) > TTS_MAX_CHARS:
            cut = text.rfind(". ", 0, TTS_MAX_CHARS)
            if cut == -1:
                cut = text.rfind(" ", 0, TTS_MAX_CHARS)
            if cut == -1:
                cut = TTS_MAX_CHARS
            chunk = text[: cut + 1]
            text = text[cut + 1 :].lstrip()
            await self._emit_chunk(direction, chunk, reason + ":split")

        await self._emit_chunk(direction, text, reason)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        now = self._now_ms()
        if self._st.buf and self._st.last_update_ms and (now - self._st.last_update_ms) >= TTS_FLUSH_TIMEOUT_MS:
            await self._flush(direction, "timeout")

        if not isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return

        msg = getattr(frame, "message", None)
        frag, is_final, _mtype = extract_llm_text_fragment(msg)

        if frag is not None:
            self._st.buf = self._append(self._st.buf, frag)
            self._st.last_update_ms = now

            if len(self._st.buf) >= TTS_MIN_CHARS and self._should_flush_on_punct(self._st.buf):
                await self._flush(direction, "punct+min")

        if is_final and self._st.buf:
            if (now - self._st.last_flush_ms) > 150:
                await self._flush(direction, "final")


class AssistantContextWriter(FrameProcessor):
    """
    Appends assistant text chunks into LLMContext (conversation progression).
    """

    def __init__(self, context: LLMContext):
        super().__init__()
        self._context = context

    def _check_started(self, frame: Frame) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if not isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return

        msg = getattr(frame, "message", None)
        if not isinstance(msg, dict):
            return
        if msg.get("label") != "assistant.tts_text":
            return

        text = _dig(msg, ["data", "text"])
        if not isinstance(text, str) or not text.strip():
            return

        self._context.add_message({"role": "assistant", "content": text.strip()})
        logger.debug(f"[CTX] appended assistant message chars={len(text.strip())}")


class TranscriptExporter(FrameProcessor):
    """
    Exports clean user transcript lines to AppState (non-invasive; does not modify frames).
    """
    def __init__(self, app_state: "AppState"):
        super().__init__()
        self._app_state = app_state

    def _check_started(self, frame: Frame) -> bool:
        return True

    def _extract_user_text(self, msg: dict) -> Optional[str]:
        # Typical STT stream: type == "bot-transcription", data.text
        # Some builds use different type strings; accept a small set of hints.
        mtype = str(msg.get("type") or "").lower()
        if not any(k in mtype for k in ("transcription", "stt", "asr")):
            return None
        for p in (["data", "text"], ["data", "transcript"], ["data", "content"]):
            got = _dig(msg, p)
            if isinstance(got, str) and got.strip():
                return got.strip()
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)
        if not isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return
        msg = getattr(frame, "message", None)
        if not isinstance(msg, dict):
            return
        text = self._extract_user_text(msg)
        if not text:
            return
        # UI-only: we *guess* we're now waiting for LLM
        await _set_phase(self._app_state, RuntimePhase.WAITING_LLM)
        self._app_state.append_transcript(f"User: {text}")
        await self._app_state.events.publish({"type": "transcript", "role": "user", "text": text, "ts_ms": now_ms()})


class AssistantTextExporter(FrameProcessor):
    """
    Exports assistant chunks (assistant.tts_text) to AppState transcript.
    """
    def __init__(self, app_state: "AppState"):
        super().__init__()
        self._app_state = app_state

    def _check_started(self, frame: Frame) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)
        if not isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return
        msg = getattr(frame, "message", None)
        if not isinstance(msg, dict):
            return
        if msg.get("label") != "assistant.tts_text":
            return
        text = _dig(msg, ["data", "text"])
        if not isinstance(text, str) or not text.strip():
            return
        clean = text.strip()
        await _set_phase(self._app_state, RuntimePhase.SPEAKING)
        self._app_state.append_transcript(f"Assistant: {clean}")
        await self._app_state.events.publish({"type": "transcript", "role": "assistant", "text": clean, "ts_ms": now_ms()})


class ChatterboxTTSProcessor(FrameProcessor):
    def __init__(self, base_url: str, api_key: str, model: str, voice: str, app_state: Optional["AppState"] = None):
        super().__init__()
        self._url = base_url.rstrip("/") + "/audio/speech"
        self._api_key = api_key.strip()
        self._model = model
        self._voice = voice
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0))
        self._app_state = app_state

    def _check_started(self, frame: Frame) -> bool:
        return True

    def _extract_tts_chunk(self, frame: Frame) -> Optional[str]:
        if not isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return None
        msg = getattr(frame, "message", None)
        if not isinstance(msg, dict):
            return None
        if msg.get("label") != "assistant.tts_text":
            return None
        text = _dig(msg, ["data", "text"])
        if isinstance(text, str) and text.strip():
            return text.strip()
        return None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        text = self._extract_tts_chunk(frame)
        if not text:
            return

        logger.info(f"[TTS] chars={len(text)} preview={text[:140]!r}")
        if self._app_state:
            await _set_phase(self._app_state, RuntimePhase.SPEAKING)

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {"model": self._model, "voice": self._voice, "input": text}
        r = await self._client.post(self._url, headers=headers, json=payload)
        r.raise_for_status()

        pcm_bytes, sample_rate, channels = wav_to_pcm16(r.content)
        await self.push_frame(
            TTSAudioRawFrame(audio=pcm_bytes, sample_rate=sample_rate, num_channels=channels),
            direction,
        )

        # UI-only: back to listening quickly (we don't try to time audio perfectly)
        if self._app_state:
            asyncio.create_task(_set_phase(self._app_state, RuntimePhase.LISTENING))

    async def cleanup(self):
        try:
            await self._client.aclose()
        finally:
            await super().cleanup()


class AfterLLMSpy(FrameProcessor):
    """Prints only accepted stream fragments so you can verify dedupe is working."""
    def __init__(self, max_logs: int = 200):
        super().__init__()
        self._n = 0
        self._max = max_logs

    def _check_started(self, frame: Frame) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if self._n >= self._max:
            return

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            msg = getattr(frame, "message", None)
            frag, fin, mtype = extract_llm_text_fragment(msg)
            if frag is not None:
                self._n += 1
                logger.debug(f"[SPY] frag={frag!r} final={fin} type={mtype}")


class RawLLMMessageSpy(FrameProcessor):
    """Logs LLM transport messages *without* applying ACCEPT_MESSAGE_TYPE.

    If you see messages here but not in AfterLLMSpy / chunker, your ACCEPT_MESSAGE_TYPE
    is filtering out the actual stream.
    """

    def __init__(self, max_logs: int = 80):
        super().__init__()
        self._n = 0
        self._max = max_logs

    def _check_started(self, frame: Frame) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if self._n >= self._max:
            return

        if not isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return

        msg = getattr(frame, "message", None)
        if not isinstance(msg, dict):
            return

        text, mtype = extract_any_text(msg)
        if text is None:
            return

        self._n += 1
        logger.debug(
            f"[LLM-RAW] type={mtype!r} accept={ACCEPT_MESSAGE_TYPE!r} preview={text[:90]!r}"
        )


def _resolve_config(app_state: Optional["AppState"]) -> dict:
    """
    Resolve endpoints/models/keys/voice from AppState.config (if present), falling back to env.
    We also normalize base URLs to end with /v1.
    """
    cfg = getattr(app_state, "config", None) if app_state else None

    llm_base = _normalize_v1_base(_nonempty(getattr(cfg, "llm_base_url", None)) or must_env("LLM_BASE_URL"))
    llm_key = _nonempty(getattr(cfg, "llm_api_key", None)) or must_env("LLM_API_KEY")
    llm_model = _nonempty(getattr(cfg, "llm_model", None)) or clean_env_str("LLM_MODEL", "gpt-oss:latest")

    stt_base = _normalize_v1_base(_nonempty(getattr(cfg, "stt_base_url", None)) or must_env("STT_BASE_URL"))
    stt_key = _nonempty(getattr(cfg, "stt_api_key", None)) or must_env("STT_API_KEY")
    stt_model = _nonempty(getattr(cfg, "stt_model", None)) or clean_env_str("STT_MODEL", "whisper-1")

    tts_base = _normalize_v1_base(_nonempty(getattr(cfg, "tts_base_url", None)) or must_env("TTS_BASE_URL"))
    tts_key = _nonempty(getattr(cfg, "tts_api_key", None)) or clean_env_str("TTS_API_KEY", "")
    tts_model = _nonempty(getattr(cfg, "tts_model", None)) or clean_env_str("TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice = _nonempty(getattr(cfg, "tts_voice", None)) or clean_env_str("TTS_VOICE", "Frieren.wav")

    sys_inst = _nonempty(getattr(cfg, "system_instruction", None)) or clean_env_str("SYSTEM_INSTRUCTION", SYSTEM_INSTRUCTION)

    return {
        "llm_base_url": llm_base,
        "llm_api_key": llm_key,
        "llm_model": llm_model,
        "stt_base_url": stt_base,
        "stt_api_key": stt_key,
        "stt_model": stt_model,
        "tts_base_url": tts_base,
        "tts_api_key": tts_key,
        "tts_model": tts_model,
        "tts_voice": tts_voice,
        "system_instruction": sys_inst,
    }


async def run_bot(webrtc_connection, app_state: Optional["AppState"] = None):
    """
    If app_state is provided (server.py does this), we publish transcript + status to SSE.
    If not, behavior matches the original working bot.
    """
    if app_state:
        app_state.active_sessions += 1
        await _set_phase(app_state, RuntimePhase.LISTENING)

    cfg = _resolve_config(app_state)

    if app_state:
        logger.info(
            f"[CFG] llm={cfg['llm_base_url']} model={cfg['llm_model']} | "
            f"stt={cfg['stt_base_url']} model={cfg['stt_model']} | "
            f"tts={cfg['tts_base_url']} model={cfg['tts_model']} voice={cfg['tts_voice']}"
        )

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(audio_in_enabled=True, audio_out_enabled=True, audio_out_10ms_chunks=2),
    )

    stt = OpenAISTTService(api_key=cfg["stt_api_key"], base_url=cfg["stt_base_url"], model=cfg["stt_model"])
    llm = OpenAILLMService(api_key=cfg["llm_api_key"], base_url=cfg["llm_base_url"], model=cfg["llm_model"])

    context = LLMContext([{"role": "system", "content": cfg["system_instruction"]}])

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    raw_llm_spy = RawLLMMessageSpy(max_logs=80)
    spy = AfterLLMSpy(max_logs=250)
    buffer = TextBufferAndChunker()
    ctx_writer = AssistantContextWriter(context)

    tts = ChatterboxTTSProcessor(
        base_url=cfg["tts_base_url"],
        api_key=cfg["tts_api_key"],
        model=cfg["tts_model"],
        voice=cfg["tts_voice"],
        app_state=app_state,
    )

    processors = [transport.input(), stt]

    # non-invasive transcript exporter (only if AppState is present)
    if app_state:
        processors.append(TranscriptExporter(app_state))

    processors += [
        user_agg,
        llm,
        raw_llm_spy,
        spy,
        buffer,
    ]

    if app_state:
        processors.append(AssistantTextExporter(app_state))

    processors += [
        ctx_writer,
        tts,
        transport.output(),
        assistant_agg,
    ]

    pipeline = Pipeline(processors)

    task = PipelineTask(pipeline, params=PipelineParams(enable_metrics=True, enable_usage_metrics=True))

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Pipecat client connected")
        if app_state:
            await _set_phase(app_state, RuntimePhase.LISTENING)
        if GREET_ON_CONNECT:
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Pipecat client disconnected")
        if app_state:
            await _set_phase(app_state, RuntimePhase.DISCONNECTED)
        await task.cancel()

    try:
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
    finally:
        if app_state:
            app_state.active_sessions = max(0, app_state.active_sessions - 1)
