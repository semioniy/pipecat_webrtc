#!/usr/bin/env python3
"""
Pipecat SmallWebRTC voice bot (STT -> LLM -> buffered TTS) for your local stack.

Fixes vs your current script:
- Conversation progression: writes assistant replies back into LLMContext (so history accumulates).
- Dedupe: only consumes one stream of LLM text fragments (default: message.type == "bot-llm-text").
- Chunking: buffers token fragments into sentence-ish chunks for natural TTS.
- Greeting: optional one-time kickoff on connect (LLMRunFrame) WITHOUT a sticky "greet me" user message.

Env (as you already use):
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

load_dotenv(override=True)


def must_env(name: str) -> str:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v.strip()


def clean_env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


SYSTEM_INSTRUCTION = clean_env_str(
    "SYSTEM_INSTRUCTION",
    "You are a low-latency voice assistant. Keep responses short and conversational. "
    "Avoid long lists unless asked. "
    "Greet the user once when the session starts. Do not introduce yourself again unless asked.",
)

# Chunking knobs (defaults are conservative; bump up for more natural speech)
TTS_MIN_CHARS = int(clean_env_str("TTS_MIN_CHARS", "90") or "90")
TTS_MAX_CHARS = int(clean_env_str("TTS_MAX_CHARS", "280") or "280")
TTS_FLUSH_TIMEOUT_MS = int(clean_env_str("TTS_FLUSH_TIMEOUT_MS", "850") or "850")
TTS_FLUSH_ON_PUNCT = clean_env_str("TTS_FLUSH_ON_PUNCT", "1") not in ("0", "false", "False")

# DEDUPE: only accept one transport stream type
ACCEPT_MESSAGE_TYPE = clean_env_str("ACCEPT_MESSAGE_TYPE", "bot-llm-text")

# Greet on connect by forcing a first LLM run (recommended for a nice UX)
GREET_ON_CONNECT = clean_env_str("GREET_ON_CONNECT", "1") not in ("0", "false", "False")


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


def extract_llm_text_fragment(msg: Any) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Returns (fragment, is_final, msg_type).

    IMPORTANT: We only accept message.type == ACCEPT_MESSAGE_TYPE by default,
    to avoid duplicates from bot-transcription or other streams.
    """
    if not isinstance(msg, dict):
        return None, False, None

    mtype = msg.get("type")
    if ACCEPT_MESSAGE_TYPE and mtype != ACCEPT_MESSAGE_TYPE:
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
        # keep tiny fragments like "!" or "’m"
        return text, final, mtype
    return None, final, mtype


@dataclass
class _BufState:
    buf: str = ""
    last_update_ms: int = 0
    last_flush_ms: int = 0


class TextBufferAndChunker(FrameProcessor):
    """
    Buffers token fragments into readable chunks and emits them as:
      OutputTransportMessageUrgentFrame(message={"label":"assistant.tts_text", ...})

    That chunk then gets:
      - spoken by TTS
      - written to LLMContext as assistant history (conversation progression)
    """

    def __init__(self):
        super().__init__()
        self._st = _BufState()

    def _check_started(self, frame: Frame) -> bool:
        return True

    def _now_ms(self) -> int:
        return int(time.time() * 1000)

    def _append(self, cur: str, frag: str) -> str:
        """
        Joining rules tuned for token streams:
        - Attach punctuation without space.
        - Attach leading apostrophes without space (I + ’m => I’m).
        - Collapse hyphen spacing later.
        """
        if not cur:
            return frag

        # keep em-dash, normalize other dashes
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
        # collapse spaces around hyphens produced by tokenization
        s = re.sub(r"\s*-\s*", "-", s)
        # collapse repeated spaces
        s = re.sub(r"[ \t]{2,}", " ", s)
        return s.strip()

    async def _emit_chunk(self, direction: FrameDirection, chunk: str, reason: str):
        chunk = self._normalize_chunk(chunk)
        if not chunk:
            return

        logger.info(f"[CHUNK] reason={reason} chars={len(chunk)} preview={chunk[:140]!r}")

        msg = {
            "label": "assistant.tts_text",
            "type": "assistant_text",
            "data": {"text": chunk, "final": True},
        }
        await self.push_frame(OutputTransportMessageUrgentFrame(message=msg), direction)

    async def _flush(self, direction: FrameDirection, reason: str):
        text = self._st.buf
        self._st.buf = ""
        self._st.last_flush_ms = self._now_ms()

        text = self._normalize_chunk(text)
        if not text:
            return

        # split into max-sized chunks
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

        # Opportunistic timeout flush
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

            # flush on punctuation once buffer is “sentence-like”
            if len(self._st.buf) >= TTS_MIN_CHARS and self._should_flush_on_punct(self._st.buf):
                await self._flush(direction, "punct+min")

        # flush on final only if there is remaining buffered text AND we didn't just flush
        if is_final and self._st.buf:
            if (now - self._st.last_flush_ms) > 150:
                await self._flush(direction, "final")


class AssistantContextWriter(FrameProcessor):
    """
    Ensures conversation progression by appending assistant text chunks into LLMContext.

    We listen for the chunker output:
      label == assistant.tts_text
    and append it as role=assistant.
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


class ChatterboxTTSProcessor(FrameProcessor):
    def __init__(self, base_url: str, api_key: str, model: str, voice: str):
        super().__init__()
        self._url = base_url.rstrip("/") + "/audio/speech"
        self._api_key = api_key.strip()
        self._model = model
        self._voice = voice
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(90.0))

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


async def run_bot(webrtc_connection):
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            audio_out_10ms_chunks=2,
        ),
    )

    stt = OpenAISTTService(
        api_key=must_env("STT_API_KEY"),
        base_url=must_env("STT_BASE_URL"),
        model=clean_env_str("STT_MODEL", "whisper-1"),
    )

    llm = OpenAILLMService(
        api_key=must_env("LLM_API_KEY"),
        base_url=must_env("LLM_BASE_URL"),
        model=clean_env_str("LLM_MODEL", "gpt-oss:latest"),
    )

    # IMPORTANT: Only system instruction in persistent context (no sticky "greet me" user message)
    context = LLMContext([{"role": "system", "content": SYSTEM_INSTRUCTION}])

    user_agg, assistant_agg = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    spy = AfterLLMSpy(max_logs=250)
    buffer = TextBufferAndChunker()
    ctx_writer = AssistantContextWriter(context)

    tts = ChatterboxTTSProcessor(
        base_url=must_env("TTS_BASE_URL"),
        api_key=clean_env_str("TTS_API_KEY", ""),
        model=clean_env_str("TTS_MODEL", "gpt-4o-mini-tts"),
        voice=clean_env_str("TTS_VOICE", "Frieren.wav"),
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_agg,
            llm,

            spy,          # show accepted LLM fragments
            buffer,       # buffer + chunk -> emits assistant.tts_text messages
            ctx_writer,   # write chunks into LLMContext so history progresses
            tts,

            transport.output(),
            assistant_agg,
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(_transport, _client):
        logger.info("Pipecat client connected")
        if GREET_ON_CONNECT:
            # One-time kickoff. If you don't want an automatic greeting, set GREET_ON_CONNECT=0.
            await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(_transport, _client):
        logger.info("Pipecat client disconnected")
        await task.cancel()

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)