# app_state.py
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Set


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class BotConfig:
    # Seed from .env on startup; UI can update later
    llm_base_url: str
    llm_api_key: str
    llm_model: str

    stt_base_url: str
    stt_api_key: str
    stt_model: str

    tts_base_url: str
    tts_api_key: str
    tts_model: str
    tts_voice: str

    system_instruction: str

    version: int = 1


@dataclass
class HealthStatus:
    llm: str = "unknown"   # ok|down|degraded|unknown
    stt: str = "unknown"
    tts: str = "unknown"
    updated_ms: int = 0
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuntimeState:
    phase: str = "idle"  # idle|listening|waiting_llm|speaking|disconnected
    updated_ms: int = 0


class EventBus:
    """
    Simple SSE fanout:
      - each client gets its own asyncio.Queue
      - publish() puts events on all queues
    """
    def __init__(self):
        self._queues: Set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        async with self._lock:
            self._queues.add(q)
        return q

    async def unsubscribe(self, q: asyncio.Queue) -> None:
        async with self._lock:
            self._queues.discard(q)

    async def publish(self, event: Dict[str, Any]) -> None:
        async with self._lock:
            queues = list(self._queues)
        for q in queues:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # drop oldest behavior: just skip
                pass


class AppState:
    def __init__(self, config: BotConfig):
        self.config = config
        self.health = HealthStatus()
        self.runtime = RuntimeState()

        self.transcript_lines: List[str] = []
        self.max_transcript_lines = 5000

        self.events = EventBus()

        # Simple counter
        self.active_sessions = 0

    def to_status(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "health": asdict(self.health),
            "runtime": asdict(self.runtime),
            "active_sessions": self.active_sessions,
        }

    def append_transcript(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        self.transcript_lines.append(line)
        if len(self.transcript_lines) > self.max_transcript_lines:
            self.transcript_lines = self.transcript_lines[-self.max_transcript_lines :]