#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from loguru import logger
from pipecat.transports.smallwebrtc.request_handler import (
    SmallWebRTCPatchRequest,
    SmallWebRTCRequest,
    SmallWebRTCRequestHandler,
)

from bot import run_bot

# Load environment variables
load_dotenv(override=True)

app = FastAPI()

# Initialize the SmallWebRTC request handler
small_webrtc_handler: SmallWebRTCRequestHandler = SmallWebRTCRequestHandler()


# -----------------------------
# Minimal server-owned event hub
# -----------------------------
class EventHub:
    def __init__(self):
        self.subs: List[asyncio.Queue] = []
        self.last_status: Dict[str, Any] = {
            "phase": "idle",
            "services": {"llm": None, "stt": None, "tts": None},
            "transcript": [],
        }

    async def publish(self, evt: Dict[str, Any]) -> None:
        # Update snapshot for /api/status (best-effort)
        t = evt.get("type")
        if t == "phase":
            self.last_status["phase"] = evt.get("phase", self.last_status["phase"])
        elif t == "health":
            self.last_status["services"] = evt.get("services", self.last_status["services"])
        elif t == "transcript":
            # keep last 200 lines
            self.last_status["transcript"].append(evt.get("line", ""))
            self.last_status["transcript"] = [x for x in self.last_status["transcript"] if x][-200:]

        dead = []
        for q in self.subs:
            try:
                q.put_nowait(evt)
            except Exception:
                dead.append(q)
        if dead:
            self.subs = [q for q in self.subs if q not in dead]


app.state.hub = EventHub()


@app.get("/api/status")
async def api_status():
    # Snapshot for UI on load/reconnect
    return JSONResponse(app.state.hub.last_status)


@app.get("/api/events")
async def api_events():
    """
    Server-Sent Events stream.
    Bot can publish to app.state.hub.publish(...) if you wire it in later.
    """
    hub: EventHub = app.state.hub
    q: asyncio.Queue = asyncio.Queue(maxsize=200)
    hub.subs.append(q)

    async def gen():
        # initial snapshot
        yield f"data: {json.dumps({'type':'init', **hub.last_status}, ensure_ascii=False)}\n\n"
        try:
            while True:
                evt = await q.get()
                yield f"data: {json.dumps(evt, ensure_ascii=False)}\n\n"
        finally:
            try:
                hub.subs.remove(q)
            except ValueError:
                pass

    return StreamingResponse(gen(), media_type="text/event-stream")


# -----------------------------
# WebRTC endpoints (unchanged)
# -----------------------------
@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""

    async def webrtc_connection_callback(connection):
        # ✅ KEEP EXACTLY AS BEFORE so we don't break anything:
        background_tasks.add_task(run_bot, connection)

    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    logger.debug(f"Received patch request: {request}")
    await small_webrtc_handler.handle_patch_request(request)
    return {"status": "success"}


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    await small_webrtc_handler.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC demo (plus minimal SSE/status)")
    parser.add_argument("--host", default="localhost", help="Host for HTTP server (default: localhost)")
    parser.add_argument("--port", type=int, default=7860, help="Port for HTTP server (default: 7860)")
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="TRACE")
    else:
        logger.add(sys.stderr, level="DEBUG")

    uvicorn.run(app, host=args.host, port=args.port)