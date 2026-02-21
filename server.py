#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import sys
import os

import httpx
from contextlib import asynccontextmanager

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
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



def _clean_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip().strip('"').strip("'")


SERVICE_URLS = {
    "llm": _clean_env("LLM_BASE_URL", ""),
    "stt": _clean_env("STT_BASE_URL", ""),
    "tts": _clean_env("TTS_BASE_URL", ""),
}


async def _probe_service(client: httpx.AsyncClient, base_url: str) -> str:
    if not base_url:
        return "unknown"

    targets = [base_url.rstrip("/") + "/models", base_url.rstrip("/")]
    for url in targets:
        try:
            r = await client.get(url)
            if r.status_code < 500:
                return "ok"
        except Exception:
            continue
    return "down"



@app.get("/api/health")
async def api_health():
    async with httpx.AsyncClient(timeout=httpx.Timeout(2.5)) as client:
        llm = await _probe_service(client, SERVICE_URLS["llm"])
        stt = await _probe_service(client, SERVICE_URLS["stt"])
        tts = await _probe_service(client, SERVICE_URLS["tts"])

    return JSONResponse({
        "health": {"llm": llm, "stt": stt, "tts": tts},
        "services": SERVICE_URLS,
    })


@app.post("/api/offer")
async def offer(request: SmallWebRTCRequest, background_tasks: BackgroundTasks):
    """Handle WebRTC offer requests via SmallWebRTCRequestHandler."""

    # Prepare runner arguments with the callback to run your bot
    async def webrtc_connection_callback(connection):
        background_tasks.add_task(run_bot, connection)

    # Delegate handling to SmallWebRTCRequestHandler
    answer = await small_webrtc_handler.handle_web_request(
        request=request,
        webrtc_connection_callback=webrtc_connection_callback,
    )
    return answer


@app.patch("/api/offer")
async def ice_candidate(request: SmallWebRTCPatchRequest):
    logger.trace(f"Received patch request: {request}")
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
    parser = argparse.ArgumentParser(description="WebRTC demo")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    parser.add_argument("--verbose", "-v", action="count")
    args = parser.parse_args()

    logger.remove(0)
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    uvicorn.run(app, host=args.host, port=args.port)
