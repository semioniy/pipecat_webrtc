#!/bin/bash
docker remove -f Pipecat-WebRTC
docker build -t pipecat-webrtc:latest .
docker run -d --network=host --name Pipecat-WebRTC --restart unless-stopped pipecat-webrtc:latest
docker logs -f Pipecat-WebRTC

