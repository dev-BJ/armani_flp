
import asyncio
import base64
import json
import numpy as np
import cv2
import threading
import time
from typing import Dict, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import sys
import logging
import nest_asyncio
import yaml
import types

# Apply nest_asyncio for Colab compatibility
nest_asyncio.apply()

def dict_to_object(d):
    if not isinstance(d, dict):
        return d
    obj = types.SimpleNamespace()
    for k, v in d.items():
        setattr(obj, k, dict_to_object(v))
    return obj

with open("configs/trt_infer.yaml", "r") as f:
    raw_cfg = yaml.safe_load(f)

cfg = dict_to_object(raw_cfg)
cfg.models = raw_cfg['models']

sys.path.append("../flp")
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncPipelineManager:
    """Thread-safe pipeline manager with async support"""
    def __init__(self):
        self.pipelines: Dict[str, FasterLivePortraitPipeline] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        self._lock = threading.Lock()
        self.source_set = False
        self.frame_count = 0

    def get_or_create_pipeline(self, client_id: str) -> FasterLivePortraitPipeline:
        with self._lock:
            if client_id not in self.pipelines:
                logger.info(f"Creating pipeline for {client_id}")
                self.pipelines[client_id] = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
                self.locks[client_id] = asyncio.Lock()
            return self.pipelines[client_id]

    def get_lock(self, client_id: str) -> asyncio.Lock:
        with self._lock:
            if client_id not in self.locks:
                self.locks[client_id] = asyncio.Lock()
            return self.locks[client_id]

    def remove_pipeline(self, client_id: str):
        with self._lock:
            if client_id in self.pipelines:
                del self.pipelines[client_id]
            if client_id in self.locks:
                del self.locks[client_id]
            logger.info(f"Removed pipeline for {client_id}")

# Global manager
manager = AsyncPipelineManager()

# FastAPI app
app = FastAPI(title="FasterLivePortrait WebSocket Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_pipelines": len(manager.pipelines),
        "timestamp": time.time()
    }

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"Client {client_id} connected from {websocket.client}")

    pipeline: Optional[FasterLivePortraitPipeline] = None

    try:
        while True:
            # Receive message (text or binary)
            try:
                data = await websocket.receive()
            except RuntimeError:
                # Connection closed
                break

            # Handle text messages (JSON commands)
            if isinstance(data, str):
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "msg": "Invalid JSON"})
                    continue

                cmd = msg.get("cmd")

                if cmd == "init":
                    # Initialize with source image
                    try:
                        img_b64 = msg.get("image", "")
                        if not img_b64:
                            await websocket.send_json({"type": "error", "msg": "No image provided"})
                            continue

                        img_bytes = base64.b64decode(img_b64)
                        source_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

                        if source_img is None:
                            await websocket.send_json({"type": "error", "msg": "Failed to decode image"})
                            continue

                        # Get pipeline (thread-safe)
                        pipeline = manager.get_or_create_pipeline(client_id)

                        # Set source (this is synchronous, run in executor to not block)
                        loop = asyncio.get_event_loop()
                        pipeline.source_set = await loop.run_in_executor(None, pipeline.prepare_source, source_img)
                        # source_set = True

                        await websocket.send_json({
                            "type": "ready",
                            "msg": "Source image set successfully",
                            "shape": source_img.shape
                        })
                        logger.info(f"Source set for {client_id}: {source_img.shape}")

                    except Exception as e:
                        logger.error(f"Init error: {e}")
                        await websocket.send_json({"type": "error", "msg": f"Init failed: {str(e)}"})

                elif cmd == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})

                elif cmd == "reset":
                    manager.remove_pipeline(client_id)
                    pipeline = None
                    # source_set = False
                    # frame_count = 0
                    await websocket.send_json({"type": "reset", "msg": "Pipeline reset"})

                elif cmd == "config":
                    # Update config if needed
                    if pipeline:
                        config = msg.get("config", {})
                        # Apply config updates here if pipeline supports it
                        await websocket.send_json({"type": "config", "config": config})

                else:
                    await websocket.send_json({"type": "error", "msg": f"Unknown command: {cmd}"})

            # Handle binary messages (driving frames)
            else:
                if not pipeline.source_set or pipeline is None and len(pipeline.src_infos) > 0:
                    await websocket.send_json({"type": "error", "msg": "Send init command first"})
                    continue

                pipeline.frame_count += 1
                start_time = time.time()

                try:
                    # Decode driving frame
                    driving_frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if driving_frame is None:
                        await websocket.send_json({"type": "error", "msg": "Failed to decode frame"})
                        continue

                    # Process frame with lock to prevent concurrent access to same pipeline
                    lock = manager.get_lock(client_id)
                    async with lock:
                        # Run inference in thread pool to not block event loop
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            lambda: pipeline.run(driving_frame, pipeline.src_imgs[0], pipeline.src_infos[0], first_frame=True)
                        )

                    if result is None:
                        await websocket.send_json({"type": "error", "msg": "Processing returned None"})
                        continue

                    # Encode result
                    encode_start = time.time()
                    _, encoded = cv2.imencode('.jpg', result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    encode_time = (time.time() - encode_start) * 1000

                    # Send metadata first
                    total_time = (time.time() - start_time) * 1000
                    await websocket.send_json({
                        "type": "frame_meta",
                        "frame_num": pipeline.frame_count,
                        "total_time_ms": round(total_time, 2),
                        "encode_time_ms": round(encode_time, 2),
                        "shape": result.shape
                    })

                    # Send binary data
                    await websocket.send_bytes(encoded.tobytes())

                    # Log every 30 frames
                    if pipeline.frame_count % 30 == 0:
                        logger.info(f"Client {client_id}: Processed {pipeline.frame_count} frames, last: {total_time:.1f}ms")

                except Exception as e:
                    logger.error(f"Frame processing error: {e}")
                    await websocket.send_json({"type": "error", "msg": str(e)})

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        manager.remove_pipeline(client_id)
        try:
            await websocket.close()
        except:
            pass

# Server runner with proper async support for Colab
class ColabServer:
    def __init__(self, host="0.0.0.0", port=9870):
        self.host = host
        self.port = port
        self.server = None
        self.loop = None
        self.thread = None

    def start(self):
        """Start server in a way that's compatible with Colab's event loop"""
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
            loop="asyncio"
        )
        self.server = uvicorn.Server(config)

        def run_server():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.server.serve())

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
        logger.info(f"Server started on {self.host}:{self.port}")

    def stop(self):
        if self.server:
            self.server.should_exit = True
        if self.thread:
            self.thread.join(timeout=5)

# Global server instance
_server = None

def start_server():
    global _server
    if _server is None:
        _server = ColabServer()
        _server.start()
    return _server

def stop_server():
    global _server
    if _server:
        _server.stop()
        _server = None

if __name__ == "__main__":
    start_server()
