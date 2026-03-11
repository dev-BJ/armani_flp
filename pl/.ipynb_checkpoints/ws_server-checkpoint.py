import sys
import os
sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        ".",
    )
)

import asyncio
import base64
import json
from PIL import Image
import io
import numpy as np
import cv2
import threading
import time
from typing import Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import yaml
import types
import tempfile
from omegaconf import OmegaConf
# from src.wrapper_trt import PersonaLive
from webcam.vid2vid_trt import Pipeline
from webcam.config import config
import torch
from types import SimpleNamespace
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pipeline = None
manager = None

# Configuration
MAX_QUEUE_SIZE = 3  # Keep low for real-time (frames older than 3 are stale)
PROCESSING_TIMEOUT = 5.0  # Seconds
FRAME_QUALITY = 85  # JPEG quality (lower = faster encode)
    
def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes))#.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def bytes_to_tensor(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    np_img = np.asarray(image)
    tensor = torch.from_numpy(np_img.copy())
    return tensor

def pil_to_frame(image: Image.Image) -> bytes:
    frame_data = io.BytesIO()
    image.save(frame_data, format="JPEG")
    return frame_data.getvalue()

@dataclass
class FramePacket:
    """Structured frame data for queue processing"""
    frame_num: int
    timestamp: float
    data: bytes
    websocket: WebSocket  # Reference for response routing

class FrameProcessor:
    """Dedicated processor with input/output queues"""
    def __init__(self, client_id: str, pipeline, maxsize: int = MAX_QUEUE_SIZE):
        self.client_id = client_id
        self.pipeline = pipeline
        self.input_queue: asyncio.Queue[FramePacket] = asyncio.Queue(maxsize=maxsize)
        # self.latest_packet: Optional[FramePacket] = None
        self._packet_event = asyncio.Event()
        # self.output_queue: asyncio.Queue[Tuple[FramePacket, Optional[np.ndarray], float]] = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self.dropped_frames = 0
        self.processed_frames = 0
        self._initialized = asyncio.Event()  # Signal when ready
        self.source_set: bool = False
        
    async def start(self):
        """Start the processing loop"""
        self._running = True
        self._task = asyncio.create_task(self._processing_loop())
        logger.info(f"Processor started for {self.client_id}")

    async def stop(self):
        """Graceful shutdown"""
        self._running = False
        self._packet_event.set()  # wake processing loop
    
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
        logger.info(f"Processor stopped for {self.client_id}")

    async def submit(self, packet: FramePacket) -> bool:
        # self.latest_packet = packet
        # self._packet_event.set()
        input_tensor = bytes_to_tensor(packet.data)
        params = SimpleNamespace()
        params.image = input_tensor
        self.pipeline.accept_new_params(params)
        return True

    async def _processing_loop(self):
        while self._running:
            images = self.pipeline.produce_outputs()
    
            if not images:
                await asyncio.sleep(0.001)
                continue
    
            frames = list(map(pil_to_frame, images))
    
            for frame in frames:
                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
    
                await self.output_queue.put(frame)
        

class QueuedPipelineManager:
    """Manager with per-client frame processors"""
    def __init__(self, pipeline: Pipeline):
        self._pipeline = pipeline
        self.pipelines: Dict[str, any] = {}
        self.processors: Dict[str, FrameProcessor] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        # self.source_set: bool = False  # Fixed: was Bool = 
        self._lock = threading.Lock()
        self._frame_counters: Dict[str, int] = {}
        
    async def get_or_create_processor(self, client_id: str):

        with self._lock:
            if client_id not in self.processors:
                logger.info(f"Creating processor for {client_id}")
    
                processor = FrameProcessor(client_id, self._pipeline)
                await processor.start()
    
                self.pipelines[client_id] = self._pipeline
                self.processors[client_id] = processor
                self.locks[client_id] = asyncio.Lock()
                self._frame_counters[client_id] = 0
    
        return self.processors[client_id]
    
    def get_lock(self, client_id: str) -> asyncio.Lock:
        with self._lock:
            if client_id not in self.locks:
                self.locks[client_id] = asyncio.Lock()
            return self.locks[client_id]
    
    async def remove_client(self, client_id: str):
        """Cleanup client resources"""
        with self._lock:
            if client_id in self.processors:
                await self.processors[client_id].stop()
                del self.processors[client_id]
            if client_id in self.pipelines:
                # self.pipelines[client_id].close()
                del self.pipelines[client_id]
            if client_id in self.locks:
                del self.locks[client_id]
            if client_id in self._frame_counters:
                del self._frame_counters[client_id]
            logger.info(f"Removed client {client_id}")

# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, manager

    logger.info("Initializing pipeline...")

    pipeline = Pipeline(config, device=device)
    manager = QueuedPipelineManager(pipeline)

    yield

    logger.info("Shutting down pipeline...")

    if pipeline:
        pipeline.close()

    pipeline = None
    manager = None

app = FastAPI(title="FasterLivePortrait Queued WebSocket Server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def hello():
    return {
        "msg": "FasterLivePortrait Queued WebSocket Server Running"
    }

@app.get("/metrics")
async def metrics():
    return {
        "clients": {
            cid: {
                "output_qsize": proc.output_queue.qsize(),
                "processed": proc.processed_frames,
                "dropped": proc.dropped_frames,
                "mode": "latest_frame"
            }
            for cid, proc in manager.processors.items()
        }
    }

@app.get("/ping")
async def ping():
    return {"status": "healthy"}
    
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    logger.info(f"Client {client_id} connected")
    
    processor: Optional[FrameProcessor] = None
    sender_task: Optional[asyncio.Task] = None
    sender_ready = asyncio.Event()  # Signal when processor is ready

    async def output_sender():
        """Dedicated task to send results back to client"""
        try:
            # Wait for processor to be initialized
            await asyncio.wait_for(sender_ready.wait(), timeout=30.0)
            
            while True:
                # Wait for processed frame with timeout
                result = None
                
                # Get and send
                result = await processor.output_queue.get()

                if result is None:
                    await websocket.send_json({
                        "type": "error",
                        "frame_num": 0.0,
                        "msg": "Processing failed"
                    })
                    continue
                await websocket.send_bytes(result)
                
        except asyncio.TimeoutError:
            # No output for 1 second, check if still running
            logger.warning(f"Sender timeout for {client_id}")
        except Exception as e:
            logger.error(f"Sender error for {client_id}: {e}")
    
    try:
        while True:
            # Receive with timeout for heartbeat
            try:
                data = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=30.0
                )
                # logger.info(f"RAW DATA: {data}")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "ping"})
                continue
            except RuntimeError:
                break

            if data.get("type") is not None and data["type"] == "websocket.disconnect":
                break
            
            # Handle text commands
            if data.get("text") is not None:
                msg = json.loads(data["text"])
                cmd = msg.get("cmd")
                
                if cmd == "init":
                    try:
                        img_b64 = msg.get("image", "")
                        if not img_b64:
                            await websocket.send_json({"type": "error", "msg": "No image"})
                            continue
                        
                        img_bytes = base64.b64decode(img_b64)
                        
                        if img_bytes is None:
                            await websocket.send_json({"type": "error", "msg": "Invalid image"})
                            continue
                        
                        # Get or create processor (creates pipeline)
                        processor = await manager.get_or_create_processor(client_id)
                        # logger.info("Processor VAR")

                        try:
                            source_img = bytes_to_pil(img_bytes)
                            processor.pipeline.fuse_reference(source_img)
                        except Exception as e:
                            logger.error(f"Source set error: {e}")
                            await websocket.send_json({"type": "Source set rror", "msg": str(e)})
                        processor.source_set = True

                        logger.info(f"Source Set {processor.source_set}")
                        
                        if not processor.source_set:
                            await websocket.send_json({"type": "error", "msg": "Source init failed"})
                            continue
                        
                        # Start sender task AFTER processor is ready
                        sender_task = asyncio.create_task(output_sender())
                        sender_ready.set()  # Signal sender to start processing

                        if sender_task is None:
                            await websocket.send_json({
                                "type": "error",
                                "msg": "Server not ready"
                            })
                            continue
                        
                        await websocket.send_json({
                            "type": "ready",
                            "shape": source_img.size,
                            "max_queue": MAX_QUEUE_SIZE
                        })
                        
                    except Exception as e:
                        logger.error(f"Init error: {e}")
                        await websocket.send_json({"type": "error", "msg": str(e)})
                
                elif cmd == "reset":
                    if processor:
                        await manager.remove_client(client_id)
                        processor = None
                        pipeline.reset()
                        sender_ready.clear()
                        if sender_task:
                            sender_task.cancel()
                            try:
                                await sender_task
                            except asyncio.CancelledError:
                                pass
                            sender_task = None
                    await websocket.send_json({"type": "reset"})
                    
                elif cmd == "stats":
                    if processor:
                        await websocket.send_json({
                            "type": "stats",
                            "processed": processor.processed_frames,
                            "dropped": processor.dropped_frames,
                            # "queue_size": processor.input_queue.qsize(),
                            "queue_mode": "latest_frame",
                            "pending_outputs": processor.output_queue.qsize()
                        })
            
            # Handle binary frame data
            elif data.get("bytes") is not None:
                data = data["bytes"]
                if processor is None or not processor.source_set:
                    await websocket.send_json({"type": "error", "msg": "Not initialized"})
                    continue
                
                # Atomic frame counter increment
                lock = manager.get_lock(client_id)
                async with lock:
                    manager._frame_counters[client_id] += 1
                    frame_num = manager._frame_counters[client_id]
                
                # Create packet and submit to queue
                packet = FramePacket(
                    frame_num=frame_num,
                    timestamp=time.perf_counter(),
                    data=data,
                    websocket=websocket
                )
                
                accepted = await processor.submit(packet)
                
                if not accepted:
                    # Queue full, notify client
                    await websocket.send_json({
                        "type": "dropped",
                        "frame_num": frame_num,
                        "reason": "queue_full",
                        "queue_size": MAX_QUEUE_SIZE
                    })
            else:
                logger.error("Unknown Type")
                
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if sender_task:
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass
        await manager.remove_client(client_id)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    uvicorn.run(
        "ws_server:app",   # string import path is safer with multiprocessing
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        access_log=None,
        reload=True
    )
