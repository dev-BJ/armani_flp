import asyncio
import base64
import json
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
import sys
import logging
import yaml
import types
import tempfile
import os
from omegaconf import OmegaConf

sys.path.append("../flp")
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MAX_QUEUE_SIZE = 5  # Keep low for real-time (frames older than 3 are stale)
PROCESSING_TIMEOUT = 5.0  # Seconds
FRAME_QUALITY = 85  # JPEG quality (lower = faster encode)

# def dict_to_object(d):
#     if not isinstance(d, dict):
#         return d
#     obj = types.SimpleNamespace()
#     for k, v in d.items():
#         setattr(obj, k, dict_to_object(v))
#     return obj

# with open("configs/trt_infer.yaml", "r") as f:
#     raw_cfg = yaml.safe_load(f)

# cfg = dict_to_object(raw_cfg)
# cfg.models = raw_cfg['models']
# cfg.infer_params = raw_cfg['infer_params']
cfg = OmegaConf.load("configs/trt_infer.yaml")

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
        # self.input_queue: asyncio.Queue[FramePacket] = asyncio.Queue(maxsize=maxsize)
        self.latest_packet: Optional[FramePacket] = None
        self._packet_event = asyncio.Event()
        self.output_queue: asyncio.Queue[Tuple[FramePacket, Optional[np.ndarray], float]] = asyncio.Queue()
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
        
    # async def stop(self):
    #     """Graceful shutdown"""
    #     self._running = False
    #     self._initialized.set()  # Unblock any waiters
    #     if self._task:
    #         self._task.cancel()
    #         try:
    #             await self._task
    #         except asyncio.CancelledError:
    #             pass
    #     # Drain queues
    #     while not self.input_queue.empty():
    #         try:
    #             self.input_queue.get_nowait()
    #         except asyncio.QueueEmpty:
    #             break
    #     logger.info(f"Processor stopped for {self.client_id}")

    async def submit(self, packet: FramePacket) -> bool:
        self.latest_packet = packet
        self._packet_event.set()
        return True
        
    # async def submit(self, packet: FramePacket) -> bool:
    #     """Submit frame for processing. Returns False if queue full (drop frame)"""
    #     try:
    #         self.input_queue.put_nowait(packet)
    #         return True
    #     except asyncio.QueueFull:
    #         self.dropped_frames += 1
    #         if self.dropped_frames % 30 == 1:
    #             logger.warning(f"Client {self.client_id}: Dropped {self.dropped_frames} frames (queue full)")
    #         return False

    async def _processing_loop(self):
        loop = asyncio.get_running_loop()
    
        while self._running:
            await self._packet_event.wait()
            self._packet_event.clear()
    
            packet = self.latest_packet
            if packet is None:
                continue
    
            start_proc = time.perf_counter()
    
            try:
                driving_frame = cv2.imdecode(
                    np.frombuffer(packet.data, np.uint8),
                    cv2.IMREAD_COLOR
                )
    
                # result = await loop.run_in_executor(
                #     None,
                #     lambda: self.pipeline.run(
                #         driving_frame,
                #         self.pipeline.src_imgs[0],
                #         self.pipeline.src_infos[0],
                #         first_frame=(packet.frame_num == 0)
                #     )
                # )

                dri_crop, out_crop, out_org, dri_motion_info = self.pipeline.run(driving_frame, self.pipeline.src_imgs[0], self.pipeline.src_infos[0], first_frame=(self.processed_frames == 0))

                if out_org is None:
                    print(f"no face in driving frame:{self.processed_frames}")
                    continue
                result = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)

                # Handle tuple or list return
                # if isinstance(result, (tuple, list)):
                #     if len(result) == 0:
                #         result = None
                #     else:
                #         result = result[0]
                
                if result is None:
                    await websocket.send_json({
                        "type": "error",
                        "frame_num": packet.frame_num,
                        "msg": "Processing returned None"
                    })
                    continue
                
                # Ensure it's np.ndarray
                if not isinstance(result, np.ndarray):
                    logger.error(f"Frame {packet.frame_num}: Invalid result type {type(result)}")
                    await websocket.send_json({
                        "type": "error",
                        "frame_num": packet.frame_num,
                        "msg": f"Invalid result type {type(result)}"
                    })
                    continue
                
                # Convert to uint8 if necessary
                if result.dtype != np.uint8:
                    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
                proc_time = time.perf_counter() - start_proc
                self.processed_frames += 1
                await self.output_queue.put((packet, result, proc_time))
    
            except Exception as e:
                logger.error(f"Processing error: {e}")
            
    # async def _processing_loop(self):
    #     """Main processing loop - runs in dedicated task"""
    #     loop = asyncio.get_running_loop()
        
    #     while self._running:
    #         try:
    #             # Get frame with timeout to allow shutdown checks
    #             packet = await asyncio.wait_for(
    #                 self.input_queue.get(), 
    #                 timeout=0.5
    #             )
    #         except asyncio.TimeoutError:
    #             continue
                
    #         start_proc = time.perf_counter()
            
    #         try:
    #             # Decode
    #             driving_frame = cv2.imdecode(
    #                 np.frombuffer(packet.data, np.uint8), 
    #                 cv2.IMREAD_COLOR
    #             )
                
    #             if driving_frame is None:
    #                 await self.output_queue.put((packet, None, 0.0))
    #                 continue
                
    #             # Process in thread pool (CPU/GPU intensive)
    #             result = await loop.run_in_executor(
    #                 None,  # Uses default executor
    #                 lambda: self.pipeline.run(
    #                     driving_frame,
    #                     self.pipeline.src_imgs[0],
    #                     self.pipeline.src_infos[0],
    #                     first_frame=(packet.frame_num == 0)
    #                 )
    #             )
                
    #             proc_time = time.perf_counter() - start_proc
    #             self.processed_frames += 1
    #             await self.output_queue.put((packet, result, proc_time))
                
    #         except Exception as e:
    #             logger.error(f"Processing error for {self.client_id}: {e}")
    #             await self.output_queue.put((packet, None, 0.0))
    #         finally:
    #             self.input_queue.task_done()

class QueuedPipelineManager:
    """Manager with per-client frame processors"""
    def __init__(self):
        self.pipelines: Dict[str, any] = {}
        self.processors: Dict[str, FrameProcessor] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
        # self.source_set: bool = False  # Fixed: was Bool = 
        self._lock = threading.Lock()
        self._frame_counters: Dict[str, int] = {}
        
    async def get_or_create_processor(self, client_id: str) -> FrameProcessor:
        """Get existing or create new processor with pipeline"""
        # logger.info(f"Before creating processor for {client_id}")
        with self._lock:
            if client_id not in self.processors:
                logger.info(f"Creating processor for {client_id}")
                
                # Create pipeline (sync, potentially slow)
                loop = asyncio.get_running_loop()
                # pipeline = await loop.run_in_executor(
                #     None,
                #     lambda: FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
                # )

                pipeline = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
                
                processor = FrameProcessor(client_id, pipeline)
                await processor.start()
                
                self.pipelines[client_id] = pipeline
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
                del self.pipelines[client_id]
            if client_id in self.locks:
                del self.locks[client_id]
            if client_id in self._frame_counters:
                del self._frame_counters[client_id]
            logger.info(f"Removed client {client_id}")

# Global manager
manager = QueuedPipelineManager()

app = FastAPI(title="FasterLivePortrait Queued WebSocket Server")
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

# @app.get("/metrics")
# async def metrics():
#     return {
#         "clients": {
#             cid: {
#                 "input_qsize": proc.input_queue.qsize(),
#                 "output_qsize": proc.output_queue.qsize(),
#                 "processed": proc.processed_frames,
#                 "dropped": proc.dropped_frames,
#                 "drop_rate": proc.dropped_frames / (proc.processed_frames + proc.dropped_frames) if (proc.processed_frames + proc.dropped_frames) > 0 else 0
#             }
#             for cid, proc in manager.processors.items()
#         }
#     }

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
                # packet, result, proc_time = await asyncio.wait_for(
                #     processor.output_queue.get(),
                #     timeout=1.0
                # )
                packet, result, proc_time = await processor.output_queue.get()
                
                if result is None:
                    await websocket.send_json({
                        "type": "error",
                        "frame_num": packet.frame_num,
                        "msg": "Processing failed"
                    })
                    continue
                
                # Encode and send
                encode_start = time.perf_counter()
                _, encoded = cv2.imencode(
                    '.jpg', 
                    result, 
                    [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_QUALITY]
                )
                encode_time = (time.perf_counter() - encode_start) * 1000
                
                total_time = (time.perf_counter() - packet.timestamp) * 1000
                
                # Send metadata
                await websocket.send_json({
                    "type": "frame_meta",
                    "frame_num": packet.frame_num,
                    "proc_time_ms": round(proc_time * 1000, 2),
                    "encode_time_ms": round(encode_time, 2),
                    "total_latency_ms": round(total_time, 2),
                    # "queue_size": processor.input_queue.qsize(),
                    "queue_mode": "latest_frame",
                    "proc_nums": processor.processed_frames,
                    "dropped_frames": processor.dropped_frames
                })
                
                # Send binary
                await websocket.send_bytes(encoded.tobytes())
                
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
                        source_img = cv2.imdecode(
                            np.frombuffer(img_bytes, np.uint8), 
                            cv2.IMREAD_COLOR
                        )
                        
                        if source_img is None:
                            await websocket.send_json({"type": "error", "msg": "Decode failed"})
                            continue
                        
                        # Get or create processor (creates pipeline)
                        processor = await manager.get_or_create_processor(client_id)
                        # logger.info("Processor VAR")

                        temp_path = None
                        
                        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                            temp_path = tmp.name
                            cv2.imwrite(temp_path, source_img)
                        
                        # Set source
                        loop = asyncio.get_running_loop()
                        # processor.source_set = await loop.run_in_executor(
                        #     None,
                        #     processor.pipeline.prepare_source,
                        #     temp_path
                        # )

                        processor.source_set = processor.pipeline.prepare_source(temp_path, realtime=True)

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
                            "shape": source_img.shape,
                            "max_queue": MAX_QUEUE_SIZE
                        })
                        
                    except Exception as e:
                        logger.error(f"Init error: {e}")
                        await websocket.send_json({"type": "error", "msg": str(e)})
                
                elif cmd == "reset":
                    if processor:
                        await manager.remove_client(client_id)
                        processor = None
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
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        loop="asyncio",
        # log_level="info",
        access_log=None
    )