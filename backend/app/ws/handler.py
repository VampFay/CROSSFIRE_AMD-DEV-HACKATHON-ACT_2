"""
WebSocket handler for live job status updates.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

router = APIRouter()


# Per-job subscriber sets
_subscribers: Dict[str, Set[WebSocket]] = {}


@router.websocket("/ws/jobs/{job_id}")
async def job_websocket(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for live job status updates.

    Client connects to /ws/jobs/{job_id} and receives JSON messages:
        {"type": "status", "job_id": "...", "data": {"status": "translating", "iteration": 1}}
        {"type": "log", "job_id": "...", "data": {"message": "Compiling..."}}
        {"type": "result", "job_id": "...", "data": {...full result...}}
        {"type": "error", "job_id": "...", "data": {"message": "..."}}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: job_id={job_id}")

    if job_id not in _subscribers:
        _subscribers[job_id] = set()
    _subscribers[job_id].add(websocket)

    try:
        while True:
            # Keep connection open; client can also send messages if needed
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                logger.debug(f"WS recv {job_id}: {msg}")
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: job_id={job_id}")
    finally:
        if job_id in _subscribers:
            _subscribers[job_id].discard(websocket)
            if not _subscribers[job_id]:
                del _subscribers[job_id]


async def broadcast(job_id: str, message_type: str, data: dict):
    """Broadcast a message to all subscribers of a job."""
    if job_id not in _subscribers:
        return

    msg = {
        "type": message_type,
        "job_id": job_id,
        "data": data,
        "timestamp": datetime.utcnow().isoformat(),
    }

    dead: list = []
    for ws in _subscribers[job_id]:
        try:
            await ws.send_json(msg)
        except Exception as e:
            logger.warning(f"Failed to send WS message: {e}")
            dead.append(ws)

    for ws in dead:
        _subscribers[job_id].discard(ws)
