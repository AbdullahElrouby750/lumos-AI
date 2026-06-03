"""
Lumos Nervous System - FastAPI Server Core

Unified asynchronous FastAPI server handling WebSocket signaling and RESTful file serving.
Runs on port 5000 with Uvicorn, optimized for Raspberry Pi 4 hardware constraints.
"""

import asyncio
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from src.network.nova_discovery import LumosDiscovery
from src.network.nova_network_models import BaseEvent, Event

# Configure logging
logger = logging.getLogger(__name__)

# Hardware-aware paths (Pi 4 optimized)
# --- BUG X-1 FIX: Point server to the project root ---
BRAIN_FILE_PATH = Path("nova_brain.pkl") 
# -----------------------------------------------------
SCENE_DIR_PATH = Path("/dev/shm")  # RAM disk for transient images
LATEST_SCENE_FILE = SCENE_DIR_PATH / "latest_scene.jpg"


class LumosServer:
    """
    FastAPI server core for the Lumos Nervous System.

    Handles WebSocket event multiplexing and REST file serving.
    Runs asynchronously in its own thread to avoid blocking the Vision Pipeline.
    """

    def __init__(self):
        """Initialize the FastAPI server with WebSocket and REST endpoints."""
        self.app = FastAPI(title="Lumos Nervous System", version="3.1")
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.active_connections: List[WebSocket] = []
        self.discovery = LumosDiscovery()
        self._server_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.command_callback: Optional[Callable[[dict[str, Any]], None]] = None

        # Setup routes
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Configure FastAPI routes for WebSocket and REST endpoints."""

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            """WebSocket endpoint for real-time event multiplexing."""
            await websocket.accept()
            self.active_connections.append(websocket)
            logger.info(f"WebSocket connection established. Total connections: {len(self.active_connections)}")

            try:
                send_task = asyncio.create_task(self._send_loop(websocket))
                receive_task = asyncio.create_task(self._receive_loop(websocket))

                done, pending = await asyncio.wait(
                    {send_task, receive_task},
                    return_when=asyncio.FIRST_EXCEPTION,
                )

                for task in pending:
                    task.cancel()

                for task in done:
                    if task.exception():
                        raise task.exception()

            except WebSocketDisconnect:
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

        @self.app.get("/api/v1/brain")
        async def get_brain_file() -> FileResponse:
            """Serve the nova_brain.pkl file from the local database."""
            if not BRAIN_FILE_PATH.exists():
                raise HTTPException(status_code=404, detail="Brain file not found")

            return FileResponse(
                path=BRAIN_FILE_PATH,
                media_type="application/octet-stream",
                filename="nova_brain.pkl"
            )

        @self.app.get("/api/v1/scene")
        async def get_latest_scene() -> FileResponse:
            """Serve the latest high-resolution scene capture from RAM disk."""
            if not LATEST_SCENE_FILE.exists():
                raise HTTPException(status_code=404, detail="Latest scene not available")

            return FileResponse(
                path=LATEST_SCENE_FILE,
                media_type="image/jpeg",
                filename="latest_scene.jpg"
            )

    def set_command_callback(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback to receive parsed JSON commands from WebSocket clients."""
        self.command_callback = callback

    async def _broadcast_event(self, event: Event) -> None:
        """Broadcast an event to all active WebSocket connections."""
        disconnected = []
        event_data = event.model_dump_json()

        for websocket in self.active_connections:
            try:
                await websocket.send_text(event_data)
            except Exception:
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            if ws in self.active_connections:
                self.active_connections.remove(ws)

    async def _send_loop(self, websocket: WebSocket) -> None:
        """Continuously send queued events to the connected client."""
        while True:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=15.0)
                await self._broadcast_event(event)
            except asyncio.TimeoutError:
                await websocket.send_text("ping")

    async def _receive_loop(self, websocket: WebSocket) -> None:
        """Continuously receive incoming JSON commands from the connected client."""
        while True:
            message = await websocket.receive_json()
            logger.info(f"Received WebSocket command: {message}")
            await self._handle_incoming_command(message)

    async def _handle_incoming_command(self, command: dict) -> None:
        """Process commands received from the mobile client."""
        logger.info(f"Processing incoming command: {command}")
        if self.command_callback and callable(self.command_callback):
            try:
                self.command_callback(command)
            except Exception as e:
                logger.error(f"Command callback execution failed: {e}")

    async def _run_server(self) -> None:
        """Run the FastAPI server with Uvicorn."""
        self.loop = asyncio.get_running_loop()
        try:
            # Start mDNS discovery
            await self.discovery.start_discovery()

            # Configure Uvicorn for Pi 4 optimization
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=5000,
                log_level="info",
                # Pi 4 optimizations
                workers=1,  # Single worker to avoid memory pressure
                loop="asyncio",
            )

            server = uvicorn.Server(config)

            # Run server until shutdown
            await server.serve()

        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            await self.discovery.stop_discovery()

    def start(self) -> None:
        """Start the server in a background thread."""
        if self._server_thread and self._server_thread.is_alive():
            logger.warning("Server is already running")
            return

        self._server_thread = threading.Thread(
            target=self._run_in_thread,
            name="LumosServer",
            daemon=True
        )
        self._server_thread.start()
        logger.info("Lumos server started in background thread")

    def _run_in_thread(self) -> None:
        """Run the async server in the thread's event loop."""
        try:
            asyncio.run(self._run_server())
        except Exception as e:
            logger.error(f"Thread error: {e}")

    def stop(self) -> None:
        """Stop the server and cleanup resources."""
        self._shutdown_event.set()
        logger.info("Lumos server shutdown initiated")

    def send_event(self, event: Event) -> None:
        """Send an event to be broadcast over WebSocket (thread-safe)."""
        if self.loop:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.event_queue.put(event),
                    self.loop
                )
            except RuntimeError:
                # If loop is closed or other error
                pass