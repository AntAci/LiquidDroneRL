"""
WebSocket bridge server for streaming 3D demo state to browser renderers.

Run:
  python demo/bridge.py

Then open demo/renderer_3d.html in a browser. The run_3d_demo.py script will
connect as a client and push JSON packets, which are broadcast to all connected
browser clients.
"""

from __future__ import annotations

import asyncio
import json
import signal
from typing import Set

import websockets
from websockets.server import WebSocketServerProtocol


CLIENTS: Set[WebSocketServerProtocol] = set()


async def broadcast(message: str) -> None:
    """Broadcast a message string to all connected clients."""
    if not CLIENTS:
        return
    # Gather with return_exceptions=True to avoid cancellation on one failure
    await asyncio.gather(
        *(safe_send(ws, message) for ws in list(CLIENTS)),
        return_exceptions=True
    )


async def safe_send(ws: WebSocketServerProtocol, message: str) -> None:
    """Send to one client, removing it if the connection is broken."""
    try:
        await ws.send(message)
    except Exception:
        # Drop broken client
        try:
            CLIENTS.remove(ws)
        except KeyError:
            pass


async def handler(websocket: WebSocketServerProtocol) -> None:
    """Accept packets and broadcast them to all connected clients."""
    CLIENTS.add(websocket)
    print(f"[bridge] Client connected. Total clients: {len(CLIENTS)}")
    try:
        async for message in websocket:
            # Optional: validate JSON
            try:
                _ = json.loads(message)
            except json.JSONDecodeError:
                # Ignore malformed, but do not drop connection
                print(f"[bridge] Invalid JSON received, ignoring")
                continue
            print(f"[bridge] Broadcasting message to {len(CLIENTS)} client(s)")
            await broadcast(message)
    finally:
        CLIENTS.discard(websocket)
        print(f"[bridge] Client disconnected. Remaining clients: {len(CLIENTS)}")


async def main() -> None:
    """Run the WebSocket server on ws://localhost:8765."""
    stop = asyncio.Future()

    # Allow Ctrl+C to stop the server gracefully
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set_result, None)

    print("[bridge] Starting WebSocket server on ws://localhost:8765")
    async with websockets.serve(handler, "127.0.0.1", 8765, ping_interval=20, ping_timeout=20):
        print("[bridge] Ready. Waiting for clients...")
        await stop
    print("[bridge] Server stopped.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass


