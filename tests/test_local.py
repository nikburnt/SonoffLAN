"""Exercise real LAN HTTP replies and cloud fallback without contacting devices."""

import asyncio
import json
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import aiohttp_client

from custom_components.sonoff.core.ewelink import XRegistry

_CREATE_TASK = asyncio.create_task
_GET_RUNNING_LOOP = asyncio.get_running_loop
_TIMEOUT = 1.0
_PAUSE = 1.5


@pytest.fixture(autouse=True)
def real_asyncio(monkeypatch):
    # Existing synchronous entity fixtures replace these module globals.
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(payload)
        self.server.connections.add(self.client_address)
        self.close_connection = False
        mode = self.server.mode
        body = json.dumps(
            {"error": 1 if mode == "error" else 0, "seq": payload["sequence"]}
        ).encode()
        if mode == "malformed":
            body = b"not json"
        elif mode in ("html", "html_charset"):
            body = b"<html>Not Found</html>"
        try:
            time.sleep(1.5 if mode == "late_headers" else 0.08)
            self.send_response(404 if mode in ("html", "html_charset") else 200)
            self.send_header(
                "Content-Type",
                {"html": "text/html", "html_charset": "text/html; charset=utf-8"}.get(
                    mode, "application/json"
                ),
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if mode == "late_body":
                time.sleep(1.5)
            if mode == "trickle_body":
                for offset in range(0, len(body), 5):
                    self.wfile.write(body[offset : offset + 5])
                    self.wfile.flush()
                    time.sleep(0.3)
            else:
                self.wfile.write(body)
                self.wfile.flush()
            self.server.completed.append(time.monotonic())
        except (BrokenPipeError, ConnectionResetError):
            pass


@contextmanager
def _server(mode="normal"):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    server.daemon_threads = True
    server.mode = mode
    server.requests = []
    server.connections = set()
    server.completed = []
    thread = threading.Thread(
        target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
    )
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def _exercise(
    server, *, pause=False, warmup=False, commands=None, timeout=_TIMEOUT, interval=0
):
    hass = HomeAssistant("/tmp/sonoff-local-timeout-test")
    resolver = aiohttp.ThreadedResolver()
    hass.data[aiohttp_client.DATA_RESOLVER] = resolver
    trace = aiohttp.TraceConfig()
    pause_next_request = False

    async def headers_sent(_session, _context, _params):
        nonlocal pause_next_request
        if pause_next_request:
            pause_next_request = False
            asyncio.get_running_loop().call_later(0.03, time.sleep, _PAUSE)

    trace.on_request_headers_sent.append(headers_sent)
    session = aiohttp_client.async_create_clientsession(
        hass, auto_cleanup=False, trace_configs=[trace]
    )
    connector = session.connector
    registry = XRegistry(session)
    registry.local.online = True
    registry.cloud.online = True
    fallbacks = []

    async def cloud_send(device, params=None, sequence=None, **_kwargs):
        fallbacks.append({"device": device["deviceid"], "params": params})
        return "online"

    registry.cloud.send = cloud_send
    device = {
        "deviceid": "1000123abc",
        "host": f"127.0.0.1:{server.server_port}",
        "local": True,
        "online": True,
    }
    params = commands or [{"switches": [{"outlet": 0, "switch": "on"}]}]

    async def send(index, command):
        if interval:
            await asyncio.sleep(index * interval)
        await registry.send(device, command, query_cloud=False, timeout_lan=timeout)

    try:
        if warmup:
            await registry.send(device, params[0], query_cloud=False)
            server.requests.clear()
            server.completed.clear()
        pause_next_request = pause
        began = time.monotonic()
        await asyncio.gather(*(send(index, p) for index, p in enumerate(params)))
        elapsed = time.monotonic() - began
        pending = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        return {
            "fallbacks": fallbacks,
            "requests": list(server.requests),
            "connections": len(server.connections),
            "completed": [t - began for t in server.completed],
            "elapsed": elapsed,
            "pending": pending,
        }
    finally:
        session.detach()
        await connector.close()
        await resolver.close()


@pytest.mark.parametrize("warmup", [False, True], ids=["fresh", "reused"])
def test_buffered_success_after_loop_pause_does_not_fall_back(warmup):
    with _server() as server:
        result = asyncio.run(_exercise(server, pause=True, warmup=warmup))
    assert result["completed"][0] < _TIMEOUT, "Device replied before the deadline"
    assert not result["fallbacks"], "A buffered successful ACK must win over timeout"
    assert not result["pending"], "The LAN request must be finished"
    assert result["connections"] == 1, "The warmup must reuse the same TCP connection"


@pytest.mark.parametrize("mode", ["late_headers", "late_body", "trickle_body"])
def test_unfinished_response_keeps_total_deadline_and_falls_back(mode):
    with _server(mode) as server:
        result = asyncio.run(_exercise(server))
    assert len(result["fallbacks"]) == 1, "An unfinished request needs cloud fallback"
    assert result["elapsed"] < 1.3, "Partial traffic must not extend the total deadline"
    assert not result["pending"], "Timed out LAN work must not continue after fallback"


@pytest.mark.parametrize("outlets", [(0, 0), (0, 1)], ids=["repeat", "two-buttons"])
def test_concurrent_commands_remain_independent_during_loop_pause(outlets):
    commands = [
        {"switches": [{"outlet": outlets[0], "switch": "on"}]},
        {"switches": [{"outlet": outlets[1], "switch": "off"}]},
    ]
    with _server() as server:
        result = asyncio.run(
            _exercise(
                server,
                pause=True,
                commands=commands,
                interval=0.01 if outlets[0] == outlets[1] else 0,
            )
        )
    assert len(result["requests"]) == 2, "Both commands must reach LAN"
    assert sorted(p["data"]["switches"][0]["switch"] for p in result["requests"]) == [
        "off",
        "on",
    ], "Neither command may be coalesced or dropped"
    assert not result["fallbacks"], (
        "Ready ACKs must not produce duplicate cloud commands"
    )


@pytest.mark.parametrize("timeout", [0, None])
def test_disabled_timeout_still_accepts_a_response(timeout):
    with _server() as server:
        result = asyncio.run(_exercise(server, timeout=timeout))
    assert not result["fallbacks"], "A disabled timeout must not cancel the request"


@pytest.mark.parametrize(
    "mode,commands,fallbacks",
    [
        ("normal", None, 0),
        ("error", None, 1),
        ("malformed", None, 1),
        ("html", [None], 0),
        ("html", None, 1),
        ("html_charset", [None], 0),
        ("html_charset", None, 1),
        ("malformed", [None], 0),
    ],
    ids=[
        "normal", "device-error", "invalid-json", "getState-404", "switches-404",
        "getState-html-charset", "switches-html-charset", "getState-non-json",
    ],
)
def test_existing_response_classification_is_preserved(mode, commands, fallbacks):
    with _server(mode) as server:
        result = asyncio.run(_exercise(server, commands=commands))
    assert len(result["fallbacks"]) == fallbacks, "LAN response classification changed"
