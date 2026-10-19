"""Deliver known-host TXT before queued work and expose honest receive timing."""

import asyncio
import json
import logging
import time

import pytest
from zeroconf import DNSText, IPVersion, ServiceStateChange
from zeroconf.asyncio import AsyncZeroconf
from zeroconf._protocol.incoming import DNSIncoming
from zeroconf._protocol.outgoing import DNSOutgoing
from zeroconf._utils.time import current_time_millis

from custom_components.sonoff.core.ewelink.local import XRegistryLocal
from .test_txt_receive import (
    _CREATE_TASK, _GET_RUNNING_LOOP, _DEVICE, _NAME, _TYPE, packet,
)
from .test_txt_resolution import partial_packet

LOGGER = "custom_components.sonoff.core.ewelink.local"


@pytest.fixture(autouse=True)
def real_asyncio(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)


@pytest.mark.parametrize("outlets", [(0, 0), (0, 1), (1, 0, 1, 0)])
def test_known_txt_precedes_already_queued_work_without_losing_burst(outlets):
    async def exercise():
        azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
        zc = azc.zeroconf
        await zc.async_wait_for_start()
        receiver = XRegistryLocal(None, {_DEVICE: {"host": "192.0.2.9:8081"}})
        order = []
        receiver.dispatcher_connect("update", lambda m: order.append(
            (m["seq"], m["params"]["localKeyPass"]["outlet"], m["received_at"])
        ))
        try:
            receiver.start(zc)
            now = current_time_millis()
            asyncio.get_running_loop().call_soon(order.append, "queued_callback")
            for index, outlet in enumerate(outlets):
                zc.record_manager.async_updates_from_response(
                    partial_packet(str(43 + index), outlet, now + index)
                )
            expected = [(str(43 + i), outlet, (now + i) / 1000)
                        for i, outlet in enumerate(outlets)]
            assert order == expected, (
                "An already received known-host TXT waited for queued Core work"
            )
            await asyncio.sleep(0)
            assert order == [*expected, "queued_callback"], (
                "Starting the receiver immediately must not drop or reorder reports"
            )
        finally:
            await receiver.stop()
            await azc.async_close()
    asyncio.run(exercise())


@pytest.mark.parametrize("host", [None, ""])
def test_initial_discovery_keeps_queued_setup_until_host_is_known(host):
    async def exercise():
        azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
        zc = azc.zeroconf
        await zc.async_wait_for_start()
        receiver = XRegistryLocal(None, {} if host is None else {_DEVICE: {"host": host}})
        order = []
        receiver.dispatcher_connect("update", lambda m: order.append(m["host"]))
        try:
            receiver.start(zc)
            asyncio.get_running_loop().call_soon(order.append, "queued_callback")
            zc.record_manager.async_updates_from_response(
                packet("43", 0, current_time_millis())
            )
            assert not order, "Initial device setup must not become a nested browser callback"
            await asyncio.sleep(.01)
            assert order == ["queued_callback", "192.0.2.1:8081"], (
                "Addressless discovery must still resolve and deliver the first state"
            )
        finally:
            await receiver.stop()
            await azc.async_close()
    asyncio.run(exercise())


def test_asyncio_without_eager_support_keeps_normal_delivery(monkeypatch):
    monkeypatch.delattr(asyncio, "eager_task_factory", raising=False)

    async def exercise():
        azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
        zc = azc.zeroconf
        await zc.async_wait_for_start()
        receiver = XRegistryLocal(None, {_DEVICE: {"host": "192.0.2.9:8081"}})
        order = []
        receiver.dispatcher_connect("update", lambda m: order.append(m["seq"]))
        try:
            receiver.start(zc)
            asyncio.get_running_loop().call_soon(order.append, "queued_callback")
            zc.record_manager.async_updates_from_response(
                partial_packet("43", 0, current_time_millis())
            )
            assert not order, "An older asyncio needs its supported task-creation path"
            await asyncio.sleep(.01)
            assert order == ["queued_callback", "43"], (
                "Python compatibility must not drop the known device's state report"
            )
        finally:
            await receiver.stop()
            await azc.async_close()
    asyncio.run(exercise())


def test_timing_keeps_original_receipt_and_distinguishes_processing(caplog):
    async def exercise():
        azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
        zc = azc.zeroconf
        await zc.async_wait_for_start()
        receiver = XRegistryLocal(None, {_DEVICE: {"host": "192.0.2.9:8081"}})
        messages = []
        receiver.dispatcher_connect("update", messages.append)
        try:
            receiver.start(zc)
            now = current_time_millis()
            zc.record_manager.async_updates_from_response(partial_packet("43", 0, now))
            await asyncio.sleep(.01)
            # Another DNS notification may refer to the same cached TXT.
            receiver._handler1(zc, _TYPE, _NAME, ServiceStateChange.Updated)
            await asyncio.sleep(.01)
            assert [m["received_at"] for m in messages] == [now / 1000] * 2, (
                "Timing instrumentation must not refresh the report's receipt time"
            )
            traces = [r.args[1] for r in caplog.records
                      if r.name == LOGGER and "LocalTiming" in r.getMessage()]
            state_traces = [t for t in traces if t["signal"] == "update"]
            assert len(state_traces) == 2, "Both data dispatches need timing markers"
            for trace in state_traces:
                assert trace["seq"] == "43"
                assert trace["txt_received_monotonic"] == now / 1000
                assert (trace["txt_received_monotonic"] <= trace["notified_monotonic"]
                        <= trace["handler_started_monotonic"]
                        <= trace["dispatch_started_monotonic"]
                        <= trace["dispatch_finished_monotonic"])
                assert abs(trace["finished_epoch"] - time.time()) < 1
                assert not {"params", "data", "iv", "devicekey"} & trace.keys(), (
                    "Receive timing must not export payloads or credentials"
                )
                json.dumps(trace)
            assert state_traces[1]["notified_monotonic"] > state_traces[0]["notified_monotonic"]
        finally:
            await receiver.stop()
            await azc.async_close()
    with caplog.at_level(logging.DEBUG, logger=LOGGER):
        asyncio.run(exercise())


def test_eager_bad_txt_does_not_escape_browser_or_block_next_report():
    async def exercise():
        azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
        zc = azc.zeroconf
        await zc.async_wait_for_start()
        receiver = XRegistryLocal(None, {_DEVICE: {"host": "192.0.2.9:8081"}})
        messages = []
        receiver.dispatcher_connect("update", messages.append)
        try:
            receiver.start(zc)
            now = current_time_millis()
            invalid = DNSOutgoing(0x8400)
            invalid.add_answer_at_time(DNSText(_NAME, 16, 0x8001, 120, b"\x03bad"), now)
            zc.record_manager.async_updates_from_response(
                DNSIncoming(invalid.packets()[0], now=now)
            )
            zc.record_manager.async_updates_from_response(partial_packet("44", 1, now + 1))
            assert [(m["seq"], m["params"]["localKeyPass"]["outlet"]) for m in messages] == [
                ("44", 1)
            ], "A bad TXT must not interrupt delivery of the next valid channel"
        finally:
            await receiver.stop()
            await azc.async_close()
    asyncio.run(exercise())
