"""M5 batching stays brief without losing shared ACKs or late sibling keys."""

import asyncio
from copy import deepcopy

import pytest

from custom_components.sonoff.core.ewelink import XRegistry

_CREATE_TASK = asyncio.create_task
_GET_RUNNING_LOOP = asyncio.get_running_loop
_SLEEP = asyncio.sleep


@pytest.fixture(autouse=True)
def real_asyncio(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)


class Clock:
    """Advance batching timers explicitly while actual tasks keep running."""

    def __init__(self):
        self.now = 0.0
        self.waiters = []
        self.scheduled = asyncio.Queue()

    async def sleep(self, delay):
        if delay == 0:
            return await _SLEEP(0)
        future = asyncio.get_running_loop().create_future()
        self.waiters.append((self.now + delay, future))
        self.scheduled.put_nowait(None)
        await future

    def advance(self, seconds):
        self.now += seconds
        for deadline, future in self.waiters:
            if deadline <= self.now and not future.done():
                future.set_result(None)


def state(outlet, value):
    return {"switches": [{"outlet": outlet, "switch": value}]}


@pytest.mark.parametrize("uiid", [None, 1])
def test_other_devices_keep_their_existing_batch_window(monkeypatch, uiid):
    asyncio.run(_other_devices_keep_their_existing_batch_window(monkeypatch, uiid))


async def _other_devices_keep_their_existing_batch_window(monkeypatch, uiid):
    clock = Clock()
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc", "extra": {"uiid": uiid}}
    started = asyncio.Event()

    async def send(device, params):
        started.set()

    registry.send = send
    task = asyncio.create_task(registry.send_bulk(device, state(0, "on")))
    try:
        await asyncio.wait_for(clock.scheduled.get(), 0.5)
        clock.advance(0.040)
        await _SLEEP(0)
        assert not started.is_set(), "The fast M5 path must not affect other devices"
        clock.advance(0.060)
        await asyncio.wait_for(started.wait(), 0.5)
        await task
    finally:
        await registry.stop()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("uiid,channels", [(160, 1), (161, 2), (162, 3)])
def test_m5_flushes_quickly_and_every_channel_still_waits_for_the_shared_ack(
    monkeypatch, uiid, channels
):
    asyncio.run(_m5_flushes_quickly(monkeypatch, uiid, channels))


async def _m5_flushes_quickly(monkeypatch, uiid, channels):
    clock = Clock()
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc", "extra": {"uiid": uiid}}
    started = asyncio.Event()
    ack = asyncio.Event()
    calls = []

    async def send(device, params):
        calls.append(deepcopy(params))
        started.set()
        await ack.wait()
        return "accepted"

    registry.send = send
    tasks = [asyncio.create_task(registry.send_bulk(device, state(0, "on")))]
    try:
        await asyncio.wait_for(clock.scheduled.get(), 0.5)
        for outlet in range(1, channels):
            clock.advance(0.010)
            tasks.append(
                asyncio.create_task(registry.send_bulk(device, state(outlet, "on")))
            )
            await _SLEEP(0)
        clock.advance(0.040 - clock.now)
        await asyncio.wait_for(started.wait(), 0.5)
        assert calls == [
            {"switches": [{"outlet": i, "switch": "on"} for i in range(channels)]}
        ], "Closely spaced sibling keys must share one batch"
        assert all(not task.done() for task in tasks), "No key may finish before ACK"
        ack.set()
        assert await asyncio.gather(*tasks) == ["accepted"] * channels
    finally:
        ack.set()
        await registry.stop()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_late_sibling_keeps_its_repeat_order_while_the_first_key_has_a_delayed_ack(
    monkeypatch,
):
    asyncio.run(_late_sibling_keeps_its_repeat_order(monkeypatch))


async def _late_sibling_keeps_its_repeat_order(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc", "extra": {"uiid": 161}}
    started = asyncio.Queue()
    acks = [asyncio.Event() for _ in range(3)]
    calls = []

    async def send(device, params):
        index = len(calls)
        calls.append(deepcopy(params))
        started.put_nowait(index)
        await acks[index].wait()

    registry.send = send

    async def sibling():
        await registry.send_bulk(device, state(1, "on"))
        await registry.send_bulk(device, state(1, "off"))

    tasks = [asyncio.create_task(registry.send_bulk(device, state(0, "on")))]
    try:
        await asyncio.wait_for(clock.scheduled.get(), 0.5)
        clock.advance(0.040)
        assert await asyncio.wait_for(started.get(), 0.5) == 0
        clock.advance(0.030)
        tasks.append(asyncio.create_task(sibling()))
        await asyncio.wait_for(clock.scheduled.get(), 0.5)
        clock.advance(0.040)
        assert await asyncio.wait_for(started.get(), 0.5) == 1
        clock.advance(10)
        await _SLEEP(0)
        assert calls == [state(0, "on"), state(1, "on")], (
            "The sibling OFF must wait for its own ON ACK, regardless of delay"
        )
        acks[1].set()
        await asyncio.wait_for(clock.scheduled.get(), 0.5)
        clock.advance(0.040)
        assert await asyncio.wait_for(started.get(), 0.5) == 2
        assert calls == [state(0, "on"), state(1, "on"), state(1, "off")]
        assert not tasks[0].done(), "Another channel's ACK cannot finish the first key"
        acks[0].set()
        acks[2].set()
        await asyncio.gather(*tasks)
    finally:
        for ack in acks:
            ack.set()
        await registry.stop()
        await asyncio.gather(*tasks, return_exceptions=True)
