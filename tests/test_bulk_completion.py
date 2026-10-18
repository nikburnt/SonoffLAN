"""Every channel in one batch must await the actual shared send result."""

import asyncio
from copy import deepcopy

import pytest

from custom_components.sonoff.core.ewelink import XRegistry

_CREATE_TASK = asyncio.create_task
_GET_RUNNING_LOOP = asyncio.get_running_loop


@pytest.fixture(autouse=True)
def real_asyncio(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)


def state(outlet, value):
    return {"switches": [{"outlet": outlet, "switch": value}]}


def test_two_channels_wait_for_the_same_transport_result():
    asyncio.run(_two_channels_wait_for_the_same_transport_result())


async def _two_channels_wait_for_the_same_transport_result():
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc"}
    started = asyncio.Event()
    ack = asyncio.Event()
    calls = []

    async def send(device, params):
        calls.append(deepcopy(params))
        started.set()
        await ack.wait()
        return "accepted"

    registry.send = send
    tasks = [
        asyncio.create_task(registry.send_bulk(device, state(i, "on"))) for i in (0, 1)
    ]
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        await asyncio.sleep(0)
        assert all(not task.done() for task in tasks), (
            "A channel returned before the shared ACK"
        )
        assert calls == [
            {"switches": [{"outlet": 0, "switch": "on"}, {"outlet": 1, "switch": "on"}]}
        ]
        ack.set()
        assert await asyncio.gather(*tasks) == ["accepted", "accepted"]
    finally:
        ack.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("cancelled_caller", [0, 1])
@pytest.mark.parametrize("uiid", [None, 161])
def test_one_cancelled_caller_does_not_cancel_the_other_channel(cancelled_caller, uiid):
    asyncio.run(
        _one_cancelled_caller_does_not_cancel_the_other_channel(cancelled_caller, uiid)
    )


async def _one_cancelled_caller_does_not_cancel_the_other_channel(
    cancelled_caller, uiid
):
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc", "extra": {"uiid": uiid}}
    started = asyncio.Event()
    ack = asyncio.Event()
    transport_cancelled = []

    async def send(device, params):
        started.set()
        try:
            await ack.wait()
        except asyncio.CancelledError:
            transport_cancelled.append(True)
            raise
        return "accepted"

    registry.send = send
    tasks = [
        asyncio.create_task(registry.send_bulk(device, state(i, "on"))) for i in (0, 1)
    ]
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        await asyncio.sleep(0)
        tasks[cancelled_caller].cancel()
        result = await asyncio.gather(tasks[cancelled_caller], return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
        assert not tasks[1 - cancelled_caller].done() and not transport_cancelled
        ack.set()
        assert await tasks[1 - cancelled_caller] == "accepted"
    finally:
        ack.set()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.parametrize("uiid", [None, 161])
def test_a_second_press_cannot_overtake_the_coalesced_on_command(uiid):
    asyncio.run(_a_second_press_cannot_overtake_the_coalesced_on_command(uiid))


async def _a_second_press_cannot_overtake_the_coalesced_on_command(uiid):
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc", "extra": {"uiid": uiid}}
    started = asyncio.Event()
    ack = asyncio.Event()
    calls = []

    async def send(device, params):
        calls.append(deepcopy(params))
        started.set()
        await ack.wait()

    registry.send = send

    async def other_key():
        await registry.send_bulk(device, state(1, "on"))
        await registry.send_bulk(device, state(1, "off"))

    tasks = [
        asyncio.create_task(registry.send_bulk(device, state(0, "on"))),
        asyncio.create_task(other_key()),
    ]
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        await asyncio.sleep(0.12)
        assert len(calls) == 1, "OFF was sent before the shared ON transport completed"
        ack.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 0.5)
        assert calls[-1] == state(1, "off")
    finally:
        ack.set()
        await asyncio.gather(*tasks, return_exceptions=True)


def test_a_batch_failure_is_delivered_to_both_channels():
    asyncio.run(_a_batch_failure_is_delivered_to_both_channels())


async def _a_batch_failure_is_delivered_to_both_channels():
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc"}

    async def send(device, params):
        raise RuntimeError("transport failed")

    registry.send = send
    results = await asyncio.gather(
        *(registry.send_bulk(device, state(i, "on")) for i in (0, 1)),
        return_exceptions=True,
    )
    assert all(isinstance(result, RuntimeError) for result in results)


def test_registry_stop_cancels_and_collects_its_batch_transport():
    asyncio.run(_registry_stop_cancels_and_collects_its_batch_transport())


async def _registry_stop_cancels_and_collects_its_batch_transport():
    registry = XRegistry(None)
    device = {"deviceid": "1000123abc"}
    started = asyncio.Event()
    cancelled = asyncio.Event()
    finish = asyncio.Event()

    async def send(device, params):
        started.set()
        try:
            await finish.wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise

    registry.send = send
    task = asyncio.create_task(registry.send_bulk(device, state(0, "on")))
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        await registry.stop()
        assert cancelled.is_set(), "Registry unload left a relay send running"
        result = await asyncio.gather(task, return_exceptions=True)
        assert isinstance(result[0], asyncio.CancelledError)
    finally:
        finish.set()
        await asyncio.gather(task, return_exceptions=True)
