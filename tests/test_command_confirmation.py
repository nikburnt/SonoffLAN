"""A relay proof requires its own successful transport and versioned report."""

import asyncio
import time
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from homeassistant.core import Context

from custom_components.sonoff.core import ewelink as registry_module
from custom_components.sonoff.core.ewelink import XRegistry
from custom_components.sonoff.switch import XSwitches

CREATE_TASK = asyncio.create_task
GET_LOOP = asyncio.get_running_loop
ATTRIBUTE = "sonoff_command_confirmation"


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", GET_LOOP)
    registry = XRegistry(None)
    registry.config = {"devices": {}}
    registry.local.online = True
    registry.cloud.online = False
    device = {
        "deviceid": "1000123abc",
        "name": "Office",
        "extra": {"uiid": 161},
        "online": True,
        "local": True,
        "params": {"switches": [{"outlet": i, "switch": "off"} for i in range(2)]},
    }
    relays = sorted(
        (e for e in registry.setup_devices([device]) if isinstance(e, XSwitches)),
        key=lambda e: e.channel,
    )
    for relay in relays:
        relay.async_set_context(Context())
    calls = []

    async def transport(device, params, command, sequence, *args):
        calls.append((sequence, deepcopy(params)))
        return "online"

    monkeypatch.setattr(registry.local, "send", transport)

    def report(sequence, values):
        registry.cloud_update(
            {
                "deviceid": device["deviceid"],
                "sequence": sequence,
                "params": {"switches": values},
            }
        )

    return registry, relays, calls, report


def proof(relay):
    return (relay.extra_state_attributes or {}).get(ATTRIBUTE)


@pytest.mark.parametrize("report_before_ack", [False, True])
def test_only_an_exact_report_and_ack_publish_a_context_bound_proof(
    setup, report_before_ack
):
    asyncio.run(exact_report_and_ack(setup, report_before_ack))


async def exact_report_and_ack(setup, before):
    registry, relays, calls, report = setup
    relay = relays[0]
    started, ack = asyncio.Event(), asyncio.Event()

    async def transport(device, params, command, sequence, *args):
        calls.append((sequence, deepcopy(params)))
        started.set()
        await ack.wait()
        return "online"

    registry.local.send = transport
    task = asyncio.create_task(relay.async_turn_on())
    try:
        await asyncio.wait_for(started.wait(), 0.5)
        sequence, params = calls[0]
        if before:
            report(sequence, params["switches"])
            assert proof(relay) is None, "A state report cannot replace the pending ACK"
        ack.set()
        await task
        if not before:
            assert proof(relay) is None, "ACK alone has no reported relay state"
            report(None, params["switches"])
            assert proof(relay) is None, "An unversioned match is not causal proof"
            report(sequence, params["switches"])
        receipt = proof(relay)
        assert receipt is not None
        assert receipt["context_id"] == relay._context.id
        assert receipt["device_id"] == "1000123abc"
        assert receipt["channel"] == 0 and receipt["state"] == "on"
        assert receipt["sequence"] == sequence
        assert isinstance(receipt["command_id"], str) and receipt["command_id"]
        assert proof(relays[1]) is None
    finally:
        ack.set()
        await asyncio.gather(task, return_exceptions=True)
        registry.local.online = False
        await registry.stop()


@pytest.mark.parametrize("delay", [0.1, 20, 600])
def test_a_delayed_old_report_cannot_restore_a_new_invocations_proof(setup, delay):
    asyncio.run(delayed_report(setup, delay))


async def delayed_report(setup, delay):
    registry, relays, calls, report = setup
    relay = relays[0]
    try:
        await relay.async_turn_on()
        report(*((calls[0][0], calls[0][1]["switches"])))
        previous = proof(relay)
        assert previous is not None
        # Reusing the HA context must still create an independent invocation.
        next_call = asyncio.create_task(relay.async_turn_off())
        await asyncio.sleep(0)
        assert proof(relay) is None
        report(calls[0][0], calls[0][1]["switches"])
        assert proof(relay) is None
        await next_call
        report(calls[1][0], calls[1][1]["switches"])
        current = proof(relay)
        assert current["command_id"] != previous["command_id"]
        assert current["state"] == "off"
        # Packet age is irrelevant: even a first-seen old report stays old.
        received = time.monotonic() + delay
        with patch.object(
            registry_module, "time", SimpleNamespace(monotonic=lambda: received)
        ):
            report(calls[0][0], calls[0][1]["switches"])
        assert proof(relay) == current and relay.is_on is False
    finally:
        registry.local.online = False
        await registry.stop()


def test_simultaneous_channels_share_sequence_but_not_invocation_or_context(setup):
    asyncio.run(simultaneous(setup))


async def simultaneous(setup):
    registry, relays, calls, report = setup
    try:
        await asyncio.gather(*(relay.async_turn_on() for relay in relays))
        assert len(calls) == 1
        sequence, params = calls[0]
        report(sequence, [params["switches"][0]])
        assert proof(relays[0]) is not None and proof(relays[1]) is None
        report(sequence, [params["switches"][1]])
        receipts = [proof(relay) for relay in relays]
        assert [r["channel"] for r in receipts] == [0, 1]
        assert [r["context_id"] for r in receipts] == [r._context.id for r in relays]
        assert receipts[0]["command_id"] != receipts[1]["command_id"]
        assert [r["sequence"] for r in receipts] == [sequence, sequence]
    finally:
        registry.local.online = False
        await registry.stop()


def test_a_failed_transport_never_issues_a_fast_confirmation(setup):
    asyncio.run(failed_transport(setup))


async def failed_transport(setup):
    registry, relays, calls, report = setup

    async def transport(device, params, command, sequence, *args):
        calls.append((sequence, deepcopy(params)))
        report(sequence, params["switches"])
        return "error"

    registry.local.send = transport
    try:
        await relays[0].async_turn_on()
        assert proof(relays[0]) is None
    finally:
        registry.local.online = False
        await registry.stop()


def test_first_old_proof_is_rejected_while_a_new_invocation_is_still_batching(setup):
    asyncio.run(first_old_proof(setup))


async def first_old_proof(setup):
    registry, relays, calls, report = setup
    relay = relays[0]
    try:
        await relay.async_turn_on()
        assert proof(relay) is None
        next_call = asyncio.create_task(relay.async_turn_off())
        await asyncio.sleep(0)
        report(calls[0][0], calls[0][1]["switches"])
        assert proof(relay) is None, (
            "The first old proof arrived after a newer invocation began"
        )
        await next_call
        report(calls[1][0], calls[1][1]["switches"])
        assert proof(relay)["state"] == "off"
    finally:
        registry.local.online = False
        await registry.stop()


def test_a_direct_versioned_lan_state_also_confirms_only_its_channel(setup):
    asyncio.run(direct_lan_proof(setup))


async def direct_lan_proof(setup):
    registry, relays, calls, _report = setup
    try:
        await relays[1].async_turn_on()
        sequence, params = calls[0]
        registry.local_update(
            {
                "deviceid": "1000123abc",
                "host": "127.0.0.1:8081",
                "localtype": "plug",
                "request_sequence": sequence,
                "seq": "device-counter",
                "params": params,
            }
        )
        assert proof(relays[0]) is None
        assert proof(relays[1])["sequence"] == sequence
        assert proof(relays[1])["channel"] == 1
    finally:
        registry.local.online = False
        await registry.stop()


@pytest.mark.parametrize("cloud", [False, True])
def test_a_direct_registry_write_invalidates_the_previous_entity_proof(setup, cloud):
    asyncio.run(direct_write_invalidates_proof(setup, cloud))


async def direct_write_invalidates_proof(setup, cloud):
    registry, relays, calls, report = setup
    relay = relays[0]
    try:
        await relay.async_turn_on()
        report(calls[0][0], calls[0][1]["switches"])
        assert proof(relay) is not None
        if cloud:
            registry.cloud.online = True

            async def send(*args, **kwargs):
                return "online"

            registry.cloud.send = send
        method = registry.send_cloud if cloud else registry.send
        await method(
            registry.devices["1000123abc"],
            {"switches": [{"outlet": 0, "switch": "off"}]},
        )
        assert proof(relay) is None, (
            "A direct service write superseded the native invocation"
        )
    finally:
        registry.local.online = False
        await registry.stop()
