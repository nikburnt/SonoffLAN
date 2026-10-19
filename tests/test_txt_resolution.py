"""Deliver frozen TXT independently of an already known device's DNS refresh."""

import asyncio

import pytest
from zeroconf import DNSAddress, DNSText, IPVersion
from zeroconf.asyncio import AsyncZeroconf
from zeroconf._protocol.incoming import DNSIncoming
from zeroconf._protocol.outgoing import DNSOutgoing
from zeroconf._utils.time import current_time_millis

from custom_components.sonoff.core.ewelink import XRegistry
from custom_components.sonoff.core.ewelink.local import encrypt
from custom_components.sonoff.event import XButtonEvent
from .test_txt_receive import (
    _CREATE_TASK, _GET_RUNNING_LOOP, _DEVICE, _NAME, _TYPE, packet,
)


@pytest.fixture(autouse=True)
def real_asyncio(monkeypatch):
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)


def partial_packet(sequence, outlet, now):
    """The office announces TXT/SRV but answers A/AAAA only later."""
    outgoing = DNSOutgoing(0x8400)
    for record in packet(sequence, outlet, now).answers():
        if not isinstance(record, DNSAddress):
            outgoing.add_answer_at_time(record, now)
    return DNSIncoming(outgoing.packets()[0], now=now)


async def receive(outlets, *, known=True, host="192.0.2.9:8081", resolve=True):
    azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
    zc = azc.zeroconf
    await zc.async_wait_for_start()
    registry = XRegistry(None)
    registry.config = {"devices": {}}
    entities = registry.setup_devices([
        {"deviceid": _DEVICE, "name": "Office address test",
         "extra": {"uiid": 161}, "params": {}, **({"host": host} if host else {})}
    ]) if known else []
    messages, hosts = [], []
    registry.local.dispatcher_connect("update", messages.append)
    registry.dispatcher_connect(_DEVICE, hosts.append)
    tasks = []
    now = current_time_millis()
    try:
        for index, outlet in enumerate(outlets, 43):
            zc.record_manager.async_updates_from_response(
                partial_packet(str(index), outlet, now + index - 43)
            )
            text = zc.cache.get_by_details(_NAME, 16, 1)
            assert isinstance(text, DNSText)
            tasks.append(asyncio.create_task(registry.local._handler2(
                zc, _TYPE, _NAME, text.text, (now + index - 43) / 1000
            )))
        await asyncio.sleep(.01)
        early = list(messages)
        if resolve:
            outgoing = DNSOutgoing(0x8400)
            for record in packet(str(42 + len(outlets)), outlets[-1], now + 10).answers():
                if isinstance(record, DNSAddress):
                    outgoing.add_answer_at_time(record, now + 10)
                    zc.record_manager.async_updates_from_response(
                DNSIncoming(outgoing.packets()[0], now=now + 10)
            )
            await asyncio.wait_for(asyncio.gather(*tasks), .5)
        else:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return {
            "early": early, "all": messages, "device": registry.devices.get(_DEVICE),
            "host_updates": [p["host"] for p in hosts if p and "host" in p],
            "button_states": [e.state for e in entities if isinstance(e, XButtonEvent)],
            "now": now,
        }
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await azc.async_close()


@pytest.mark.parametrize("outlets", [(0, 0), (0, 1), (1, 0, 1, 0)])
def test_known_host_does_not_delay_fast_repeat_or_independent_channels(outlets):
    result = asyncio.run(receive(outlets))
    expected = [(str(i), outlet) for i, outlet in enumerate(outlets, 43)]
    assert [(m["seq"], m["params"]["localKeyPass"]["outlet"]) for m in result["early"]] == expected, (
        "Already known device TXT must be delivered before DNS A/AAAA arrives"
    )
    assert result["all"] == result["early"], "Address refresh must not replay button/state payloads"
    assert [m["received_at"] for m in result["early"]] == [
        (result["now"] + i) / 1000 for i in range(len(outlets))
    ], "Frozen TXT receipt times must survive background address resolution"
    assert result["device"]["host"] == "192.0.2.1:8081", "A changed IP must still be learned"
    assert result["host_updates"] == ["192.0.2.1:8081"]
    assert result["button_states"] == [None, None], "Address refresh must not create phantom M5 actions"


@pytest.mark.parametrize("known,host", [(False, None), (True, None)])
def test_new_or_addressless_device_keeps_initial_discovery(known, host):
    result = asyncio.run(receive((0,), known=known, host=host))
    assert not result["early"], "A first discovery still needs a usable host"
    assert len(result["all"]) == 1
    assert result["all"][0]["host"] == "192.0.2.1:8081"


def test_known_host_payload_survives_missing_dns_answers():
    result = asyncio.run(receive((0, 1), resolve=False))
    assert [m["seq"] for m in result["early"]] == ["43", "44"], (
        "A lost address reply must not discard an already received state report"
    )
    assert result["device"]["host"] == "192.0.2.9:8081"
    assert not result["host_updates"]


@pytest.mark.parametrize("valid", [False, True])
def test_address_refresh_validates_ciphertext_without_replaying_state(valid):
    async def exercise():
        registry = XRegistry(None)
        registry.config = {"devices": {}}
        original = [{"outlet": 0, "switch": "on"}, {"outlet": 1, "switch": "on"}]
        device = {
            "deviceid": _DEVICE, "name": "Office metadata test",
            "extra": {"uiid": 161}, "devicekey": "synthetic-test-key",
            "host": "192.0.2.9:8081", "params": {"switches": original},
        }
        entities = registry.setup_devices([device])
        updates = []
        registry.dispatcher_connect(_DEVICE, updates.append)
        msg = encrypt({"data": {
            "switches": [{"outlet": 0, "switch": "off"}],
            "localKeyPass": {"outlet": 1, "key": 0},
        }}, "synthetic-test-key" if valid else "incorrect-test-key")
        msg.update(deviceid=_DEVICE, host="192.0.2.1:8081", localtype="plug", seq="45")
        registry.local_update_host(msg)
        assert device["host"] == ("192.0.2.1:8081" if valid else "192.0.2.9:8081")
        assert device["devicekey"] == "synthetic-test-key", "Metadata must retain the encryption key"
        assert device["params"]["switches"] == original, "Address refresh must not dispatch relay state"
        assert updates == ([{"host": "192.0.2.1:8081"}] if valid else [])
        assert all(e.state is None for e in entities if isinstance(e, XButtonEvent))
    asyncio.run(exercise())


def test_address_refresh_does_not_recreate_a_removed_device():
    registry = XRegistry(None)
    registry.local_update_host({"deviceid": _DEVICE, "host": "192.0.2.1:8081", "localtype": "plug"})
    assert not registry.devices, "Late discovery must not revive an unloaded device"
