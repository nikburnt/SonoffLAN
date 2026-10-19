"""Keep individual TXT updates when the async resolver runs after a burst."""

import asyncio
import ipaddress
import json

import pytest
from zeroconf import DNSAddress, DNSPointer, DNSService, DNSText, IPVersion
from zeroconf.asyncio import AsyncZeroconf
from zeroconf._protocol.incoming import DNSIncoming
from zeroconf._protocol.outgoing import DNSOutgoing
from zeroconf._utils.time import current_time_millis

from custom_components.sonoff.core.ewelink import XRegistry
from custom_components.sonoff.core.ewelink.local import XRegistryLocal
from custom_components.sonoff.event import XButtonEvent

_CREATE_TASK = asyncio.create_task
_GET_RUNNING_LOOP = asyncio.get_running_loop
_TYPE = "_ewelink._tcp.local."
_DEVICE = "1000123abc"
_NAME = f"eWeLink_{_DEVICE}.{_TYPE}"
_HOST = "office-receive-test.local."


def packet(sequence, outlet, now, *, metadata=True):
    params = {
        "localKeyPass": {"outlet": outlet, "key": 0},
        "switches": [{"outlet": i, "switch": "off"} for i in (0, 1)],
        "triggerType": 11,
    }
    properties = {
        "id": _DEVICE,
        "type": "plug",
        "seq": sequence,
        "data1": json.dumps(params, separators=(",", ":")),
    }
    strings = [(key + "=" + value).encode() for key, value in properties.items()]
    text = b"".join(bytes([len(value)]) + value for value in strings)
    records = [DNSText(_NAME, 16, 0x8001, 120, text)]
    if metadata:
        records += [
            DNSPointer(_TYPE, 12, 1, 120, _NAME),
            DNSService(_NAME, 33, 0x8001, 120, 0, 0, 8081, _HOST),
            DNSAddress(
                _HOST, 1, 0x8001, 120, ipaddress.IPv4Address("192.0.2.1").packed
            ),
        ]
    outgoing = DNSOutgoing(0x8400)
    for record in records:
        outgoing.add_answer_at_time(record, now)
    incoming = DNSIncoming(outgoing.packets()[0], now=now)
    assert incoming.valid
    return incoming


@pytest.fixture(autouse=True)
def real_asyncio(monkeypatch):
    # Legacy synchronous entity fixtures replace these asyncio functions.
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)


async def receive(outlets, *, serial=False, metadata=True):
    # Real parser, cache, browser and resolver; no sockets or household traffic.
    azc = AsyncZeroconf(interfaces=[], unicast=True, ip_version=IPVersion.V4Only)
    zc = azc.zeroconf
    await zc.async_wait_for_start()
    registry = XRegistryLocal(None)
    messages = []
    registry.dispatcher_connect("update", messages.append)
    now = current_time_millis()
    try:
        registry.start(zc)
        zc.record_manager.async_updates_from_response(
            packet("43", 0, now, metadata=metadata)
        )
        # Let discovery complete (or let the incomplete resolver start waiting).
        await asyncio.sleep(.01)
        for index, outlet in enumerate(outlets, 44):
            zc.record_manager.async_updates_from_response(
                packet(str(index), outlet, now + index - 43, metadata=metadata)
            )
            if serial:
                await asyncio.sleep(.01)
        if not metadata:
            # Resolve SRV/A only after multiple TXT versions are already cached.
            response = packet(str(43 + len(outlets)), outlets[-1], now + 10)
            zc.record_manager.async_updates_from_response(response)
        await asyncio.sleep(.05)
        return messages, now
    finally:
        await registry.stop()
        await azc.async_close()


@pytest.mark.parametrize("outlets", [(1, 0), (1, 1), (1, 0, 1, 0)])
def test_burst_keeps_each_sequence_and_its_own_button(outlets):
    messages, _ = asyncio.run(receive(outlets))
    assert [(m["seq"], m["params"]["localKeyPass"]["outlet"]) for m in messages] == [
        ("43", 0),
        *[(str(index), outlet) for index, outlet in enumerate(outlets, 44)],
    ], "Fast repeats and independent buttons must retain their original TXT payloads"


def test_burst_keeps_original_receipt_times():
    messages, now = asyncio.run(receive((1, 0)))
    assert [(m["seq"], m["received_at"]) for m in messages] == [
        ("43", now / 1000),
        ("44", (now + 1) / 1000),
        ("45", (now + 2) / 1000),
    ], "A later cached TXT must not replace the earlier report's receipt time"


def test_serial_receive_still_keeps_both_buttons():
    messages, _ = asyncio.run(receive((1, 0), serial=True))
    assert [m["seq"] for m in messages] == ["43", "44", "45"]


def test_late_metadata_does_not_replace_pending_txt_payloads():
    messages, _ = asyncio.run(receive((1, 0), metadata=False))
    received = {(m["seq"], m["params"]["localKeyPass"]["outlet"]) for m in messages}
    assert received == {
        ("43", 0), ("44", 1), ("45", 0)
    }, "SRV/A resolution must not discard earlier TXT versions"


def test_received_txt_snapshots_do_not_publish_phantom_button_events():
    messages, _ = asyncio.run(receive((1, 0)))
    registry = XRegistry(None)
    registry.config = {"devices": {}}
    entities = registry.setup_devices(
        [
            {
                "deviceid": _DEVICE,
                "name": "Office receive test",
                "extra": {"uiid": 161},
                "params": {},
            }
        ]
    )
    buttons = [entity for entity in entities if isinstance(entity, XButtonEvent)]
    assert len(buttons) == 2, "Exercise the real two-channel M5 button consumers"
    for message in messages:
        registry.local_update(message)
    assert all(
        button.state is None for button in buttons
    ), "Delivered LAN snapshots must retain the existing phantom-press guard"
