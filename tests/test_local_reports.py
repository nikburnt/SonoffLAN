"""Keep cached mDNS snapshots from undoing newer per-channel state reports."""

import asyncio
import ipaddress
import json
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from zeroconf import DNSText
from zeroconf._cache import DNSCache

from custom_components.sonoff.core.ewelink import SIGNAL_UPDATE, XRegistry
from custom_components.sonoff.core.ewelink import local as local_module
from custom_components.sonoff.event import XButtonEvent
from custom_components.sonoff.switch import XSwitches

_CREATE_TASK = asyncio.create_task
_GET_RUNNING_LOOP = asyncio.get_running_loop
_SERVICE_TYPE = "_ewelink._tcp.local."
_DEVICE_ID = "1000123abc"
_SERVICE_NAME = f"eWeLink_{_DEVICE_ID}.{_SERVICE_TYPE}"


@pytest.fixture
def reports(monkeypatch):
    # The old synchronous entity fixtures replace these asyncio globals.
    monkeypatch.setattr(asyncio, "create_task", _CREATE_TASK)
    monkeypatch.setattr(asyncio, "get_running_loop", _GET_RUNNING_LOOP)
    clock = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    registry = XRegistry(None)
    registry.config = {"devices": {}}
    registry.cloud.online = registry.local.online = True
    device = {
        "deviceid": _DEVICE_ID,
        "name": "Office Window",
        "extra": {"uiid": 161},
        "online": True,
        "local": True,
        "host": "192.168.1.2:8081",
        "params": {
            "switches": [
                {"outlet": 0, "switch": "on"},
                {"outlet": 1, "switch": "on"},
            ]
        },
    }
    entities = registry.setup_devices([device])
    relays = [entity for entity in entities if isinstance(entity, XSwitches)]
    buttons = [entity for entity in entities if isinstance(entity, XButtonEvent)]
    cache = DNSCache()
    zc = SimpleNamespace(cache=cache)

    def cloud(switches, sequence=None):
        registry.cloud.dispatcher_send(
            SIGNAL_UPDATE,
            {
                "deviceid": _DEVICE_ID,
                "sequence": sequence,
                "params": {"switches": switches},
            },
        )

    async def lan(states, *, received_at, sequence="325"):
        params = {
            "switches": [
                {"outlet": outlet, "switch": state}
                for outlet, state in enumerate(states)
            ],
            "triggerType": 11,
            "localKeyPass": {"outlet": 0, "key": 0},
            "rssi": -61,
        }
        properties = {
            b"id": _DEVICE_ID.encode(),
            b"type": b"plug",
            b"seq": sequence.encode(),
            b"data1": json.dumps(params).encode(),
        }
        text = b"".join(
            bytes([len(k) + len(v) + 1]) + k + b"=" + v for k, v in properties.items()
        )
        record = DNSText(_SERVICE_NAME, 16, 1, 120, text)
        record.created = received_at * 1000
        cache.async_add_records([record])
        info = SimpleNamespace(
            name=_SERVICE_NAME,
            text=text,
            properties=properties,
            addresses=[ipaddress.IPv4Address("192.168.1.2").packed],
            port=8081,
            server="office-window.local.",
            async_request=AsyncMock(return_value=True),
        )
        # Only the DNS request is external; use real TXT records and cache times.
        with patch.object(local_module, "AsyncServiceInfo", return_value=info):
            await registry.local._handler2(zc, _SERVICE_TYPE, _SERVICE_NAME)

    return SimpleNamespace(
        clock=clock,
        registry=registry,
        device=device,
        relays=relays,
        buttons=buttons,
        cloud=cloud,
        lan=lan,
    )


def test_first_delayed_lan_snapshot_cannot_undo_newer_cloud_confirmation(reports):
    """The office seq=325 packet predates the confirmed second Main command."""
    reports.clock[0] = 100.489
    reports.cloud([{"outlet": 0, "switch": "on"}], "1792253487242")
    reports.clock[0] = 100.583
    asyncio.run(reports.lan(["off", "on"], received_at=100.219))

    assert [relay.is_on for relay in reports.relays] == [True, True]
    assert reports.device["params"]["switches"][0]["switch"] == "on"
    assert reports.device["params"]["rssi"] == -61
    assert reports.device["local"] is True
    assert all(button.state is None for button in reports.buttons)


def test_repeated_txt_ttl_refresh_does_not_make_an_old_snapshot_new(reports):
    """Reannouncing the same TXT version must retain its first observation time."""
    reports.clock[0] = 100.300
    asyncio.run(reports.lan(["off", "on"], received_at=100.219))
    reports.clock[0] = 100.489
    reports.cloud([{"outlet": 0, "switch": "on"}], "1792253487242")
    reports.clock[0] = 101.345
    asyncio.run(reports.lan(["off", "on"], received_at=101.345))

    assert [relay.is_on for relay in reports.relays] == [True, True]


def test_delayed_snapshot_filters_channels_independently(reports):
    """An older Main value must not discard the independent Ambient update."""
    reports.clock[0] = 100.489
    reports.cloud([{"outlet": 0, "switch": "on"}])
    reports.clock[0] = 100.583
    asyncio.run(reports.lan(["off", "off"], received_at=100.219))

    assert [relay.is_on for relay in reports.relays] == [True, False]


def test_fresh_lan_changes_and_restarted_counter_are_still_accepted(reports):
    """No grace period or numeric LAN/cloud counter comparison blocks new data."""
    reports.clock[0] = 100.489
    reports.cloud([{"outlet": 0, "switch": "on"}])
    reports.clock[0] = 100.590
    asyncio.run(reports.lan(["off", "on"], received_at=100.589, sequence="326"))
    assert [relay.is_on for relay in reports.relays] == [False, True]

    reports.clock[0] = 100.600
    asyncio.run(reports.lan(["on", "off"], received_at=100.599, sequence="1"))
    assert [relay.is_on for relay in reports.relays] == [True, False]


def test_delayed_old_version_after_new_txt_cannot_roll_state_back(reports):
    """Keep observation times when an old TXT version reappears out of order."""
    reports.clock[0] = 100.300
    asyncio.run(reports.lan(["off", "on"], received_at=100.219))
    reports.clock[0] = 100.600
    asyncio.run(reports.lan(["on", "on"], received_at=100.599, sequence="326"))
    reports.clock[0] = 100.700
    asyncio.run(reports.lan(["off", "on"], received_at=100.699))
    assert [relay.is_on for relay in reports.relays] == [True, True]


def test_filtered_minimal_snapshot_cannot_become_a_phantom_press(reports):
    """Filtering relay values must retain the upstream state-snapshot marker."""
    reports.clock[0] = 100.489
    reports.cloud([{"outlet": 0, "switch": "on"}])

    async def publish():
        reports.registry.local.dispatcher_send(
            SIGNAL_UPDATE,
            {
                "deviceid": _DEVICE_ID,
                "received_at": 100.219,
                "params": {
                    "switches": [{"outlet": 0, "switch": "off"}],
                    "localKeyPass": {"outlet": 0, "key": 0},
                },
            },
        )

    reports.clock[0] = 100.583
    asyncio.run(publish())
    assert all(button.state is None for button in reports.buttons)
    assert reports.relays[0].is_on
