"""Reject reports tied to commands superseded before their delayed delivery."""

import asyncio
from time import monotonic as REAL_MONOTONIC
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from custom_components.sonoff.core.ewelink import SIGNAL_UPDATE
from custom_components.sonoff.core.ewelink.local import XRegistryLocal, encrypt

from .test_local_reports import (
    reports as reports,  # noqa: PLC0414 - shared pytest fixture
)


def acknowledge_off_then_on(reports, monkeypatch):
    registry = reports.registry
    monkeypatch.setattr(registry.local, "send", AsyncMock(return_value="online"))
    monkeypatch.setattr(
        registry, "sequence", AsyncMock(side_effect=["1792304661763", "1792304662011"])
    )
    for now, value, sequence in (
        (100.1, "off", "1792304661763"),
        (100.2, "on", "1792304662011"),
    ):
        reports.clock[0] = now
        params = {"switches": [{"outlet": 0, "switch": value}]}
        asyncio.run(registry.send(reports.device, params))
        reports.clock[0] = now + 0.01
        reports.cloud(params["switches"], sequence)
    assert reports.relays[0].is_on is True


@pytest.mark.parametrize("delay", [0.1, 20.0, 600.0])
def test_delayed_own_cloud_ack_never_undoes_a_newer_success(
    reports, monkeypatch, delay
):
    acknowledge_off_then_on(reports, monkeypatch)
    reports.clock[0] = 100.21 + delay
    reports.cloud([{"outlet": 0, "switch": "off"}], "1792304661763")

    assert reports.relays[0].is_on is True
    assert reports.device["params"]["switches"][0]["switch"] == "on"
    assert all(button.state is None for button in reports.buttons)


@pytest.mark.parametrize("delay", [0.1, 20.0, 600.0])
@pytest.mark.parametrize("transport", ["lan", "cloud"])
def test_first_seen_unversioned_report_cannot_undo_a_confirmed_command(
    reports, monkeypatch, delay, transport
):
    acknowledge_off_then_on(reports, monkeypatch)
    reports.clock[0] = 100.21 + delay
    if transport == "lan":
        asyncio.run(reports.lan(["off", "on"], received_at=reports.clock[0]))
    else:
        reports.cloud([{"outlet": 0, "switch": "off"}])
    assert reports.relays[0].is_on is True
    assert reports.device["params"]["switches"][0]["switch"] == "on"


def test_an_uncommanded_sibling_and_sensor_still_update(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    reports.clock[0] = 120.0
    asyncio.run(reports.lan(["off", "off"], received_at=120.0))
    assert [relay.is_on for relay in reports.relays] == [True, False]
    assert reports.device["params"]["rssi"] == -61


def test_filtering_cloud_snapshot_keeps_its_non_press_marker(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    reports.clock[0] = 100.3
    reports.registry.cloud.dispatcher_send(
        SIGNAL_UPDATE,
        {
            "deviceid": reports.device["deviceid"],
            "params": {
                "switches": [{"outlet": 0, "switch": "off"}],
                "localKeyPass": {"outlet": 0, "key": 0},
            },
        },
    )
    assert all(button.state is None for button in reports.buttons)
    assert reports.relays[0].is_on is True


def test_new_commands_are_not_frozen_by_an_older_confirmation(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    reports.registry.sequence = AsyncMock(return_value="1792304664000")
    reports.clock[0] = 100.3
    asyncio.run(
        reports.registry.send(
            reports.device, {"switches": [{"outlet": 0, "switch": "off"}]}
        )
    )
    reports.cloud([{"outlet": 0, "switch": "off"}], "1792304664000")
    assert reports.relays[0].is_on is False


def test_a_failed_command_does_not_create_a_state_barrier(reports, monkeypatch):
    reports.registry.cloud.online = False
    monkeypatch.setattr(reports.registry.local, "send", AsyncMock(return_value="error"))
    asyncio.run(
        reports.registry.send(
            reports.device, {"switches": [{"outlet": 0, "switch": "on"}]}
        )
    )
    reports.cloud([{"outlet": 0, "switch": "off"}])
    assert reports.relays[0].is_on is False


def test_matching_feedback_before_ack_is_released_only_after_success(
    reports, monkeypatch
):
    acknowledge_off_then_on(reports, monkeypatch)
    reports.registry.sequence = AsyncMock(return_value="1792304664000")

    async def transport(*_args):
        reports.registry.local.dispatcher_send(
            SIGNAL_UPDATE,
            {
                "deviceid": reports.device["deviceid"],
                "received_at": 100.3,
                "params": {"switches": [{"outlet": 0, "switch": "off"}]},
            },
        )
        assert reports.relays[0].is_on is True
        return "online"

    reports.registry.local.send = transport
    reports.clock[0] = 100.3
    asyncio.run(
        reports.registry.send(
            reports.device, {"switches": [{"outlet": 0, "switch": "off"}]}
        )
    )
    assert reports.relays[0].is_on is False


def test_late_completion_of_an_older_command_cannot_reopen_its_state(
    reports, monkeypatch
):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.sequence = AsyncMock(side_effect=["1792304664000", "1792304665000"])

    async def transport(device, params, *_args):
        if params["switches"][0]["switch"] == "off":
            await registry.send(device, {"switches": [{"outlet": 0, "switch": "on"}]})
        return "online"

    registry.local.send = transport
    asyncio.run(
        registry.send(reports.device, {"switches": [{"outlet": 0, "switch": "off"}]})
    )
    reports.cloud([{"outlet": 0, "switch": "off"}], "1792304664000")
    assert reports.relays[0].is_on is True


class StateResponse:
    def __init__(self, params):
        self.headers = {"Content-Type": "application/json"}
        self.payload = {"error": 0, "seq": "1", **encrypt({"data": params}, "test-key")}

    async def read(self):
        return b""

    async def json(self):
        return self.payload

    def release(self):
        pass


def prepare_state_read(reports):
    local = reports.registry.local
    local.send = XRegistryLocal.send.__get__(local)
    reports.device["devicekey"] = "test-key"
    return local


def test_direct_state_read_supersedes_old_commands_and_cloud_query_cache(
    reports, monkeypatch
):
    acknowledge_off_then_on(reports, monkeypatch)
    local = prepare_state_read(reports)
    local.session = SimpleNamespace(
        post=AsyncMock(
            return_value=StateResponse({"switches": [{"outlet": 0, "switch": "off"}]})
        )
    )
    reports.registry.sequence = AsyncMock(return_value="1792304666000")
    reports.clock[0] = 100.4
    asyncio.run(reports.registry.send(reports.device))
    assert reports.relays[0].is_on is False

    reports.clock[0] = 100.5
    reports.cloud([{"outlet": 0, "switch": "on"}], "1792304662011")
    assert reports.relays[0].is_on is False


def test_state_read_started_before_a_newer_command_cannot_undo_it(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    local = prepare_state_read(reports)
    reports.registry.sequence = AsyncMock(
        side_effect=["1792304666000", "1792304667000"]
    )

    async def post(url, **kwargs):
        if url.endswith("getState"):
            await reports.registry.send(
                reports.device, {"switches": [{"outlet": 0, "switch": "on"}]}
            )
            return StateResponse({"switches": [{"outlet": 0, "switch": "off"}]})
        response = StateResponse({})
        response.payload = {"error": 0, "sequence": kwargs["json"]["sequence"]}
        return response

    local.session = SimpleNamespace(post=post)
    asyncio.run(reports.registry.send(reports.device))
    assert reports.relays[0].is_on is True


def test_cloud_only_button_actions_remain_fast_and_per_channel(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)

    async def press():
        with (
            patch.object(
                reports.buttons[0],
                "_trigger_event",
                wraps=reports.buttons[0]._trigger_event,
            ) as main,
            patch.object(
                reports.buttons[1],
                "_trigger_event",
                wraps=reports.buttons[1]._trigger_event,
            ) as ambient,
        ):
            for outlet in (0, 1, 0, 1):
                reports.registry.cloud.dispatcher_send(
                    SIGNAL_UPDATE,
                    {
                        "deviceid": reports.device["deviceid"],
                        "params": {"localKeyPass": {"outlet": outlet, "key": 0}},
                    },
                )
        return main.call_count, ambient.call_count

    assert asyncio.run(press()) == (2, 2)


def test_a_read_started_during_an_inflight_write_is_not_proof(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.sequence = AsyncMock(side_effect=["write-3", "read-4"])

    async def transport(device, params, _command, sequence, *_args):
        if params:
            await registry.send(device)
        else:
            registry.local.dispatcher_send(
                SIGNAL_UPDATE,
                {
                    "deviceid": device["deviceid"],
                    "request_sequence": sequence,
                    "params": {"switches": [{"outlet": 0, "switch": "off"}]},
                },
            )
            assert reports.relays[0].is_on is True
        return "online"

    registry.local.send = transport
    asyncio.run(
        registry.send(reports.device, {"switches": [{"outlet": 0, "switch": "on"}]})
    )
    assert reports.relays[0].is_on is True


def test_cancelled_write_does_not_block_a_later_verified_read(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.sequence = AsyncMock(side_effect=["cancelled-write", "fresh-read"])
    registry.local.send = AsyncMock(side_effect=asyncio.CancelledError)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            registry.send(
                reports.device, {"switches": [{"outlet": 0, "switch": "off"}]}
            )
        )
    local = prepare_state_read(reports)
    local.session = SimpleNamespace(
        post=AsyncMock(
            return_value=StateResponse({"switches": [{"outlet": 0, "switch": "off"}]})
        )
    )
    asyncio.run(registry.send(reports.device))
    assert reports.relays[0].is_on is False


def test_reload_clears_old_relay_command_evidence(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    monkeypatch.setattr(registry.local, "stop", AsyncMock())
    monkeypatch.setattr(registry.cloud, "stop", AsyncMock())
    asyncio.run(registry.stop())
    from custom_components.sonoff.switch import XSwitches

    relays = [
        entity
        for entity in registry.setup_devices([reports.device])
        if isinstance(entity, XSwitches)
    ]
    reports.cloud([{"outlet": 0, "switch": "off"}])
    assert relays[0].is_on is False


def test_other_device_models_keep_existing_report_behavior(reports, monkeypatch):
    reports.device["extra"]["uiid"] = 4
    acknowledge_off_then_on(reports, monkeypatch)
    reports.cloud([{"outlet": 0, "switch": "off"}])
    assert reports.relays[0].is_on is False


def test_two_coalesced_channels_keep_their_latest_independent_state(
    reports, monkeypatch
):
    import time

    monkeypatch.setattr(time, "monotonic", REAL_MONOTONIC)
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.sequence = AsyncMock(side_effect=["both-off", "both-on"])

    async def commands():
        for value, sequence in (("off", "both-off"), ("on", "both-on")):
            await asyncio.gather(
                *(
                    registry.send_bulk(
                        reports.device,
                        {"switches": [{"outlet": channel, "switch": value}]},
                    )
                    for channel in (0, 1)
                )
            )
            reports.cloud(
                [{"outlet": channel, "switch": value} for channel in (0, 1)],
                sequence,
            )
        await reports.lan(["off", "off"], received_at=REAL_MONOTONIC())

    asyncio.run(commands())
    assert [relay.is_on for relay in reports.relays] == [True, True]


@pytest.mark.parametrize("network_delay", [0.0, 20.0, 600.0])
def test_recorded_office_cycle_cannot_replay_old_off(
    reports, monkeypatch, network_delay
):
    import json
    from pathlib import Path

    acknowledge_off_then_on(reports, monkeypatch)
    path = Path(__file__).parent / "fixtures" / "office_delayed_reports.json"
    frames = json.loads(path.read_text())["reports"]
    ordered = sorted(
        frames,
        key=lambda frame: (
            frame["after_ack"] + (network_delay if frame["old_state"] else 0)
        ),
    )
    for frame in ordered:
        now = 100.21 + frame["after_ack"]
        if frame["old_state"]:
            now += network_delay
        reports.clock[0] = now
        message = {"deviceid": reports.device["deviceid"], "params": frame["params"]}
        if frame["transport"] == "lan":
            message.update(seq=frame["sequence"], received_at=now)
            reports.registry.local.dispatcher_send(SIGNAL_UPDATE, message)
        else:
            message["sequence"] = frame["sequence"]
            reports.registry.cloud.dispatcher_send(SIGNAL_UPDATE, message)
        assert [relay.is_on for relay in reports.relays] == [True, True]
        assert all(button.state is None for button in reports.buttons)


def test_numeric_cloud_sequence_identifies_its_own_pending_command(
    reports, monkeypatch
):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.sequence = AsyncMock(return_value="1792304664000")

    async def transport(_device, _params, _command, sequence, *_args):
        reports.cloud([{"outlet": 0, "switch": "off"}], int(sequence))
        assert reports.relays[0].is_on is False
        return "online"

    registry.local.send = transport
    reports.clock[0] = 100.4
    asyncio.run(
        registry.send(reports.device, {"switches": [{"outlet": 0, "switch": "off"}]})
    )
    assert reports.relays[0].is_on is False


def test_direct_command_reply_can_report_a_different_actual_state(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    local = prepare_state_read(reports)
    local.session = SimpleNamespace(
        post=AsyncMock(
            return_value=StateResponse({"switches": [{"outlet": 0, "switch": "on"}]})
        )
    )
    reports.registry.sequence = AsyncMock(return_value="actual-state-response")
    asyncio.run(
        reports.registry.send(
            reports.device, {"switches": [{"outlet": 0, "switch": "off"}]}
        )
    )
    reports.cloud([{"outlet": 0, "switch": "off"}])
    assert reports.relays[0].is_on is True


def test_newer_failed_write_does_not_hide_a_successful_pending_write(
    reports, monkeypatch
):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.cloud.online = False
    registry.sequence = AsyncMock(side_effect=["successful-off", "failed-on"])

    async def transport(device, params, _command, sequence):
        if params["switches"][0]["switch"] == "on":
            return "error"
        await registry.send(device, {"switches": [{"outlet": 0, "switch": "on"}]})
        registry.local.dispatcher_send(
            SIGNAL_UPDATE,
            {
                "deviceid": device["deviceid"],
                "request_sequence": sequence,
                "params": {"switches": [{"outlet": 0, "switch": "off"}]},
            },
        )
        assert reports.relays[0].is_on is False
        return "online"

    registry.local.send = transport
    asyncio.run(
        registry.send(reports.device, {"switches": [{"outlet": 0, "switch": "off"}]})
    )
    assert reports.relays[0].is_on is False


def test_cloud_error_payload_is_not_a_confirmed_state(reports, monkeypatch):
    acknowledge_off_then_on(reports, monkeypatch)
    registry = reports.registry
    registry.cloud.online = False
    registry.sequence = AsyncMock(return_value="failed-command")

    async def transport(device, _params, _command, sequence):
        registry.cloud.dispatcher_send(
            SIGNAL_UPDATE,
            {
                "deviceid": device["deviceid"],
                "sequence": sequence,
                "error": 1,
                "params": {"switches": [{"outlet": 0, "switch": "off"}]},
            },
        )
        assert reports.relays[0].is_on is True
        return "error"

    registry.local.send = transport
    asyncio.run(
        registry.send(reports.device, {"switches": [{"outlet": 0, "switch": "off"}]})
    )
    assert reports.relays[0].is_on is True
