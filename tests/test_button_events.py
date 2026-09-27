from unittest.mock import patch

import pytest

from custom_components.sonoff.core.ewelink import SIGNAL_UPDATE
from custom_components.sonoff.event import XButtonEvent

from . import DEVICEID, init


def setup_buttons(uiid=162):
    registry, entities = init(
        {
            "extra": {"uiid": uiid},
            "params": {"localKeyPass": {"key": 0, "outlet": 0}},
        }
    )
    return registry, [e for e in entities if isinstance(e, XButtonEvent)]


def cloud_press(registry, outlet=0, key=0, deviceid=DEVICEID):
    registry.cloud.dispatcher_send(
        SIGNAL_UPDATE,
        {"deviceid": deviceid, "params": {"localKeyPass": {"key": key, "outlet": outlet}}},
    )


@pytest.mark.parametrize("uiid,count", [(160, 1), (161, 2), (162, 3)])
def test_stable_button_identifiers_and_no_startup_events(uiid, count):
    registry, buttons = setup_buttons(uiid)
    assert [b.unique_id for b in buttons] == [
        f"{DEVICEID}_button_{i + 1}" for i in range(count)
    ]
    assert all(b.state is None for b in buttons)
    assert registry.send_args is None


@pytest.mark.parametrize(
    "key,event_type,attributes",
    [
        (0, "press_end", {}),
        (1, "multi_press_end", {"multi_press_count": 2}),
        (2, "long_press_end", {}),
        (3, "multi_press_end", {"multi_press_count": 3}),
    ],
)
def test_button_action_mapping(key, event_type, attributes):
    registry, buttons = setup_buttons()
    cloud_press(registry, outlet=1, key=key)
    assert buttons[1].state_attributes == {"event_type": event_type, **attributes}
    assert buttons[0].state is None
    assert buttons[2].state is None
    assert registry.send_args is None


def test_fast_repeats_and_other_buttons_are_independent():
    registry, buttons = setup_buttons()
    with (
        patch.object(buttons[0], "_trigger_event", wraps=buttons[0]._trigger_event) as first,
        patch.object(buttons[1], "_trigger_event", wraps=buttons[1]._trigger_event) as second,
    ):
        cloud_press(registry, 0)
        cloud_press(registry, 1)
        cloud_press(registry, 0)
        cloud_press(registry, 1)
    assert first.call_count == 2
    assert second.call_count == 2
    assert registry.send_args is None


@pytest.mark.parametrize("trigger", [0, 2, 11, 29])
@pytest.mark.parametrize("full_snapshot", [False, True])
def test_lan_reports_never_replay_a_button(trigger, full_snapshot):
    registry, buttons = setup_buttons()
    cloud_press(registry)
    previous_state = buttons[0].state
    assert previous_state is not None
    for seq in (1, 1, 2, 3):
        params = {"triggerType": trigger, "localKeyPass": {"key": 0, "outlet": 0}}
        if full_snapshot:
            params["switches"] = [{"outlet": 0, "switch": "on"}]
        registry.local.dispatcher_send(
            SIGNAL_UPDATE, {"deviceid": DEVICEID, "params": params, "seq": seq}
        )
    assert buttons[0].state == previous_state
    assert registry.send_args is None


def test_cloud_state_snapshot_does_not_replay_cached_action():
    registry, buttons = setup_buttons()
    registry.cloud.dispatcher_send(
        SIGNAL_UPDATE,
        {
            "deviceid": DEVICEID,
            "params": {
                "switches": [{"outlet": 0, "switch": "on"}],
                "localKeyPass": {"key": 0, "outlet": 0},
            },
        },
    )
    assert all(b.state is None for b in buttons)
    assert registry.send_args is None


def test_events_do_not_leak_between_devices():
    registry, entities = init(
        [
            {"deviceid": deviceid, "extra": {"uiid": 160}, "params": {}}
            for deviceid in (DEVICEID, "other_device")
        ]
    )
    buttons = [e for e in entities if isinstance(e, XButtonEvent)]
    cloud_press(registry)
    assert buttons[0].state is not None
    assert buttons[1].state is None
