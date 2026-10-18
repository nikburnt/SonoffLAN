"""Keep confirmed M5 relay state until a causally newer update is available."""

import time
from dataclasses import dataclass, field


@dataclass
class _Request:
    sequence: str
    deviceid: str
    version: int
    started: float
    values: dict[int, str] = field(default_factory=dict)
    before: dict[int, int] = field(default_factory=dict)
    observed: set[int] = field(default_factory=set)


class RelayReports:
    """Per-channel command/read evidence; arrival time is not a state version."""

    def __init__(self):
        self._version = 0
        self._commands: dict[str, dict[int, _Request]] = {}
        self._confirmed: dict[str, dict[int, tuple[int, str]]] = {}
        self._pending: dict[str, _Request] = {}
        self._reads: dict[str, _Request] = {}

    def start(self, device: dict, sequence: str, params: dict | None):
        if device.get("extra", {}).get("uiid") not in (160, 161, 162):
            return None
        did = device["deviceid"]
        sequence = str(sequence)
        if params and "switches" not in params:
            return None
        if not params and not self._confirmed.get(did):
            return None
        self._version += 1
        request = _Request(sequence, did, self._version, time.monotonic())
        if params:
            request.values = {c["outlet"]: c["switch"] for c in params["switches"]}
            commands = self._commands.setdefault(did, {})
            for outlet in request.values:
                commands[outlet] = request
        else:
            busy = {
                outlet
                for pending in self._pending.values()
                if pending.deviceid == did
                for outlet in pending.values
            }
            request.before = {
                outlet: self._commands[did][outlet].version
                for outlet in self._confirmed[did]
                if outlet not in busy
            }
            self._reads[did] = request
        self._pending[sequence] = request
        return request

    def finish(self, request: _Request | None, success: bool):
        if request is None or self._pending.get(request.sequence) is not request:
            return []
        del self._pending[request.sequence]
        released = []
        if success:
            for outlet, value in request.values.items():
                states = self._confirmed.setdefault(request.deviceid, {})
                if request.version <= states.get(outlet, (0, ""))[0]:
                    continue
                states[outlet] = (request.version, value)
                if outlet in request.observed:
                    released.append({"outlet": outlet, "switch": value})
        return released

    def accept(self, deviceid, channel, received_at, sequence=None, direct=False):
        sequence = str(sequence) if sequence is not None else None
        outlet, value = channel["outlet"], channel["switch"]
        states = self._confirmed.get(deviceid, {})
        version, expected = states.get(outlet, (0, None))
        command = self._commands.get(deviceid, {}).get(outlet)
        reply = self._pending.get(sequence)
        if not reply or reply.deviceid != deviceid or outlet not in reply.values:
            reply = command
        if (
            reply
            and sequence == reply.sequence
            and reply.version >= version
            and (direct or value == reply.values[outlet])
        ):
            self._confirmed.setdefault(deviceid, {})[outlet] = (reply.version, value)
            return True, True
        # Unversioned feedback is not published against an existing fence. If
        # it matches a pending write, that write's ACK can validate it later.
        for pending in self._pending.values():
            if (
                pending.deviceid == deviceid
                and pending.values.get(outlet) == value
                and received_at >= pending.started
            ):
                pending.observed.add(outlet)
        read = self._reads.get(deviceid)
        if (
            direct
            and read
            and sequence == read.sequence
            and outlet in read.before
            and command
            and read.before[outlet] == command.version
            and read.version >= version
        ):
            del read.before[outlet]
            self._confirmed[deviceid][outlet] = (read.version, value)
            return True, True
        return expected is None or value == expected, False

    def clear(self):
        self._commands.clear()
        self._confirmed.clear()
        self._pending.clear()
        self._reads.clear()
