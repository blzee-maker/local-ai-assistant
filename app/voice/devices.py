"""Choosing which microphone the assistant is allowed to listen on.

A wake word that runs on the built-in array listens to the whole room: the
television, the person next to you, a call on speaker. Nominating a device
instead — a headset, a pair of buds — makes the microphone a close-talking one
and, more usefully, makes *putting it on* the gesture that enables listening.

This is a convenience, not the guarantee. Listening is still off until it is
switched on and is always visible while it runs; "my buds are connected" is not
consent, in the same way that having approved folder access is not consent to
start a disk scan.

Nomination rather than classification, because Windows tells us almost nothing.
PortAudio exposes a device *name* and nothing else — no form factor, no bus, no
built-in flag — so "is this an external microphone?" could only be guessed from
the string, which fails in both directions: a virtual audio cable reads as
external, and one pair of earbuds can enumerate as three devices at once. The
form factor does exist in the Windows MMDevice API, behind COM. Asking the user
once is cheaper and cannot be wrong.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from config import settings

PREFERENCE_PATH = Path(settings.upload_dir).parent / "voice_device.json"


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    channels: int


def _sounddevice():
    try:
        import sounddevice as sd

        return sd
    except Exception:
        return None


def refresh() -> None:
    """Re-read the device list from the operating system.

    PortAudio caches devices when it initialises, so a microphone plugged in
    after the assistant started is invisible until it is torn down and brought
    back. Measured at about 20 ms, which is nothing next to how often this is
    called — once per listening attempt, not per frame.
    """
    sd = _sounddevice()
    if sd is None:
        return
    try:
        sd._terminate()
        sd._initialize()
    except Exception:
        # A failure here means the device list is merely stale, which is far
        # better than taking the listener down with it (rule 10).
        pass


def input_devices(rescan: bool = False) -> list[InputDevice]:
    """Every device that can capture, newest view of the system."""
    sd = _sounddevice()
    if sd is None:
        return []
    if rescan:
        refresh()
    try:
        found = []
        for index, device in enumerate(sd.query_devices()):
            if device.get("max_input_channels", 0) > 0:
                found.append(
                    InputDevice(
                        index=index,
                        name=str(device.get("name", "")).strip(),
                        channels=int(device["max_input_channels"]),
                    )
                )
        return found
    except Exception:
        return []


def default_input_name() -> str | None:
    sd = _sounddevice()
    if sd is None:
        return None
    try:
        return str(sd.query_devices(kind="input")["name"]).strip()
    except Exception:
        return None


# ── the nominated device ─────────────────────────────────────────
def remember(name: str) -> None:
    """Record which microphone the assistant may listen on."""
    PREFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREFERENCE_PATH.write_text(
        json.dumps({"name": name}, indent=2), encoding="utf-8"
    )


def forget() -> None:
    PREFERENCE_PATH.unlink(missing_ok=True)


def remembered() -> str | None:
    """The nominated device name, or None when any microphone will do."""
    try:
        data = json.loads(PREFERENCE_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    name = data.get("name")
    return str(name).strip() if name else None


def matches(device_name: str, wanted: str) -> bool:
    """Is this device the nominated one?

    Compared loosely because the same microphone appears under several host
    APIs with names that differ in truncation and punctuation: MME clips at 31
    characters, so "Microphone Array (Intel Smart Sound Technology)" arrives as
    "Microphone Array (Intel® Smart". Substring either way catches both.
    """
    left = " ".join(device_name.lower().split())
    right = " ".join(wanted.lower().split())
    if not left or not right:
        return False
    return right in left or left in right


def find(wanted: str, rescan: bool = True) -> InputDevice | None:
    """The nominated device if it is connected right now, else None."""
    for device in input_devices(rescan=rescan):
        if matches(device.name, wanted):
            return device
    return None


def present(wanted: str | None, rescan: bool = True) -> bool:
    """Is listening allowed at this moment?

    No nomination means any microphone is acceptable, which is what someone who
    has never run `/listen device` expects.
    """
    if not wanted:
        return True
    return find(wanted, rescan=rescan) is not None
