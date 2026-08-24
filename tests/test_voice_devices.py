"""Tests for nominating which microphone the assistant may listen on.

Nomination rather than classification, because Windows gives PortAudio only a
device *name* — no form factor, no bus, no built-in flag. "Is this an external
microphone?" could only be guessed from the string, and it fails both ways: a
virtual audio cable reads as external, and one pair of earbuds can enumerate as
three devices at once.

The behaviour that matters: when the nominated microphone is absent, the loop
waits. It must never quietly fall back to the built-in array, because listening
to the whole room is the thing naming a device was meant to prevent.
"""
from __future__ import annotations

import pytest

from app.voice import devices


@pytest.fixture(autouse=True)
def isolated_preference(tmp_path, monkeypatch):
    """Never touch the real preference file."""
    monkeypatch.setattr(devices, "PREFERENCE_PATH", tmp_path / "voice_device.json")


def fake_inputs(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(
        devices,
        "input_devices",
        lambda rescan=False: [
            devices.InputDevice(index=i, name=n, channels=1)
            for i, n in enumerate(names)
        ],
    )


# ── remembering a choice ─────────────────────────────────────────
def test_no_nomination_means_any_microphone():
    """Someone who has never chosen a device expects it to just work."""
    assert devices.remembered() is None
    assert devices.present(None) is True


def test_a_choice_survives_being_written_and_read():
    devices.remember("Galaxy Buds Pro")
    assert devices.remembered() == "Galaxy Buds Pro"

    devices.forget()
    assert devices.remembered() is None


def test_a_corrupt_preference_file_is_not_fatal(tmp_path):
    """Degrade, never crash: a damaged preference means "any microphone", not a
    dead listener (rule 10)."""
    devices.PREFERENCE_PATH.write_text("{not json", encoding="utf-8")
    assert devices.remembered() is None


# ── matching the same device across host APIs ────────────────────
def test_the_same_microphone_matches_despite_truncation():
    """MME clips device names at 31 characters, so one microphone appears as
    both "Microphone Array (Intel Smart" and the full name. Either has to match
    the other or the nominated device looks absent half the time."""
    short = "Microphone Array (Intel Smart"
    full = "Microphone Array (Intel Smart Sound Technology for Digital Microphones)"

    assert devices.matches(short, full)
    assert devices.matches(full, short)


def test_a_different_microphone_does_not_match():
    assert not devices.matches("Galaxy Buds Pro", "Microphone Array (Intel Smart")
    assert not devices.matches("", "Galaxy Buds")
    assert not devices.matches("Galaxy Buds", "")


# ── presence decides whether listening is allowed ────────────────
def test_the_nominated_device_being_absent_forbids_listening(monkeypatch):
    """The whole point of naming a device: no buds, no listening. Falling back
    to the built-in array would put the room back on the microphone."""
    fake_inputs(monkeypatch, ["Microphone Array (Intel Smart"])

    assert devices.present("Galaxy Buds", rescan=False) is False
    assert devices.find("Galaxy Buds", rescan=False) is None


def test_the_nominated_device_being_present_allows_it(monkeypatch):
    fake_inputs(monkeypatch, ["Microphone Array (Intel Smart", "Galaxy Buds Pro"])

    found = devices.find("Galaxy Buds", rescan=False)
    assert found is not None
    assert found.index == 1
    assert devices.present("Galaxy Buds", rescan=False) is True


def test_no_devices_at_all_is_handled(monkeypatch):
    fake_inputs(monkeypatch, [])
    assert devices.present("anything", rescan=False) is False


# ── the real machine ─────────────────────────────────────────────
def test_the_system_reports_at_least_one_microphone():
    """A live check, because every test above uses a stand-in. If PortAudio
    stops enumerating, the fakes would happily keep passing."""
    found = devices.input_devices(rescan=True)
    assert found, "no input devices enumerated at all"
    assert all(d.name for d in found)


def test_rescanning_does_not_raise():
    """PortAudio caches its device list at start-up, so a microphone plugged in
    afterwards is invisible until it is re-initialised. Measured at ~20 ms."""
    devices.refresh()
    assert devices.input_devices(rescan=True)
