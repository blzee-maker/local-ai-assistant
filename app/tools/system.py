"""System awareness and control — what the machine is doing, and stopping it.

Three read tools and one destructive one. The read tools are unremarkable; the
destructive one is where the care goes, because "kill that process" is the first
thing this assistant can do that a user cannot undo.

Guards on ending a process, in order of how badly each failure would land:

* **Protected processes are refused outright.** Terminating csrss.exe or
  winlogon.exe does not close a program, it bluescreens Windows. No confirmation
  prompt should be able to authorise that, so the check sits before the prompt.
* **Ambiguous names are refused, not guessed.** "close chrome" with fourteen
  chrome.exe processes must not become fourteen terminations. The matches are
  listed and the user picks (rule 6).
* **Identity is re-verified after confirmation.** PIDs are recycled. Between
  the model naming PID 8420 and the user approving it, that PID can belong to
  something else entirely — so the name and start time are checked again at the
  moment of the kill, not just when the target was chosen.
* **The assistant will not end itself, its parent, or its own model server.**
  Not a safety issue so much as an obviously useless outcome.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Any

from app.tools.base import Risk, Tool, ToolContext, ToolResult

# Ending any of these does not close a program — it takes the machine down or
# logs the user out. No prompt can authorise it.
_PROTECTED_WINDOWS = {
    "system", "system idle process", "registry", "memory compression",
    "smss.exe", "csrss.exe", "wininit.exe", "winlogon.exe", "services.exe",
    "lsass.exe", "lsaiso.exe", "svchost.exe", "fontdrvhost.exe", "dwm.exe",
    "sihost.exe", "ctfmon.exe", "spoolsv.exe", "audiodg.exe", "conhost.exe",
    "msmpeng.exe", "securityhealthservice.exe", "wudfhost.exe",
}
_PROTECTED_POSIX = {
    "init", "systemd", "kernel_task", "launchd", "kthreadd", "systemd-journald",
    "systemd-logind", "dbus-daemon", "sshd", "windowserver", "loginwindow",
}

# Terminating the model server would end the conversation mid-sentence.
_SELF_CRITICAL = {"ollama.exe", "ollama", "ollama app.exe", "llama-server", "llama-server.exe"}

# Windows accounts for unused CPU time against a pseudo-process, so on an idle
# machine "System Idle Process" sits at the top of any CPU ranking with 70-90%.
# Reporting it as the biggest consumer inverts the truth: that number is how
# much CPU is *free*. Excluded from process listings entirely.
_IDLE_PROCESS_NAMES = {"system idle process", "idle"}
_IDLE_PROCESS_PIDS = {0}

BYTES_GB = 1024 ** 3
BYTES_MB = 1024 ** 2


def is_idle_process(pid: int, name: str) -> bool:
    return pid in _IDLE_PROCESS_PIDS or (name or "").lower().strip() in _IDLE_PROCESS_NAMES


def _psutil():
    import psutil

    return psutil


def protected_names() -> set[str]:
    return _PROTECTED_WINDOWS if sys.platform == "win32" else _PROTECTED_POSIX


def is_protected(name: str) -> bool:
    return (name or "").lower().strip() in protected_names()


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cpu_percent: float
    memory_mb: float
    create_time: float
    username: str = ""

    @property
    def identity(self) -> tuple[int, str, int]:
        """PID alone is not an identity — PIDs get recycled."""
        return (self.pid, self.name.lower(), int(self.create_time))


def sample_processes(
    interval: float = 0.4,
    limit: int | None = None,
    sort_by: str = "cpu",
) -> list[ProcessInfo]:
    """Snapshot running processes with a real CPU reading.

    `cpu_percent()` needs two samples to mean anything; the first call on a
    process always returns 0.0. So we prime every process, wait briefly, then
    read — otherwise "what is using my CPU" answers "nothing, everything is 0%".
    """
    psutil = _psutil()

    procs = []
    for proc in psutil.process_iter(["pid", "name", "create_time", "username"]):
        try:
            proc.cpu_percent(None)  # prime
            procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    time.sleep(interval)

    cpu_count = psutil.cpu_count() or 1
    results: list[ProcessInfo] = []
    for proc in procs:
        try:
            info = proc.info
            if is_idle_process(info["pid"], info["name"] or ""):
                continue
            # psutil reports per-core percentages; normalise so the numbers add
            # up to something a human recognises as "percent of this machine".
            cpu = proc.cpu_percent(None) / cpu_count
            memory = proc.memory_info().rss / BYTES_MB
            results.append(
                ProcessInfo(
                    pid=info["pid"],
                    name=info["name"] or "?",
                    cpu_percent=cpu,
                    memory_mb=memory,
                    create_time=info["create_time"] or 0.0,
                    username=(info.get("username") or "").split("\\")[-1],
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # The ordering is not cosmetic, because the list is then truncated. Sorted
    # by CPU and cut to five, "the top 5 things consuming my memory" returned
    # the five busiest processes and called them the biggest memory users —
    # the actual largest one need not appear at all.
    if sort_by == "memory":
        results.sort(key=lambda p: (p.memory_mb, p.cpu_percent), reverse=True)
    else:
        results.sort(key=lambda p: (p.cpu_percent, p.memory_mb), reverse=True)
    return results[:limit] if limit else results


def cpu_name() -> str:
    """A human-recognisable processor name.

    `platform.processor()` returns something like "Intel64 Family 6 Model 140"
    on Windows — technically correct and useless to a person asking what CPU
    they have. The registry holds the marketing name the machine was sold under,
    which is what they mean.
    """
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            with key:
                value, _type = winreg.QueryValueEx(key, "ProcessorNameString")
            if value:
                return " ".join(str(value).split())
        except Exception:
            pass

    import platform

    return platform.processor() or platform.machine() or "unknown"


def hardware_info() -> dict[str, Any]:
    """Static facts about the machine — the things "what are my specs?" means."""
    import platform

    psutil = _psutil()
    uname = platform.uname()

    frequency = None
    try:
        freq = psutil.cpu_freq()
        if freq and freq.max:
            frequency = round(freq.max / 1000.0, 2)
        elif freq and freq.current:
            frequency = round(freq.current / 1000.0, 2)
    except Exception:
        frequency = None

    return {
        "cpu_name": cpu_name(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_max_ghz": frequency,
        "os": f"{uname.system} {uname.release}",
        "os_version": uname.version,
        "architecture": uname.machine,
        "hostname": uname.node,
    }


def system_snapshot() -> dict[str, Any]:
    """Everything the status tool reports, in one structure."""
    psutil = _psutil()

    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    battery = None
    try:
        sensor = psutil.sensors_battery()
        if sensor is not None:
            battery = {
                "percent": round(sensor.percent),
                "plugged_in": bool(sensor.power_plugged),
                "minutes_left": (
                    round(sensor.secsleft / 60)
                    if sensor.secsleft and sensor.secsleft > 0 else None
                ),
            }
    except Exception:
        battery = None

    disks = []
    for part in psutil.disk_partitions(all=False):
        try:
            usage = psutil.disk_usage(part.mountpoint)
        except (PermissionError, OSError):
            continue  # empty optical drive, unreadable mount
        disks.append(
            {
                "mount": part.mountpoint,
                "total_gb": round(usage.total / BYTES_GB, 1),
                "free_gb": round(usage.free / BYTES_GB, 1),
                "used_gb": round(usage.used / BYTES_GB, 1),
                "percent_used": usage.percent,
            }
        )

    return {
        "hardware": hardware_info(),
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpu_count": psutil.cpu_count(),
        "memory": {
            "total_gb": round(memory.total / BYTES_GB, 2),
            "available_gb": round(memory.available / BYTES_GB, 2),
            "used_gb": round((memory.total - memory.available) / BYTES_GB, 2),
            "percent_used": memory.percent,
        },
        "swap": {
            "total_gb": round(swap.total / BYTES_GB, 2),
            "percent_used": swap.percent,
        },
        "battery": battery,
        "disks": disks,
        "uptime_hours": round((time.time() - psutil.boot_time()) / 3600, 1),
    }


def snapshot_focus(text: str) -> str | None:
    """Which part of the machine the question is about, if it is only one.

    Filtering happens here rather than in the prompt because asking a 3B model
    to ignore most of what it was handed does not work. Given the whole machine
    plus a ten-row process table and told to report only the memory, it answered
    "a detailed report on memory usage" with the processor, the OS and every
    drive; when the process table was added it started dropping the process
    names instead. A section never shown also cannot be misquoted.
    """
    lowered = text.lower()
    areas = {
        "memory": ("memory", "ram", "swap"),
        "disk": ("disk", "drive", "storage", "space", "ssd", "hard drive"),
        "cpu": ("cpu", "processor", "core"),
    }
    hits = [name for name, words in areas.items() if any(w in lowered for w in words)]
    # Only when the question is about exactly one area. Two means a comparison
    # or a general enquiry, and both want the whole reading.
    return hits[0] if len(hits) == 1 else None


def describe_snapshot(snapshot: dict[str, Any], focus: str | None = None) -> str:
    """Snapshot as prose the model can answer from.

    `focus` narrows it to one area (see `snapshot_focus`); the default is the
    whole machine, which is what a general question wants.
    """
    memory = snapshot["memory"]
    hardware = snapshot.get("hardware") or {}
    sections: dict[str, list[str]] = {
        "hardware": [], "cpu": [], "memory": [], "misc": [], "disk": []
    }

    if hardware:
        cores = hardware.get("cpu_cores_physical")
        threads = hardware.get("cpu_cores_logical")
        speed = hardware.get("cpu_max_ghz")
        cpu_line = f"Processor: {hardware.get('cpu_name', 'unknown')}"
        if cores or threads:
            cpu_line += f" ({cores or '?'} cores / {threads or '?'} threads"
            cpu_line += f", up to {speed} GHz)" if speed else ")"
        sections["hardware"].append(cpu_line)
        sections["hardware"].append(
            f"Operating system: {hardware.get('os', 'unknown')} "
            f"({hardware.get('architecture', '')})".strip()
        )

    # Every figure is labelled with which way round it runs. A 3B model given
    # "415.8 GB (72% used)" reported it back as "72% free space" — an inversion
    # that turns a nearly-full disk into a healthy one. Spelling out both halves
    # leaves nothing to infer.
    sections["cpu"].append(
        f"CPU load: {snapshot['cpu_percent']:.0f}% of capacity in use, "
        f"across {snapshot['cpu_count']} cores"
    )
    sections["memory"].append(
        f"Memory: {memory['total_gb']:.2f} GB total, "
        # Derived when absent: a missing key must not take the whole report
        # down over a number we can work out (rule 10).
        f"{memory.get('used_gb', memory['total_gb'] - memory['available_gb']):.2f}"
        " GB in use, "
        f"{memory['available_gb']:.2f} GB still free "
        f"({memory['percent_used']:.0f}% of memory is in use)"
    )
    if snapshot["swap"]["total_gb"]:
        sections["memory"].append(f"Swap: {snapshot['swap']['percent_used']:.0f}% used")

    sections["misc"].append(f"Uptime: {snapshot['uptime_hours']:.1f} hours")
    battery = snapshot.get("battery")
    if battery:
        state = "charging" if battery["plugged_in"] else "on battery"
        left = (
            f", about {battery['minutes_left']} minutes left"
            if battery["minutes_left"] else ""
        )
        sections["misc"].append(f"Battery: {battery['percent']}% ({state}{left})")

    if snapshot["disks"]:
        sections["disk"].append(
            f"Drives attached: {len(snapshot['disks'])} "
            f"({', '.join(d['mount'] for d in snapshot['disks'])}). "
            "There are no other drives on this machine."
        )
    for disk in snapshot["disks"]:
        used_gb = disk.get("used_gb", disk["total_gb"] - disk["free_gb"])
        sections["disk"].append(
            f"Drive {disk['mount']}: {disk['total_gb']:.1f} GB total, "
            f"{used_gb:.1f} GB in use, "
            f"{disk['free_gb']:.1f} GB still free "
            f"({disk['percent_used']:.0f}% of the drive is in use)"
        )

    # A small model will not infer "1.8GB free is tight" on its own, and this is
    # the whole question behind "why is my laptop slow?".
    if memory["percent_used"] >= 85:
        sections["memory"].append(
            "NOTE: memory is nearly exhausted, which is the most likely cause of "
            "slowness on this machine."
        )
    if snapshot["cpu_percent"] >= 85:
        sections["cpu"].append(
            "NOTE: the processor is saturated right now."
        )

    if focus in ("memory", "disk"):
        wanted = [focus]
    elif focus == "cpu":
        wanted = ["hardware", "cpu"]
    else:
        wanted = ["hardware", "cpu", "memory", "misc", "disk"]

    lines = [line for name in wanted for line in sections[name]]
    return "\n".join(lines)


def describe_processes(
    procs: list[ProcessInfo], limit: int = 10, sort_by: str = "cpu"
) -> str:
    # Say which way the list is ordered. Presented as a bare "top processes",
    # a memory-ordered list reads as a CPU one and the model ranks it wrongly.
    ordered = "largest first by memory" if sort_by == "memory" else "busiest first by CPU"
    lines = [f"Top processes, {ordered} (percent of total CPU, resident memory):"]
    for rank, proc in enumerate(procs[:limit], 1):
        lines.append(
            f"#{rank}. {proc.name} (PID {proc.pid}) — "
            f"{proc.cpu_percent:.1f}% CPU, {proc.memory_mb:,.0f} MB"
        )
    return "\n".join(lines)



def count_triggers(text: str, triggers) -> int:
    """How many distinct trigger phrases appear. The tie-break for the backstop."""
    lowered = text.lower()
    return sum(1 for trigger in triggers if trigger in lowered)


# Two things a question can ask for beyond a single number. Both were learned
# from the same session: "get me a detailed report on memory usage" and "list
# top 5 things that are consuming my memory" came back as the identical one-line
# total, because neither the depth nor the subject of the question reached the
# tool.
_DETAIL_PHRASES = (
    "detailed", "in detail", "full report", "report on", "breakdown",
    "break down", "comprehensive", "in depth", "in-depth", "everything",
    "complete picture", "analysis", "analyse", "analyze", "elaborate",
)

# Deliberately generous, for the same asymmetry as the trigger lists: a false
# positive costs one extra process sample, a miss answers the wrong question.
_CONSUMPTION_PHRASES = (
    "what is using", "what's using", "what is consuming", "what's consuming",
    "consuming", "consume", "taking up", "taking my", "eating", "hogging",
    "using the most", "using most", "most memory", "most ram", "most cpu",
    "which process", "what process", "processes", "top 3", "top 5", "top 10",
    "biggest", "worst offender", "what is running", "what's running",
)


_RESOURCE_WORDS = (
    "memory", "ram", "cpu", "processor", "resource", "resources",
    "performance", "slow", "sluggish",
)


def preferred_process_sort(text: str) -> str:
    """Order the process list by whatever the question is about."""
    lowered = text.lower()
    memory = any(word in lowered for word in ("memory", "ram"))
    cpu = any(word in lowered for word in ("cpu", "processor", "busy", "compute"))
    return "memory" if memory and not cpu else "cpu"


def asks_about_resources(text: str) -> bool:
    """Is the question about memory or CPU, as opposed to disks or the OS?"""
    lowered = text.lower()
    return any(word in lowered for word in _RESOURCE_WORDS)


def asks_for_detail(text: str) -> bool:
    """Did the user ask for more than a one-line answer?"""
    lowered = text.lower()
    return any(phrase in lowered for phrase in _DETAIL_PHRASES)


def asks_what_is_consuming(text: str) -> bool:
    """Is the question about *what* is using the resources, not how much?"""
    lowered = text.lower()
    return any(phrase in lowered for phrase in _CONSUMPTION_PHRASES)


# ── tools ────────────────────────────────────────────────────────
class SystemStatusTool(Tool):
    name = "system_status"
    description = "Report CPU, memory, disk, battery, and uptime for this machine"
    risk = Risk.READ

    # Deliberately generous. For a read-only tool the two failure modes are not
    # remotely symmetric: a false match costs one cheap local call, while a
    # missed match means the model answers from imagination — which it does
    # confidently. Asked "give me my system information" with a narrower list,
    # it invented 16 GB of RAM and three hard drives on a machine with 7.8 GB
    # and one. Keyword pre-filters exist to save cost, never at the price of
    # correctness (CLAUDE.md conventions).
    _TRIGGERS = (
        # the machine itself
        "system", "machine", "computer", "laptop", "desktop pc", "my pc",
        "this pc", "hardware", "spec", "specs", "specification", "configuration",
        # components
        "cpu", "processor", "core", "ghz", "memory", "ram", "gb of",
        "disk", "drive", "drives", "storage", "ssd", "hard drive",
        "battery", "power", "uptime", "operating system", "windows version",
        # symptoms
        "slow", "sluggish", "freezing", "lagging", "hot", "overheating",
        "resources", "performance", "free space", "how much space",
        "running out of space",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Get current CPU, memory, swap, disk, battery and uptime for "
                    "the user's machine. Use for questions about system health, "
                    "resources, free space, or why the computer is slow."
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    def match_score(self, text: str) -> int:
        return count_triggers(text, self._TRIGGERS)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        snapshot = system_snapshot()
        memory = snapshot["memory"]
        question = context.request_text
        detailed = asks_for_detail(question)

        # A totals reading cannot say *what* is using the memory, and answering
        # a narrower question than the one asked is still a wrong answer: asked
        # to "list top 5 things that are consuming my memory" this tool replied
        # with the total, identical to the previous turn. So when the question
        # is about what is responsible, the process table travels with the
        # totals rather than the two tools competing to answer half of it.
        # Also sampled for a detailed resource question, not only an explicit
        # "what is using it". "Get me a detailed report on memory usage" was
        # answered with three lines of totals: a detailed report on memory that
        # cannot say what the memory went to is not a detailed report.
        wants_processes = asks_what_is_consuming(question) or (
            detailed and asks_about_resources(question)
        )
        order = preferred_process_sort(question)
        processes = sample_processes(limit=10, sort_by=order) if wants_processes else []

        # Narrowed to the area asked about, when there is exactly one. The
        # model is not asked to filter; it is handed only what it needs.
        focus = snapshot_focus(question)
        reading = (
            "Live reading of the user's machine, taken just now:\n"
            + describe_snapshot(snapshot, focus)
        )
        if processes:
            reading += "\n\n" + describe_processes(processes, 10, order)

        rules = [
            "Answer using ONLY these figures, and quote them exactly.",
            "These readings replace any hardware or storage numbers stated "
            "earlier in this conversation, including your own previous answers "
            "— if an earlier reply disagrees with these, the earlier reply "
            "was wrong.",
            # The wording here was once "report the lines above and nothing
            # else", added to stop the model apologising for data it had not
            # been given. It overshot: asked "how much of my memory is in use?"
            # the assistant recited the processor, the OS, the memory and every
            # drive. Answering far more than was asked is its own wrong answer.
            "Answer the question that was actually asked, and nothing more. A "
            "question about one thing gets the lines about that one thing: "
            "asked how much memory is in use, answer about memory and say "
            "nothing about the processor, the operating system or the drives.",
            "Give the full overview (processor and operating system first, then "
            "memory and drives) only when the question is a general one about "
            "the machine, which you can tell from the reading above: when it "
            "covers one area only, that is the area asked about.",
            "Whatever you quote, the reading is complete as it stands: never "
            "invent a section, and never mention, apologise for, or draw "
            "attention to anything not listed. Naming what you were not given "
            "reads as a failure when nothing is missing.",
        ]

        if processes:
            rules.append(
                "This question asks what is responsible, so name the "
                "processes listed above with their own figures. The totals "
                "alone do not answer it. Keep them in the order given and copy "
                "the numbers exactly: the list is already sorted for this "
                "question, and re-ranking it turns a correct answer into a "
                "wrong one."
            )

        # The closing line, and the one a 3B model follows most reliably, so
        # depth and scope are decided here together. Split apart, whichever went
        # last won and the other was silently dropped — moving the phrasing
        # rule to the end once brought back uptime, swap and battery for a
        # question about a single drive.
        if detailed:
            # Detail widens the answer, it does not unscope it. Without the
            # second half here, "a detailed report on memory usage" came back
            # with the processor, the OS and every drive: the model read
            # "detailed" as "everything you have".
            # Two earlier attempts at this line failed in opposite ways.
            # "Give every figure that bears on what they asked" was read as
            # "everything you have" and returned the processor, the OS and every
            # drive. Adding "one per line, largest first" then made it strip the
            # labels and sort numerically — "1. 7.43 GB / 2. 0.35 GB / 3. 96%",
            # figures with nothing saying what they measured. What works is
            # asking for the lines it was given, unchanged.
            rules.append(
                "To close: the user asked for detail, so reproduce the lines "
                "above, each keeping its own label and its numbers exactly as "
                "written. \"Memory: 7.78 GB total, 7.00 GB in use\" stays a "
                "labelled line and never becomes a bare figure. Then list the "
                "processes in the order given, without renumbering or "
                "reordering them. Add nothing about parts of the machine the "
                "question did not touch."
            )
        else:
            # Both worked examples are here because it copies the shape it is
            # shown: given only the memory one it phrased memory answers well
            # and still answered about the drive with a bare "117.5 GB".
            rules.append(
                "To close: answer only what was asked, in one short sentence "
                "that names the figure. Asked how much memory is in use, say "
                "something like '93% of memory is in use, 0.57 GB still "
                "free', and say nothing at all about the processor, the "
                "drives, the uptime or the operating system. Asked how much "
                "space is left on a drive, say something like 'C: has 117.5 GB "
                "still free of 415.8 GB'."
            )

        display = (
            f"system: {snapshot['cpu_percent']:.0f}% CPU "
            + chr(0xB7)
            + f" {memory['available_gb']:.2f} GB free of "
            f"{memory['total_gb']:.2f} GB"
        )
        if processes:
            display += " " + chr(0xB7) + f" {len(processes)} processes sampled"

        return ToolResult(
            ok=True,
            content=reading + "\n\n" + "\n".join(rules)
            + "\n\n" + f"User's question: {question}",
            display=display,
            meta={**snapshot, "processes": [proc.__dict__ for proc in processes]},
        )


class TopProcessesTool(Tool):
    name = "top_processes"
    description = "List the processes using the most CPU and memory right now"
    risk = Risk.READ

    # Widened after a real miss. "list top 5 things that are consuming my
    # memory" matched none of these, so this tool never even became a
    # candidate and the totals-only reading answered unopposed. For a read
    # tool the failure modes are not symmetric: a false match costs one cheap
    # process sample, a miss answers a different question than the one asked.
    _TRIGGERS = (
        "what is using", "what's using", "which process", "what process",
        "top processes", "running processes", "using my cpu", "using my memory",
        "using my ram", "eating my", "hogging", "task manager", "processes",
        "consuming", "consume", "taking up", "taking my", "eating",
        "using the most", "using most", "most memory", "most ram", "most cpu",
        "top 3", "top 5", "top 10", "biggest", "worst offender",
        "what is running", "what's running", "program", "programs", "app",
        "apps", "application", "applications",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "List the processes currently consuming the most CPU and "
                    "memory. Use when the user asks what is running, what is "
                    "using resources, or what is slowing the machine down."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "How many processes to list (default 10).",
                        }
                    },
                    "required": [],
                },
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    # Without this the registry scores an unscored tool as 1 while
    # system_status returns its trigger count, so the backstop preferred the
    # totals reading on almost every resource question. Tools competing for the
    # same questions have to be scored the same way.
    def match_score(self, text: str) -> int:
        return count_triggers(text, self._TRIGGERS)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        try:
            limit = int(arguments.get("limit") or 10)
        except (TypeError, ValueError):
            limit = 10
        limit = max(1, min(limit, 25))

        order = preferred_process_sort(context.request_text)
        procs = sample_processes(limit=limit, sort_by=order)
        if not procs:
            return ToolResult.failure(
                "No process information was available. Tell the user plainly.",
                display="no process data",
            )

        # The totals travel with the list. "500 MB" means nothing without
        # knowing whether the machine has 8 GB or 64 — and a question about
        # what is consuming memory is usually also a question about how much
        # is left.
        memory = system_snapshot()["memory"]
        totals = (
            f"Machine totals for context: {memory['total_gb']:.2f} GB of memory, "
            f"{memory['used_gb']:.2f} GB in use, "
            f"{memory['available_gb']:.2f} GB still free "
            f"({memory['percent_used']:.0f}% used)."
        )

        return ToolResult(
            ok=True,
            content=(
                f"{describe_processes(procs, limit, order)}\n\n{totals}\n\n"
                "Answer using only these figures. The list is already "
                "sorted for this question: keep the rows in the order given "
                "and copy the names and numbers exactly, re-ranking nothing. "
                "Give the user the number of entries they asked for. Do not "
                "suggest ending a process unless asked.\n\n"
                f"User's question: {context.request_text}"
            ),
            display=f"sampled {len(procs)} processes",
            meta={"processes": [p.__dict__ for p in procs[:limit]], "memory": memory},
        )


class EndProcessTool(Tool):
    """Terminate a process. The first genuinely irreversible capability."""

    name = "end_process"
    description = "Terminate a running program by name or process id"
    risk = Risk.DESTRUCTIVE

    _TRIGGERS = (
        "kill", "terminate", "end process", "force quit", "force close",
        "shut down", "shutdown", "close the process", "stop the process",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Terminate a running process. Use only when the user "
                    "explicitly asks to kill, end, or force-quit a program. "
                    "Unsaved work in that program will be lost."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Process name, e.g. 'chrome' or 'spotify'.",
                        },
                        "pid": {
                            "type": "integer",
                            "description": "Exact process id, when known.",
                        },
                    },
                    "required": [],
                },
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    # ── target resolution ────────────────────────────────────────
    def _resolve(self, arguments: dict) -> tuple[ProcessInfo | None, ToolResult | None]:
        psutil = _psutil()
        pid = arguments.get("pid")
        name = str(arguments.get("name") or "").strip().lower()

        if pid is not None:
            try:
                proc = psutil.Process(int(pid))
                with proc.oneshot():
                    info = ProcessInfo(
                        pid=proc.pid,
                        name=proc.name(),
                        cpu_percent=0.0,
                        memory_mb=proc.memory_info().rss / BYTES_MB,
                        create_time=proc.create_time(),
                    )
                return info, None
            except psutil.NoSuchProcess:
                return None, ToolResult.failure(
                    f"No process with PID {pid} is running. Tell the user it was "
                    "not found; nothing was changed.",
                    display=f"no process with PID {pid}",
                )
            except (psutil.AccessDenied, ValueError):
                return None, ToolResult.failure(
                    f"PID {pid} could not be inspected (access denied). Tell the "
                    "user nothing was changed.",
                    display=f"PID {pid} inaccessible",
                )

        if not name:
            return None, ToolResult.failure(
                "The user asked to end a program but did not say which. Ask them "
                "for the program name. Nothing was changed."
            )

        matches = [
            proc for proc in sample_processes(interval=0.0)
            if name in proc.name.lower()
        ]
        if not matches:
            return None, ToolResult.failure(
                f"No running process matches '{name}'. Tell the user it is not "
                "running; nothing was changed.",
                display=f"no process matching '{name}'",
            )

        # Several matches: refusing beats guessing. Killing all of them because
        # the name was ambiguous is exactly the irreversible mistake to avoid.
        distinct = {proc.pid for proc in matches}
        if len(distinct) > 1:
            listing = ", ".join(
                f"{proc.name} (PID {proc.pid}, {proc.memory_mb:,.0f} MB)"
                for proc in sorted(matches, key=lambda p: -p.memory_mb)[:8]
            )
            return None, ToolResult.failure(
                f"{len(distinct)} processes match '{name}': {listing}. Ask the "
                "user which PID they mean. Do not guess, and make clear nothing "
                "has been ended.",
                display=f"{len(distinct)} processes match '{name}' — need a PID",
            )

        return matches[0], None

    def confirmation_prompt(self, arguments: dict) -> str | None:
        """Name the actual target, so the user approves a program rather than a
        tool call they cannot evaluate."""
        target, _refusal = self._resolve(arguments)
        if target is None:
            return None
        return (
            f"End {target.name} (PID {target.pid}, "
            f"{target.memory_mb:,.0f} MB)? Unsaved work will be lost."
        )

    def precheck(self, arguments: dict, context: ToolContext) -> ToolResult | None:
        """Refuse impossible targets before the user is asked to approve one.

        Runs ahead of confirmation, so a request to end a protected process is
        turned down outright rather than prompting for permission the tool was
        never going to honour.
        """
        target, refusal = self._resolve(arguments)
        if refusal is not None:
            return refusal
        assert target is not None

        if is_protected(target.name):
            return ToolResult.failure(
                f"'{target.name}' is a critical operating-system process.",
                display=f"refused: {target.name} is a protected system process",
                final_text=(
                    f"No — {target.name} (PID {target.pid}) is a critical "
                    "operating-system process. Ending it would crash the "
                    "machine, so it was refused. Nothing was changed."
                ),
            )

        if target.name.lower() in _SELF_CRITICAL:
            return ToolResult.failure(
                f"'{target.name}' runs the model powering this assistant.",
                display=f"refused: {target.name} runs this assistant",
                final_text=(
                    f"No — {target.name} (PID {target.pid}) is the model server "
                    "running this assistant, so ending it would cut the "
                    "conversation off. Close it yourself if you mean to."
                ),
            )

        if target.pid in {os.getpid(), os.getppid()}:
            return ToolResult.failure(
                "That process is this assistant itself.",
                display="refused: that is this assistant",
                final_text=(
                    f"No — PID {target.pid} is this assistant. Close the "
                    "terminal instead."
                ),
            )
        return None

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        psutil = _psutil()

        # Defence in depth: precheck already ran through the registry, but run()
        # must stay safe when called directly (tests, or a future caller).
        refusal = self.precheck(arguments, context)
        if refusal is not None:
            return refusal

        target, resolve_refusal = self._resolve(arguments)
        if resolve_refusal is not None:
            return resolve_refusal
        assert target is not None

        # PIDs are recycled. The target was chosen, then a human took time to
        # read a prompt — long enough for that PID to belong to something else.
        try:
            live = psutil.Process(target.pid)
            with live.oneshot():
                if (
                    live.name().lower() != target.name.lower()
                    or int(live.create_time()) != int(target.create_time)
                ):
                    return ToolResult.failure(
                        f"PID {target.pid} is no longer {target.name} — it now "
                        "belongs to a different process, so nothing was ended. "
                        "Tell the user to try again.",
                        display=f"refused: PID {target.pid} was recycled",
                    )
        except psutil.NoSuchProcess:
            return ToolResult.failure(
                f"{target.name} (PID {target.pid}) exited before it could be "
                "ended. Tell the user it is no longer running.",
                display=f"{target.name} already exited",
            )

        try:
            live.terminate()
            live.wait(timeout=3)
        except psutil.TimeoutExpired:
            return ToolResult(
                ok=True,
                content=(
                    f"A termination request was sent to {target.name} (PID "
                    f"{target.pid}) but it has not exited yet. Tell the user it "
                    "may be shutting down, or may need to be closed manually.\n\n"
                    f"User's request: {context.request_text}"
                ),
                display=f"{target.name} did not exit within 3s",
                meta={"pid": target.pid, "name": target.name, "exited": False},
            )
        except psutil.AccessDenied:
            return ToolResult.failure(
                f"Access was denied ending {target.name} (PID {target.pid}); it "
                "likely needs administrator rights. Nothing was changed.",
                display=f"access denied ending {target.name}",
            )
        except psutil.NoSuchProcess:
            return ToolResult(
                ok=True,
                content=f"{target.name} had already exited.",
                display=f"{target.name} already exited",
                meta={"pid": target.pid, "name": target.name, "exited": True},
            )

        return ToolResult(
            ok=True,
            content=(
                f"{target.name} (PID {target.pid}) was ended at the user's "
                "request. Confirm this briefly.\n\n"
                f"User's request: {context.request_text}"
            ),
            display=f"ended {target.name} (PID {target.pid})",
            meta={"pid": target.pid, "name": target.name, "exited": True},
        )


def system_tools() -> list[Tool]:
    return [SystemStatusTool(), TopProcessesTool(), EndProcessTool()]
