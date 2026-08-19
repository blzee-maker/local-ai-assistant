"""Opening, raising and closing applications by name.

The behaviour people actually want from "open Spotify" is not "launch Spotify".
It is: put Spotify in front of me. If it is already running, raise it; if it is
minimised, restore it; only start it if it genuinely is not there. One tool does
all three, because from the outside they are one request.

Closing is deliberately not killing. `end_process` already exists for that and
is DESTRUCTIVE for good reason — on Windows it is `TerminateProcess`, which
stops the program where it stands and takes any unsaved work with it. Asking a
window to close is what its X button does: the application gets to prompt about
unsaved changes, and the user answers. So `close_app` is a WRITE, and a
force-quit remains a separate, louder thing.

Two windows are never closed: anything on the protected-process list, and the
terminal this conversation is running in. The second is not hypothetical —
"close terminal" would otherwise end the conversation issuing the request.
"""
from __future__ import annotations

import time

from app import windows as win
from app.tools.base import Risk, Tool, ToolContext, ToolResult
from app.tools.system import count_triggers, is_protected

# How long to wait for a window to actually disappear after asking it to close.
# An application showing "save changes?" will still be there, and that is a
# different outcome from having closed it — reported differently (rule 5).
CLOSE_TIMEOUT_S = 4.0

# Two candidate apps within this many points are a genuine ambiguity. "visual
# studio" scores Visual Studio 2017 and Visual Studio Code identically on this
# machine; guessing between them opens the wrong program, so it asks.
AMBIGUITY_MARGIN = 6

_COMMAND_PHRASES = (
    "bring up", "pull up", "switch to", "fire up", "show me", "open up",
    "start up", "launch", "open", "start", "run", "close", "quit", "exit",
    "shut down", "shut", "bring", "pull", "show",
)

# Words that survive the verb but say nothing about which app is meant.
_NOISE = {
    "please", "for", "me", "up", "the", "a", "an", "app", "application",
    "program", "window", "now", "on", "screen", "main", "my", "to", "front",
    "foreground", "again", "it", "that", "and",
}


def app_name_from(text: str) -> str:
    """Reduce "bring visual studio up on the main screen" to "visual studio".

    Needed because the backstop calls a tool with no arguments at all, and a 3B
    model often omits them even when it does emit a call. A tool that only works
    when the model fills in a field is a tool that mostly does not work.
    """
    lowered = " ".join(text.lower().split()).strip(" ?.!,")
    for phrase in sorted(_COMMAND_PHRASES, key=len, reverse=True):
        if lowered == phrase:
            # A bare verb names nothing. Better to ask which application than
            # to go looking for one called "open".
            return ""
        if lowered.startswith(phrase + " "):
            lowered = lowered[len(phrase) + 1:]
            break
    words = [word for word in lowered.split() if word not in _NOISE]
    return " ".join(words).strip()


def _requested_name(arguments: dict, context: ToolContext) -> str:
    given = (arguments or {}).get("name") or ""
    given = str(given).strip()
    # Even when the model supplies a name it often supplies the whole sentence.
    return app_name_from(given) if given else app_name_from(context.request_text)


def app_label(window: win.WindowInfo) -> str:
    """What to call an application on screen.

    Window titles are conventionally "<document> - <application>", so the tail
    is the name a person would use: "phase1.txt - Notepad" is Notepad, and
    "Build AI assistant ... - Visual Studio Code" is Visual Studio Code.
    Falling back to the executable is a last resort — "Closed Notepad.exe"
    reads like a task manager, not an assistant.
    """
    tail = window.title.rsplit(" - ", 1)[-1].strip() if window.title else ""
    if tail and len(tail) <= 40:
        return tail
    stem = window.process[:-4] if window.process.lower().endswith(".exe") else window.process
    return stem or window.process


def _own_pids() -> set[int]:
    """This process, everything that launched it, and its console.

    Two routes to the same answer because neither alone is reliable. The
    terminal is usually an ancestor, but not when the assistant was started
    some other way; GetConsoleWindow finds the console window directly, but
    returns nothing for a detached process. Together they cover both, and this
    is what stops "close terminal" ending the conversation that asked for it.
    """
    import psutil

    pids: set[int] = set()
    try:
        process: object | None = psutil.Process()
        for _ in range(12):  # bounded: a cycle here would hang the turn
            if process is None:
                break
            pids.add(process.pid)
            process = process.parent()
    except Exception:
        pass

    console = win.console_window()
    if console:
        for window in win.visible_windows():
            if window.hwnd == console:
                pids.add(window.pid)
    return pids


def _unavailable() -> ToolResult:
    return ToolResult.failure(
        "Application control is only implemented for Windows.",
        display="not supported here",
        final_text="I can only open and close applications on Windows.",
    )


class OpenAppTool(Tool):
    """Bring an application to the front, starting it if necessary."""

    name = "open_app"
    description = "Open an application, or bring it to the front if already running"

    # WRITE: it starts programs. Consent is recorded once and remembered, which
    # is right for something the user asks for by name every time — a
    # per-invocation prompt on "open Spotify" would be noise, and noise is what
    # teaches people to approve prompts without reading them.
    risk = Risk.WRITE

    _TRIGGERS = (
        "open", "launch", "start", "bring up", "pull up", "switch to",
        "fire up", "show me", "bring", "maximise", "maximize", "restore",
        "focus", "front", "foreground",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Open an application by name, or bring it to the front if "
                    "it is already running or minimised. Use for requests like "
                    "open Spotify, launch Chrome, bring Visual Studio up."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The application name, e.g. 'Spotify'.",
                        }
                    },
                    "required": ["name"],
                },
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    def match_score(self, text: str) -> int:
        return count_triggers(text, self._TRIGGERS)

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        if not win.available():
            return _unavailable()

        wanted = _requested_name(arguments, context)
        if not wanted:
            return ToolResult.failure(
                "No application name could be determined from the request.",
                display="no app named",
                final_text="Which application would you like me to open?",
            )

        # Already on screen? Then this was never a request to launch anything.
        open_windows = win.find_windows(wanted)
        if open_windows:
            target = open_windows[0]
            label = app_label(target)
            if win.raise_window(target.hwnd):
                return ToolResult(
                    ok=True,
                    display=f"raised {target.process}",
                    final_text=f"{label} was already open — brought it to the front.",
                    meta={"action": "raised", "process": target.process},
                )
            # Windows can refuse a foreground change. Say what happened rather
            # than reporting a success that the user can see did not occur.
            return ToolResult(
                ok=True,
                display=f"{target.process} already running",
                final_text=(
                    f"{label} is already running, but Windows would not let me "
                    "bring it to the front. Its taskbar button should be "
                    "flashing."
                ),
                meta={"action": "flashed", "process": target.process},
            )

        candidates = win.find_installed_scored(wanted)
        if not candidates:
            return ToolResult.failure(
                f"No installed application matches '{wanted}'.",
                display=f"no app called {wanted}",
                final_text=f"I could not find an app called '{wanted}' on this machine.",
            )

        if (
            len(candidates) > 1
            and candidates[0][0] - candidates[1][0] < AMBIGUITY_MARGIN
        ):
            names = [app.name for _score, app in candidates[:3]]
            listed = ", ".join(names[:-1]) + " or " + names[-1]
            return ToolResult.failure(
                f"'{wanted}' matches several applications: {', '.join(names)}.",
                display="ambiguous app name",
                final_text=f"Did you mean {listed}?",
            )

        app = candidates[0][1]
        if not win.launch(app):
            return ToolResult.failure(
                f"Launching {app.name} failed.",
                display=f"could not start {app.name}",
                final_text=f"I could not start {app.name}.",
            )

        context.report_progress(f"starting {app.name}")
        appeared = win.wait_for_window(wanted)
        if appeared is None:
            # Started, but nothing on screen yet. Slow applications and
            # background ones both land here, and neither is a failure.
            return ToolResult(
                ok=True,
                display=f"started {app.name}",
                final_text=f"Started {app.name}. It has not put a window up yet.",
                meta={"action": "launched", "app": app.name},
            )

        win.raise_window(appeared.hwnd)
        return ToolResult(
            ok=True,
            display=f"opened {app.name}",
            final_text=f"Opened {app.name}.",
            meta={"action": "opened", "app": app.name},
        )


class CloseAppTool(Tool):
    """Ask an application to close, the way its X button does."""

    name = "close_app"
    description = "Close an application by name, letting it save first"

    # WRITE, not DESTRUCTIVE. WM_CLOSE is a request the application may refuse
    # or answer with a save prompt, so nothing is destroyed here. `end_process`
    # keeps the DESTRUCTIVE label because TerminateProcess genuinely does
    # destroy unsaved work.
    risk = Risk.WRITE

    _TRIGGERS = (
        "close", "quit", "exit", "shut", "hide", "dismiss", "get rid of",
        "stop",
    )

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Close a running application by name, letting it prompt "
                    "about unsaved work. Use for requests like close Spotify or "
                    "quit Chrome. Not for force-quitting a frozen program."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The application name, e.g. 'Spotify'.",
                        }
                    },
                    "required": ["name"],
                },
            },
        }

    def matches(self, text: str) -> bool:
        lowered = text.lower()
        return any(trigger in lowered for trigger in self._TRIGGERS)

    def match_score(self, text: str) -> int:
        return count_triggers(text, self._TRIGGERS)

    def precheck(self, arguments: dict, context: ToolContext) -> ToolResult | None:
        """Refuse before asking, when the answer is already no.

        The same lesson as `end_process`: prompting and *then* refusing trains
        people to wave prompts through, and the prompt itself was a lie.
        """
        if not win.available():
            return _unavailable()

        wanted = _requested_name(arguments, context)
        if not wanted:
            return None

        mine = _own_pids()
        console = win.console_window()
        for window in win.find_windows(wanted):
            if window.pid in mine or (console and window.hwnd == console):
                return ToolResult.failure(
                    "That window is the terminal this conversation is running in.",
                    display="refused: own terminal",
                    final_text=(
                        "That is the window this conversation is running in, so "
                        "I will not close it."
                    ),
                )
            if is_protected(window.process):
                return ToolResult.failure(
                    f"{window.process} is a protected system process.",
                    display=f"refused: {window.process} is protected",
                    final_text=(
                        f"{window.process} is part of Windows itself, so I will "
                        "not close it."
                    ),
                )
        return None

    def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        if not win.available():
            return _unavailable()

        wanted = _requested_name(arguments, context)
        if not wanted:
            return ToolResult.failure(
                "No application name could be determined from the request.",
                display="no app named",
                final_text="Which application would you like me to close?",
            )

        targets = win.find_windows(wanted)
        if not targets:
            return ToolResult.failure(
                f"No open window matches '{wanted}'.",
                display=f"{wanted} not running",
                final_text=f"{wanted} does not appear to be running.",
            )

        # Every top-level window of the chosen application, not just the best
        # match: an editor with three windows open is one application to the
        # person who asked, and leaving two behind is not "closed".
        pid = targets[0].pid
        process = app_label(targets[0])
        for window in targets:
            if window.pid == pid:
                win.close_window(window.hwnd)

        deadline = time.time() + CLOSE_TIMEOUT_S
        while time.time() < deadline:
            if not any(w.pid == pid for w in win.visible_windows()):
                return ToolResult(
                    ok=True,
                    display=f"closed {process}",
                    final_text=f"Closed {process}.",
                    meta={"action": "closed", "process": process},
                )
            time.sleep(0.3)

        return ToolResult(
            ok=True,
            display=f"{process} asked to close",
            final_text=(
                f"I asked {process} to close and it is still open, which usually "
                "means it is waiting on unsaved work. Check its window."
            ),
            meta={"action": "close_requested", "process": process},
        )


def app_tools() -> list[Tool]:
    return [OpenAppTool(), CloseAppTool()]
