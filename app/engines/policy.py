"""The offline guarantee, enforced in code rather than by convention.

This project's entire premise is that nothing leaves the machine. Ollama can
also route inference to hosted models — anything tagged ``-cloud``, such as
``gpt-oss:120b-cloud``. Those look identical to local models in `ollama list`
and are selected the same way, so a single ``/model gpt-oss:120b-cloud`` would
have sent prompts, retrieved document chunks, and injected file contents to a
remote server while the UI still said "running fully offline".

Nothing in the app selected a cloud model, but nothing stopped one either, and a
privacy guarantee that depends on the user not typing the wrong thing is not a
guarantee. So model names are validated wherever one is resolved.

Remote models are refused by default and can be enabled deliberately with
``ASSISTANT_ALLOW_REMOTE_MODELS=true`` — an informed choice, not an accident.
"""
from __future__ import annotations

# Ollama marks hosted models with a `-cloud` tag suffix.
_CLOUD_TAG_SUFFIX = "-cloud"


class RemoteModelBlocked(ValueError):
    """Raised when a hosted model is requested while offline-only is enforced."""


def is_remote_model(name: str) -> bool:
    """True if `name` refers to a hosted model rather than local weights."""
    if not name:
        return False
    _repo, _, tag = name.partition(":")
    return tag.lower().endswith(_CLOUD_TAG_SUFFIX)


def check_model(name: str, *, allow_remote: bool) -> None:
    """Raise if `name` would send data off the machine and that isn't allowed."""
    if not allow_remote and is_remote_model(name):
        raise RemoteModelBlocked(
            f"'{name}' is a cloud-hosted model — using it would send your "
            "prompts and file contents off this machine, which this assistant "
            "exists to prevent. Choose a local model, or set "
            "ASSISTANT_ALLOW_REMOTE_MODELS=true if you genuinely want that."
        )


def partition_models(names: list[str]) -> tuple[list[str], list[str]]:
    """Split a model list into (local, remote)."""
    local = [n for n in names if not is_remote_model(n)]
    remote = [n for n in names if is_remote_model(n)]
    return local, remote
