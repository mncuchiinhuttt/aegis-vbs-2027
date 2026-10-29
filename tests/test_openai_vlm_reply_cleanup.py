"""Gateway-appended text must be removable from VLM replies without touching clean ones."""

import importlib
import os
import sys
from types import SimpleNamespace

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "inference-code"))

FOOTER = "\n\n> Thanks for using ExampleRouter! Report problems at https://router.example/issues"


def _response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _load(monkeypatch, regex):
    if regex is None:
        monkeypatch.delenv("OPENAI_VLM_STRIP_REGEX", raising=False)
    else:
        monkeypatch.setenv("OPENAI_VLM_STRIP_REGEX", regex)
    import models.openai_vlm as module

    return importlib.reload(module)


def test_configured_footer_is_removed_so_json_replies_stay_parseable(monkeypatch):
    module = _load(monkeypatch, r"\s*>\s*Thanks for using ExampleRouter!.*")

    reply = module._reply_text(_response('{"found": true, "answer": "red"}' + FOOTER))

    assert reply == '{"found": true, "answer": "red"}'


def test_replies_are_untouched_when_no_pattern_is_configured(monkeypatch):
    module = _load(monkeypatch, None)

    assert module._reply_text(_response("  0.8" + FOOTER)) == "  0.8" + FOOTER
    assert module._reply_text(_response(None)) is None
