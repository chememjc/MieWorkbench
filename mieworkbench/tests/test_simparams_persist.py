# ============================================================================
# test_simparams_persist.py — simulation settings travel with the project.
#
# Contract (user request, c-engine round):
#   * Simulation menu has a "Simulation Settings…" action, separate from
#     Run Pipeline, sharing the SAME ConfigMatrix widget (one source of
#     truth for the values).
#   * persist_simparams() writes the current settings into the workspace's
#     simparams.json AND repacks the open .MieWB, so the settings are
#     stored with the project; reopening the archive restores them.
#   * Bare .FCStd sessions (no archive) are a no-op, never an error.
# ============================================================================
import json
import os
import sys
import zipfile

import pytest

sys.path.insert(0, os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from mieworkbench.mainwindow import MainWindow                # noqa: E402
import miewb_tool                                             # noqa: E402


@pytest.fixture
def window(qtbot):
    w = MainWindow()
    qtbot.addWidget(w)
    return w


def test_settings_menu_action_exists(window):
    assert window.settings_action is not None
    texts = [a.text() for a in window.settings_action.parent().actions()] \
        if window.settings_action.parent() else []
    assert window.settings_action.text().replace("&", "") \
        == "Simulation Settings…"
    # separate from the run action, same shared matrix widget
    assert window.settings_action is not window.run_action


def test_persist_simparams_noop_without_archive(window):
    window.miewb_path = None
    assert window.persist_simparams() is False


def test_persist_simparams_roundtrips_into_miewb(window, tmp_path):
    # a minimal fake FCStd payload is enough for pack_miewb (stored blob)
    fcstd = tmp_path / "model.FCStd"
    with zipfile.ZipFile(fcstd, "w") as z:
        z.writestr("Document.xml", "<Document/>")
    archive = tmp_path / "model.MieWB"
    miewb_tool.pack_miewb(str(fcstd), str(archive), simparams={})

    ws = tmp_path / "ws"
    ws.mkdir()
    window.model_path = str(fcstd)
    window.miewb_path = str(archive)
    window.workspace = str(ws)

    window.config_matrix.set_values({"rays": "12345", "engine": "python"})
    assert window.persist_simparams() is True

    # workspace sidecar written (ConfigMatrix normalizes rays to float)
    side = json.loads((ws / "simparams.json").read_text())
    assert float(side.get("rays")) == 12345.0
    assert side.get("engine") == "python"

    # archive member updated
    with zipfile.ZipFile(archive) as z:
        packed = json.loads(z.read("simparams.json"))
    assert float(packed.get("rays")) == 12345.0
    assert packed.get("engine") == "python"
