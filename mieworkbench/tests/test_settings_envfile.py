"""Settings <-> miewb.env integration (core/settings.py).

The Settings wrapper's path fields must read/write miewb.env (the single
source of truth) instead of QSettings, honor exported-env-var locks, and
migrate legacy QSettings-stored paths exactly once.
"""

import os

import pytest

from mieworkbench.core import settings as settings_mod
from mieworkbench.core.settings import (
    Settings, FIELDS, _LEGACY_DEFAULTS, _MIGRATION_FLAG)


@pytest.fixture
def scratch_env(tmp_path):
    p = tmp_path / "miewb.env"
    p.write_text("# scratch header\n"
                 "MIEWB_FREECAD=/tools/FreeCAD.AppImage\n"
                 "MIEWB_OPTICS_PYTHON=/tools/optics/bin/python\n"
                 "MIEWB_PVPYTHON=\n")
    return p


@pytest.fixture
def clean_qsettings():
    """Isolate the migration flag + legacy keys around each test."""
    s = Settings.__new__(Settings)  # raw QSettings access, no migration
    from PySide6.QtCore import QSettings
    s._qs = QSettings(settings_mod.ORG_NAME, settings_mod.APP_NAME)
    saved = {k: s._qs.value(k, None)
             for k, _e, _k, _l in FIELDS}
    saved[_MIGRATION_FLAG] = s._qs.value(_MIGRATION_FLAG, None)
    yield s._qs
    for k, v in saved.items():
        if v is None:
            s._qs.remove(k)
        else:
            s._qs.setValue(k, v)
    s._qs.sync()


def test_get_reads_env_file_live(scratch_env, clean_qsettings):
    s = Settings(env_file=scratch_env)
    assert s.freecad() == "/tools/FreeCAD.AppImage"
    assert s.pvpython() is None            # configured absent
    # a live file edit is visible without a new Settings object
    scratch_env.write_text(scratch_env.read_text().replace(
        "/tools/FreeCAD.AppImage", "/other/FreeCAD.AppImage"))
    assert s.freecad() == "/other/FreeCAD.AppImage"


def test_set_writes_env_file_preserving_comments(scratch_env,
                                                 clean_qsettings):
    s = Settings(env_file=scratch_env)
    s.set("pvpython", "/pv/bin/pvpython")
    text = scratch_env.read_text()
    assert "# scratch header" in text
    assert "MIEWB_PVPYTHON=/pv/bin/pvpython" in text
    assert s.pvpython() == "/pv/bin/pvpython"
    # QSettings must NOT have stored it
    assert clean_qsettings.value("pvpython", None) in (None, "")


def test_dir_default_derives_from_repo(scratch_env, clean_qsettings):
    s = Settings(env_file=scratch_env)
    assert s.geometry_dir().endswith("geometry")
    s.set("geometry_dir", "/elsewhere/geom")
    assert s.geometry_dir() == "/elsewhere/geom"


def test_env_var_locks_field(scratch_env, clean_qsettings, monkeypatch):
    monkeypatch.setenv("MIEWB_FREECAD", "/exported/FreeCAD.AppImage")
    s = Settings(env_file=scratch_env)
    assert s.env_locked("freecad")
    assert not s.env_locked("pvpython")
    assert s.freecad() == "/exported/FreeCAD.AppImage"


def test_env_overrides_resolved_including_absent(scratch_env,
                                                 clean_qsettings):
    s = Settings(env_file=scratch_env)
    out = s.env_overrides()
    assert out["MIEWB_FREECAD"] == "/tools/FreeCAD.AppImage"
    assert out["MIEWB_PVPYTHON"] == ""     # configured absent
    assert out["MIEWB_RESULTS_DIR"].endswith("results")


def test_migration_moves_user_values_once(scratch_env, clean_qsettings):
    qs = clean_qsettings
    qs.remove(_MIGRATION_FLAG)
    # a genuine user value + a stored-but-untouched legacy default
    qs.setValue("pvpython", "/user/custom/pvpython")
    qs.setValue("freecad", _LEGACY_DEFAULTS["freecad"][0])
    qs.sync()

    s = Settings(env_file=scratch_env)
    cfg_text = scratch_env.read_text()
    # freecad: legacy default discarded, file value untouched
    assert "MIEWB_FREECAD=/tools/FreeCAD.AppImage" in cfg_text
    # pvpython: file already HAS the key (empty) -> file wins, no migrate
    assert "/user/custom/pvpython" not in cfg_text
    # legacy keys removed from QSettings, flag set
    assert qs.value("freecad", None) in (None, "")
    assert qs.value(_MIGRATION_FLAG, None)

    # second construction: no-op (idempotent)
    qs.setValue("pvpython", "/user/again")   # would migrate if flag ignored
    qs.sync()
    Settings(env_file=scratch_env)
    assert "/user/again" not in scratch_env.read_text()


def test_migration_appends_missing_key(tmp_path, clean_qsettings):
    qs = clean_qsettings
    qs.remove(_MIGRATION_FLAG)
    qs.setValue("optics_python", "/user/optics/python")
    qs.sync()
    env = tmp_path / "miewb.env"   # no MIEWB_OPTICS_PYTHON key at all
    env.write_text("MIEWB_FREECAD=/a\n")
    Settings(env_file=env)
    assert "MIEWB_OPTICS_PYTHON=/user/optics/python" in env.read_text()
