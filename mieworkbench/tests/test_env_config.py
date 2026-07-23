"""miewb.env single-source-of-truth contract (scripts/common.py).

Import-time behavior (hard-require, escape hatch, configured-absent) is
tested in SUBPROCESSES with a controlled environment + MIEWB_ENV_FILE,
so results never depend on this process's already-imported common module
or the developer's real miewb.env. Parser/editor behavior is tested
in-process on temp files.
"""

import os
import subprocess
import sys

import pytest

REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "scripts"))
import common  # noqa: E402


def _clean_env(**extra):
    """Environment with every MIEWB_* removed, then `extra` applied."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MIEWB_")}
    env.update(extra)
    return env


def _import_common(env, code="import common"):
    return subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0, %r); %s"
         % (os.path.join(REPO, "scripts"), code)],
        env=env, capture_output=True, text=True)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def test_load_env_file_rules(tmp_path):
    p = tmp_path / "miewb.env"
    p.write_text(
        "# comment\n"
        "\n"
        "  # indented comment\n"
        "MIEWB_FREECAD=/a/b\r\n"                      # CRLF
        " MIEWB_OPTICS_PYTHON = /with/spaces \n"      # whitespace
        "MIEWB_PVPYTHON=/first\n"
        "MIEWB_PVPYTHON=/second\n"                    # later dup wins
        "MIEWB_CENGINE=/path/with=equals\n")          # split on FIRST '='
    cfg = common.load_env_file(p)
    assert cfg["MIEWB_FREECAD"] == "/a/b"
    assert cfg["MIEWB_OPTICS_PYTHON"] == "/with/spaces"
    assert cfg["MIEWB_PVPYTHON"] == "/second"
    assert cfg["MIEWB_CENGINE"] == "/path/with=equals"


def test_load_env_file_missing_is_empty(tmp_path):
    assert common.load_env_file(tmp_path / "nope.env") == {}


def test_load_env_file_rejects_bare_line(tmp_path):
    p = tmp_path / "miewb.env"
    p.write_text("MIEWB_FREECAD=/a\nnot a kv line\n")
    with pytest.raises(ValueError) as err:
        common.load_env_file(p)
    assert "2" in str(err.value)  # names the line


# ---------------------------------------------------------------------------
# import-time resolution (subprocess)
# ---------------------------------------------------------------------------
def test_unresolved_import_fails_naming_setup(tmp_path):
    env = _clean_env(MIEWB_ENV_FILE=str(tmp_path / "absent.env"))
    r = _import_common(env)
    assert r.returncode != 0
    assert "setup_env.sh" in r.stderr
    assert "MIEWB_FREECAD" in r.stderr
    assert "UnconfiguredError" in r.stderr


def test_escape_hatch_imports_with_none(tmp_path):
    env = _clean_env(MIEWB_ENV_FILE=str(tmp_path / "absent.env"),
                     MIEWB_ALLOW_UNCONFIGURED="1")
    r = _import_common(env, code=(
        "import common; assert common.FREECAD_APPIMAGE is None; "
        "assert common.UNCONFIGURED == ('MIEWB_FREECAD', "
        "'MIEWB_OPTICS_PYTHON', 'MIEWB_PVPYTHON'); "
        "import sys\n"
        "try:\n"
        "    common.require_tool('MIEWB_FREECAD')\n"
        "except common.UnconfiguredError as e:\n"
        "    assert 'setup_env.sh' in str(e)\n"
        "else:\n"
        "    sys.exit('require_tool did not raise')"))
    assert r.returncode == 0, r.stderr


def test_env_var_beats_file(tmp_path):
    f = tmp_path / "miewb.env"
    f.write_text("MIEWB_FREECAD=/from/file\n"
                 "MIEWB_OPTICS_PYTHON=/opt/py\n"
                 "MIEWB_PVPYTHON=/opt/pv\n")
    env = _clean_env(MIEWB_ENV_FILE=str(f), MIEWB_FREECAD="/from/env")
    r = _import_common(env, code=(
        "import common; "
        "assert common.FREECAD_APPIMAGE == '/from/env', "
        "common.FREECAD_APPIMAGE"))
    assert r.returncode == 0, r.stderr


def test_exported_empty_is_configured_absent(tmp_path):
    f = tmp_path / "miewb.env"
    f.write_text("MIEWB_FREECAD=/a\nMIEWB_OPTICS_PYTHON=/b\n"
                 "MIEWB_PVPYTHON=/from/file\n")
    env = _clean_env(MIEWB_ENV_FILE=str(f), MIEWB_PVPYTHON="")
    r = _import_common(env, code=(
        "import common; assert common.PVPYTHON is None; "
        "assert common.UNCONFIGURED == ()"))
    assert r.returncode == 0, r.stderr


def test_file_empty_is_configured_absent_no_error(tmp_path):
    f = tmp_path / "miewb.env"
    f.write_text("MIEWB_FREECAD=/a\nMIEWB_OPTICS_PYTHON=/b\n"
                 "MIEWB_PVPYTHON=\n")
    env = _clean_env(MIEWB_ENV_FILE=str(f))
    r = _import_common(env, code=(
        "import common; assert common.PVPYTHON is None; "
        "assert common.UNCONFIGURED == (); "
        "assert common.FREECAD_APPIMAGE == '/a'"))
    assert r.returncode == 0, r.stderr


def test_dir_overrides_from_file(tmp_path):
    f = tmp_path / "miewb.env"
    f.write_text("MIEWB_FREECAD=/a\nMIEWB_OPTICS_PYTHON=/b\n"
                 "MIEWB_PVPYTHON=\nMIEWB_RESULTS_DIR=/elsewhere/results\n")
    env = _clean_env(MIEWB_ENV_FILE=str(f))
    r = _import_common(env, code=(
        "import common; "
        "assert str(common.RESULTS_DIR) == '/elsewhere/results'; "
        "assert str(common.GEOMETRY_DIR).endswith('/geometry')"))
    assert r.returncode == 0, r.stderr


# ---------------------------------------------------------------------------
# update_env_file
# ---------------------------------------------------------------------------
def test_update_env_file_preserves_comments_and_replaces(tmp_path):
    p = tmp_path / "miewb.env"
    p.write_text("# header comment\n"
                 "MIEWB_FREECAD=/old\n"
                 "# trailing comment\n")
    common.update_env_file({"MIEWB_FREECAD": "/new",
                            "MIEWB_PVPYTHON": ""}, p)
    text = p.read_text()
    assert "# header comment\n" in text
    assert "# trailing comment\n" in text
    assert "MIEWB_FREECAD=/new\n" in text
    assert "MIEWB_FREECAD=/old" not in text
    assert "MIEWB_PVPYTHON=\n" in text            # appended, empty value
    # round-trips through the parser
    cfg = common.load_env_file(p)
    assert cfg == {"MIEWB_FREECAD": "/new", "MIEWB_PVPYTHON": ""}


def test_update_env_file_drops_stale_duplicates(tmp_path):
    # parser is later-dup-wins, so the editor must not leave an old
    # duplicate line masking the new value
    p = tmp_path / "miewb.env"
    p.write_text("MIEWB_FREECAD=/one\nMIEWB_FREECAD=/two\n")
    common.update_env_file({"MIEWB_FREECAD": "/new"}, p)
    assert p.read_text().splitlines() == ["MIEWB_FREECAD=/new"]
    assert common.load_env_file(p) == {"MIEWB_FREECAD": "/new"}


def test_update_env_file_creates_with_header(tmp_path):
    p = tmp_path / "fresh.env"
    common.update_env_file({"MIEWB_FREECAD": "/a"}, p)
    lines = p.read_text().splitlines()
    assert lines[0].startswith("#")
    assert "MIEWB_FREECAD=/a" in lines


# ---------------------------------------------------------------------------
# template honesty
# ---------------------------------------------------------------------------
def test_example_file_lists_every_known_key():
    example = os.path.join(REPO, "miewb.env.example")
    assert os.path.exists(example)
    text = open(example).read()
    for key in common.KNOWN_ENV_KEYS:
        assert ("\n%s=" % key in text) or ("# %s=" % key in text), key
    # and the non-commented subset parses clean
    common.load_env_file(example)
