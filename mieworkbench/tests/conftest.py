import os

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "freecad: needs the real FreeCAD AppImage worker "
        "(slow; run with MIEWB_RUN_FREECAD=1)")
    config.addinivalue_line(
        "markers", "needs_gl: needs a real OpenGL context (skipped offscreen)")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("MIEWB_RUN_FREECAD") == "1":
        return
    skip = pytest.mark.skip(reason="set MIEWB_RUN_FREECAD=1 to run "
                                   "FreeCAD integration tests")
    for item in items:
        if "freecad" in item.keywords:
            item.add_marker(skip)
