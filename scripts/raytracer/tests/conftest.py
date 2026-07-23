import os

# Tests must import on an unconfigured machine (tool-needing tests skip
# when the resolved tool is None).
os.environ.setdefault("MIEWB_ALLOW_UNCONFIGURED", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running e2e test (full trace + gather)")
