def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: long-running e2e test (full trace + gather)")
