def pytest_configure(config):
    config.addinivalue_line("markers", "slow: end-to-end test (~2 min on one CPU)")
