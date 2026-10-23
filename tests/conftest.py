import pytest


@pytest.fixture(autouse=True)
def no_real_credentials_in_tests(monkeypatch):
    # Engineering tests never need real credentials; real probes have separate commands.
    monkeypatch.setenv("CHAT_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("CHAT_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv("EMBEDDING_API_KEY", "test-only-not-a-real-key")
