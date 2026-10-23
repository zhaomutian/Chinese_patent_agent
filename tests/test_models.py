import httpx
import pytest

from patent_agent.models import Embeddings, ModelError
from patent_agent.settings import Settings


def response(data):
    return httpx.Response(
        200, json={"data": data}, request=httpx.Request("POST", "https://example.test/embeddings")
    )


def test_embedding_reorders_and_omits_dimensions(monkeypatch):
    s = Settings().model_copy(update={"embedding_dimensions": 2, "model_max_retries": 0})

    def post(url, **kwargs):
        assert "dimensions" not in kwargs["json"]
        assert kwargs["json"]["input"] == ["甲", "乙"]
        return response([{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}])

    monkeypatch.setattr(httpx, "post", post)
    assert Embeddings(s).encode(["甲", "乙"]) == [[1.0, 0.0], [0.0, 1.0]]


@pytest.mark.parametrize(
    "rows",
    [
        [{"index": 0, "embedding": [1.0]}],
        [{"index": 2, "embedding": [1.0, 0.0]}],
        [{"index": 0, "embedding": [0.0, 0.0]}],
    ],
)
def test_embedding_rejects_invalid_vectors(monkeypatch, rows):
    s = Settings().model_copy(update={"embedding_dimensions": 2, "model_max_retries": 0})
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: response(rows))
    with pytest.raises(ModelError):
        Embeddings(s).encode(["甲"])


def test_embedding_auth_error_redacted_and_not_retried(monkeypatch):
    s = Settings()
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return httpx.Response(
            401,
            json={"error": "SECRET_TOKEN"},
            request=httpx.Request("POST", "https://secret.example/embeddings"),
        )

    monkeypatch.setattr(httpx, "post", post)
    with pytest.raises(ModelError) as err:
        Embeddings(s).encode(["甲"])
    assert "401" in str(err.value)
    assert "SECRET_TOKEN" not in str(err.value) and "secret.example" not in str(err.value)
    assert len(calls) == 1
