from jedi.plugins import pytest
from uaclient.api.u.pro.security.fix.cve.plan.v1 import endpoint

from predictionguard import PredictionGuard


def test_models_list():
    test_client = PredictionGuard()

    response = test_client.models.list()

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_completion_chat():
    test_client = PredictionGuard()

    response = test_client.models.list(
        endpoint="completion-chat"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_completion():
    test_client = PredictionGuard()

    response = test_client.models.list(
        endpoint="completion"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_vision():
    test_client = PredictionGuard()

    response = test_client.models.list(
        endpoint="vision"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_text_embeddings():
    test_client = PredictionGuard()

    response = test_client.models.list(
        endpoint="text-embeddings"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_image_embeddings():
    test_client = PredictionGuard()

    response = test_client.models.list(
        endpoint="image-embeddings"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_tokenize():
    test_client = PredictionGuard()

    response = test_client.models.list(
        endpoint="tokenize"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_fail():
    test_client = PredictionGuard()

    models_error = ""

    with pytest.raises(ValueError, match=models_error):
        test_client.models.list(
            endpoint="fail"
        )