import os

from predictionguard import PredictionGuard


def test_detokenize_create():
    test_client = PredictionGuard()

    response = test_client.detokenize.create(
        model=os.environ["TEST_CHAT_MODEL"],
        tokens=[896, 686, 77651, 419, 914, 13]
    )

    assert len(response) > 0
    assert len(response["text"]) > 0
    assert type(response["text"]) is str


def test_detokenize_list():
    test_client = PredictionGuard()

    response = test_client.detokenize.list_models()

    assert len(response) > 0
    assert type(response[0]) is str
