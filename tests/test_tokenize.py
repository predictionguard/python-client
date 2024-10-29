import os

from predictionguard import PredictionGuard


def test_tokenize_create():
    test_client = PredictionGuard()

    response = test_client.tokenize.create(
        model=os.environ["TEST_MODEL_NAME"],
        input="Tokenize this please."
    )

    assert len(response) > 0
    assert type(response["tokens"][0]["id"]) is int

