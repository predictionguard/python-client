import os

from predictionguard import PredictionGuard


def test_completions_create():
    test_client = PredictionGuard()

    response = test_client.completions.create(
        model=os.environ["TEST_MODEL_NAME"], prompt="Tell me a joke"
    )

    assert len(response["choices"][0]["text"]) > 0


def test_completions_create_batch():
    test_client = PredictionGuard()

    response = test_client.completions.create(
        model=os.environ["TEST_MODEL_NAME"],
        prompt=["Tell me a joke.", "Tell me a cool fact."],
    )

    assert len(response["choices"]) > 1
    assert len(response["choices"][0]["text"]) > 0
    assert len(response["choices"][1]["text"]) > 0


def test_completions_list_models():
    test_client = PredictionGuard()

    response = test_client.completions.list_models()

    assert len(response) > 0
    assert type(response[0]) is str
