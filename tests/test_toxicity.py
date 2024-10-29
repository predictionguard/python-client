from predictionguard import PredictionGuard


def test_toxicity_check():
    test_client = PredictionGuard()

    response = test_client.toxicity.check(text="This is a perfectly fine statement.")

    assert type(response["checks"][0]["score"]) is float