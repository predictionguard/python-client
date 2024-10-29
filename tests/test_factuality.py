from predictionguard import PredictionGuard


def test_factuality_check():
    test_client = PredictionGuard()

    response = test_client.factuality.check(
        reference="The sky is blue", text="The sky is green"
    )

    assert type(response["checks"][0]["score"]) is float
