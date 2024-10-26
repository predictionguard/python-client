from predictionguard import PredictionGuard


def test_injection_check():
    test_client = PredictionGuard()

    response = test_client.injection.check(
        prompt="hi hello", detect=True
    )

    assert type(response["checks"][0]["probability"]) is float
