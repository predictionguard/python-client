from predictionguard import PredictionGuard


def test_pii_check():
    test_client = PredictionGuard()

    response = test_client.pii.check(
        prompt="Hello my name is John Doe. Please repeat that back to me.",
        replace=True,
        replace_method="random",
    )

    assert len(response["checks"][0]["new_prompt"]) > 0
