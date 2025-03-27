from predictionguard import PredictionGuard


def test_translate_create():
    test_client = PredictionGuard()

    response = test_client.translate.create(
        text="The sky is blue", source_lang="eng", target_lang="fra"
    )

    assert type(response["best_score"]) is float
    assert len(response["best_translation"])