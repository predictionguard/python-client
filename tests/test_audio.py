import os

from predictionguard import PredictionGuard


def test_audio_transcribe_success():
    test_client = PredictionGuard()

    response = test_client.audio.transcriptions.create(
        model="base",
        file="fixtures/test_audio.wav"
    )

    print(response)

    assert len(response["text"]) > 0