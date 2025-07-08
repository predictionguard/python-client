from predictionguard import PredictionGuard


def test_audio_transcribe_success():
    test_client = PredictionGuard()

    response = test_client.audio.transcriptions.create(
        model="base",
        file="fixtures/test_audio.wav"
    )

    assert len(response["text"]) > 0

def test_audio_transcribe_timestamps_success():
    test_client = PredictionGuard()

    response = test_client.audio.transcriptions.create(
        model="base",
        file="fixtures/test_audio.wav",
        timestamp_granularities=["word", "segment"],
        response_format="verbose_json"
    )

    assert len(response["text"]) > 0
    assert len(response["segments"]) > 0
    assert len(response["segments"][0]["text"]) > 0
    assert len(response["words"]) > 0
    assert len(response["words"][0]["text"]) > 0

def test_audio_transcribe_diarization_success():
    test_client = PredictionGuard()

    response = test_client.audio.transcriptions.create(
        model="base",
        file="fixtures/test_audio.wav",
        diarization=True,
        response_format="verbose_json"
    )

    assert len(response["text"]) > 0
    assert len(response["segments"]) > 0
    assert len(response["segments"][0]["text"]) > 0

def test_audio_transcribe_diarization_timestamps_success():
    test_client = PredictionGuard()

    response = test_client.audio.transcriptions.create(
        model="base",
        file="fixtures/test_audio.wav",
        diarization=True,
        timestamp_granularities=["word"],
        response_format="verbose_json"
    )

    assert len(response["text"]) > 0
    assert len(response["words"]) > 0
    assert len(response["words"][0]["text"]) > 0