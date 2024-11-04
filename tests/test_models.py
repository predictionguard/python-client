from predictionguard import PredictionGuard


def test_models_list():
    test_client = PredictionGuard()

    response = test_client.models.list()

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_chat_completion():
    test_client = PredictionGuard()

    response = test_client.models.list(
        capability="chat-completion"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_chat_with_image():
    test_client = PredictionGuard()

    response = test_client.models.list(
        capability="chat-with-image"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_completion():
    test_client = PredictionGuard()

    response = test_client.models.list(
        capability="completion"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_embedding():
    test_client = PredictionGuard()

    response = test_client.models.list(
        capability="embedding"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_embedding_with_image():
    test_client = PredictionGuard()

    response = test_client.models.list(
        capability="embedding-with-image"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str


def test_models_list_tokenize():
    test_client = PredictionGuard()

    response = test_client.models.list(
        capability="tokenize"
    )

    assert len(response["data"]) > 0
    assert type(response["data"][0]["id"]) is str
