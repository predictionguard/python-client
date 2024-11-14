import os
import base64

from predictionguard import PredictionGuard


def test_embeddings_create_text():
    test_client = PredictionGuard()

    inputs = "Test embeddings"

    response = test_client.embeddings.create(
        model=os.environ["TEST_TEXT_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_text_batch():
    test_client = PredictionGuard()

    inputs = ["Test embeddings", "More test embeddings"]

    response = test_client.embeddings.create(
        model=os.environ["TEST_TEXT_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"]) > 1
    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float
    assert len(response["data"][1]["embedding"]) > 0
    assert type(response["data"][1]["embedding"][0]) is float


def test_embeddings_create_text_object():
    test_client = PredictionGuard()

    inputs = [{"text": "How many computers does it take to screw in a lightbulb?"}]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_tokens():
    test_client = PredictionGuard()

    inputs = [0, 8647, 6, 55720, 59725, 7, 5, 2]

    response = test_client.embeddings.create(
        model=os.environ["TEST_TEXT_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_tokens_batch():
    test_client = PredictionGuard()

    inputs = [
        [0, 8647, 6, 55720, 59725, 7, 5, 2],
        [0, 5455, 3034, 6, 55720, 59725, 7, 5, 2]
    ]

    response = test_client.embeddings.create(
        model=os.environ["TEST_TEXT_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"]) > 1
    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float
    assert len(response["data"][1]["embedding"]) > 0
    assert type(response["data"][1]["embedding"][0]) is float


def test_embeddings_create_image_file():
    test_client = PredictionGuard()

    inputs = [{"image": "fixtures/test_image1.jpeg"}]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_image_url():
    test_client = PredictionGuard()

    inputs = [
        {"image": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg"}
    ]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_image_b64():
    test_client = PredictionGuard()

    with open("fixtures/test_image1.jpeg", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    inputs = [{"image": b64_image}]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_data_uri():
    test_client = PredictionGuard()

    with open("fixtures/test_image1.jpeg", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    data_uri = "data:image/jpeg;base64," + b64_image

    inputs = [{"image": data_uri}]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float


def test_embeddings_create_both():
    test_client = PredictionGuard()

    inputs = [{"text": "Tell me a joke.", "image": "fixtures/test_image1.jpeg"}]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"])


def test_embeddings_create_text_object_batch():
    test_client = PredictionGuard()

    inputs = [{"text": "Tell me a joke."}, {"text": "Tell me a fact."}]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"]) > 1
    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float
    assert len(response["data"][1]["embedding"]) > 0
    assert type(response["data"][1]["embedding"][0]) is float


def test_embeddings_create_image_batch():
    test_client = PredictionGuard()

    inputs = [
        {"image": "fixtures/test_image1.jpeg"},
        {"image": "fixtures/test_image2.jpeg"},
    ]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"]) > 1
    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float
    assert len(response["data"][1]["embedding"]) > 0
    assert type(response["data"][1]["embedding"][0]) is float


def test_embeddings_create_both_batch():
    test_client = PredictionGuard()

    inputs = [
        {"text": "Tell me a joke.", "image": "fixtures/test_image1.jpeg"},
        {"text": "Tell me a fun fact.", "image": "fixtures/test_image2.jpeg"},
    ]

    response = test_client.embeddings.create(
        model=os.environ["TEST_MULTIMODAL_EMBEDDINGS_MODEL"], input=inputs
    )

    assert len(response["data"]) > 1
    assert len(response["data"][0]["embedding"]) > 0
    assert type(response["data"][0]["embedding"][0]) is float
    assert len(response["data"][1]["embedding"]) > 0
    assert type(response["data"][1]["embedding"][0]) is float


def test_embeddings_list_models():
    test_client = PredictionGuard()

    response = test_client.embeddings.list_models()

    assert len(response) > 0
    assert type(response[0]) is str