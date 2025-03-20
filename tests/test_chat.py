import os
import base64

import pytest

from predictionguard import PredictionGuard


def test_chat_completions_create():
    test_client = PredictionGuard()

    response = test_client.chat.completions.create(
        model=os.environ["TEST_MODEL_NAME"],
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_create_string():
    test_client = PredictionGuard()

    response = test_client.chat.completions.create(
        model=os.environ["TEST_MODEL_NAME"],
        messages="Tell me a joke"
    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_create_stream():
    test_client = PredictionGuard()

    response_list = []
    for res in test_client.chat.completions.create(
        model=os.environ["TEST_MODEL_NAME"],
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": "Tell me a joke."},
        ],
        stream=True,
    ):
        response_list.append(res)

    assert len(response_list) > 1


def test_chat_completions_create_stream_output_fail():
    test_client = PredictionGuard()

    streaming_error = "Factuality and toxicity checks are not supported when streaming is enabled.".replace(
        "\n", ""
    )

    response_list = []
    with pytest.raises(ValueError, match=streaming_error):
        for res in test_client.chat.completions.create(
            model=os.environ["TEST_MODEL_NAME"],
            messages=[
                {"role": "system", "content": "You are a helpful chatbot."},
                {"role": "user", "content": "Tell me a joke."},
            ],
            stream=True,
            output={"toxicity": True},
        ):
            response_list.append(res)


def test_chat_completions_create_vision_image_file():
    test_client = PredictionGuard()

    response = test_client.chat.completions.create(
        model=os.environ["TEST_VISION_MODEL"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "fixtures/test_image1.jpeg"},
                    },
                ],
            }
        ],
    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_create_vision_image_url():
    test_client = PredictionGuard()

    response = test_client.chat.completions.create(
        model=os.environ["TEST_VISION_MODEL"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg"
                        },
                    },
                ],
            }
        ],
    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_create_vision_image_b64():
    test_client = PredictionGuard()

    with open("fixtures/test_image1.jpeg", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = test_client.chat.completions.create(
        model=os.environ["TEST_VISION_MODEL"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": b64_image}},
                ],
            }
        ],
    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_create_vision_data_uri():
    test_client = PredictionGuard()

    with open("fixtures/test_image1.jpeg", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    data_uri = "data:image/jpeg;base64," + b64_image

    response = test_client.chat.completions.create(
        model=os.environ["TEST_VISION_MODEL"],
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_create_vision_stream_fail():
    test_client = PredictionGuard()

    streaming_error = "Streaming is not currently supported when using vision."

    response_list = []
    with pytest.raises(ValueError, match=streaming_error):
        for res in test_client.chat.completions.create(
            model=os.environ["TEST_VISION_MODEL"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "fixtures/test_image1.jpeg"},
                        },
                    ],
                }
            ],
            stream=True,
        ):
            response_list.append(res)


def test_chat_completions_create_tool_call():
    test_client = PredictionGuard()

    response = test_client.chat.completions.create(
        model=os.environ["TEST_MODEL_NAME"],
        messages=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": "Tell me a joke."},
        ],

    )

    assert len(response["choices"][0]["message"]["content"]) > 0


def test_chat_completions_list_models():
    test_client = PredictionGuard()

    response = test_client.chat.completions.list_models()

    assert len(response) > 0
    assert type(response[0]) is str
