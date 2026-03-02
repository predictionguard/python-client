import os
import base64

import pytest

from predictionguard import PredictionGuard


def test_responses_create():
    test_client = PredictionGuard()

    response = test_client.responses.create(
        model=os.environ["TEST_RESPONSES_MODEL"],
        input=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": "Tell me a joke."},
        ],
    )

    assert len(response["output"][0]["content"]) > 0


def test_responses_create_string():
    test_client = PredictionGuard()

    response = test_client.responses.create(
        model=os.environ["TEST_RESPONSES_MODEL"],
        input="Tell me a joke"
    )

    assert len(response["output"][0]["content"]) > 0


# def test_responses_create_stream():
#     test_client = PredictionGuard()
#
#     response_list = []
#     for res in test_client.responses.create(
#         model=os.environ["TEST_RESPONSES_MODEL"],
#         input=[
#             {"role": "system", "content": "You are a helpful chatbot."},
#             {"role": "user", "content": "Tell me a joke."},
#         ],
#         stream=True,
#     ):
#         response_list.append(res)
#
#     assert len(response_list) > 1


def test_responses_create_vision_image_file():
    test_client = PredictionGuard()

    response = test_client.responses.create(
        model=os.environ["TEST_VISION_MODEL"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "fixtures/test_image1.jpeg",
                    },
                ],
            }
        ],
    )

    assert len(response["output"][0]["content"]) > 0


def test_responses_create_vision_image_url():
    test_client = PredictionGuard()

    response = test_client.responses.create(
        model=os.environ["TEST_VISION_MODEL"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://farm4.staticflickr.com/3300/3497460990_11dfb95dd1_z.jpg"
                    },
                ],
            }
        ],
    )

    assert len(response["output"][0]["content"]) > 0


def test_responses_create_vision_image_b64():
    test_client = PredictionGuard()

    with open("fixtures/test_image1.jpeg", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    response = test_client.responses.create(
        model=os.environ["TEST_VISION_MODEL"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {"type": "input_image", "image_url": b64_image},
                ],
            }
        ],
    )

    assert len(response["output"][0]["content"]) > 0


def test_responses_create_vision_data_uri():
    test_client = PredictionGuard()

    with open("fixtures/test_image1.jpeg", "rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    data_uri = "data:image/jpeg;base64," + b64_image

    response = test_client.responses.create(
        model=os.environ["TEST_VISION_MODEL"],
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What is in this image?"},
                    {"type": "input_image", "image_url": data_uri},
                ],
            }
        ],
    )

    assert len(response["output"][0]["content"]) > 0


def test_responses_create_vision_stream_fail():
    test_client = PredictionGuard()

    streaming_error = "Streaming is not currently supported when using vision."

    response_list = []
    with pytest.raises(ValueError, match=streaming_error):
        for res in test_client.responses.create(
            model=os.environ["TEST_VISION_MODEL"],
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What is in this image?"},
                        {
                            "type": "input_image",
                            "image_url": "fixtures/test_image1.jpeg",
                        },
                    ],
                }
            ],
            stream=True,
        ):
            response_list.append(res)


def test_responses_create_tool_call():
    test_client = PredictionGuard()

    response = test_client.responses.create(
        model=os.environ["TEST_RESPONSES_MODEL"],
        input=[
            {"role": "system", "content": "You are a helpful chatbot."},
            {"role": "user", "content": "Tell me a joke."},
        ],

    )

    assert len(response["output"][0]["content"]) > 0


def test_responses_list_models():
    test_client = PredictionGuard()

    response = test_client.responses.list_models()

    assert len(response) > 0
    assert type(response[0]) is str
