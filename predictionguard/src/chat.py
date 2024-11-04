import re
import json
import os
import base64

import requests
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.parse
import uuid

from ..version import __version__


class Chat:
    """Chat generates chat completions based on a conversation history.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provide clever and sometimes funny responses.",
            },
            {
                "role": "user",
                "content": "What's up!"
            },
            {
                "role": "assistant",
                "content": "Well, technically vertically out from the center of the earth."
            },
            {
                "role": "user",
                "content": "Haha. Good one."
            },
        ]

        result = client.chat.completions.create(
            model="Hermes-2-Pro-Llama-3-8B", messages=messages, max_tokens=500
        )

        print(json.dumps(result, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

        self.completions: ChatCompletions = ChatCompletions(self.api_key, self.url)


class ChatCompletions:
    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
        self,
        model: str,
        messages: Union[str, List[Dict[str, Any]]],
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.99,
        top_k: Optional[float] = 50,
        stream: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Creates a chat request for the Prediction Guard /chat API.

        :param model: The ID(s) of the model to use.
        :param messages: The content of the call, an array of dictionaries containing a role and content.
        :param input: A dictionary containing the PII and injection arguments.
        :param output: A dictionary containing the consistency, factuality, and toxicity arguments.
        :param max_tokens: The maximum amount of tokens the model should return.
        :param temperature: The consistency of the model responses to the same prompt. The higher the more consistent.
        :param top_p: The sampling for the model to use.
        :param top_k: The Top-K sampling for the model to use.
        :param stream: Option to stream the API response
        :return: A dictionary containing the chat response.
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _generate_chat
        args = (
            model,
            messages,
            input,
            output,
            max_tokens,
            temperature,
            top_p,
            top_k,
            stream,
        )

        # Run _generate_chat
        choices = self._generate_chat(*args)

        return choices

    def _generate_chat(
        self,
        model,
        messages,
        input,
        output,
        max_tokens,
        temperature,
        top_p,
        top_k,
        stream,
    ):
        """
        Function to generate a single chat response.
        """

        def return_dict(url, headers, payload):
            response = requests.request(
                "POST", url + "/chat/completions", headers=headers, data=payload
            )
            # If the request was successful, print the proxies.
            if response.status_code == 200:
                ret = response.json()
                return ret
            elif response.status_code == 429:
                raise ValueError(
                    "Could not connect to Prediction Guard API. "
                    "Too many requests, rate limit or quota exceeded."
            )
            else:
                # Check if there is a json body in the response. Read that in,
                # print out the error field in the json body, and raise an exception.
                err = ""
                try:
                    err = response.json()["error"]
                except Exception:
                    pass
                raise ValueError("Could not make prediction. " + err)

        def stream_generator(url, headers, payload, stream):
            with requests.post(
                url + "/chat/completions",
                headers=headers,
                data=payload,
                stream=stream,
            ) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")
                        formatted_return = (
                            "{" + (decoded_line.replace("data", '"data"', 1)) + "}"
                        )
                        try:
                            dict_return = json.loads(formatted_return)
                        except json.decoder.JSONDecodeError:
                            pass
                        else:
                            try:
                                dict_return["data"]["choices"][0]["delta"]["content"]
                            except KeyError:
                                pass
                            else:
                                yield dict_return

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        if type(messages) is list:
            for message in messages:
                if type(message["content"]) is list:
                    for entry in message["content"]:
                        if entry["type"] == "image_url":
                            image_data = entry["image_url"]["url"]
                            if stream:
                                raise ValueError(
                                    "Streaming is not currently supported when using vision."
                                )
                            else:
                                image_url_check = urllib.parse.urlparse(image_data)
                                data_uri_pattern = re.compile(
                                    r'^data:([a-zA-Z0-9!#$&-^_]+/[a-zA-Z0-9!#$&-^_]+)?(;base64)?,.*$'
                                )

                                if os.path.exists(image_data):
                                    with open(image_data, "rb") as image_file:
                                        image_input = base64.b64encode(
                                            image_file.read()
                                        ).decode("utf-8")

                                    image_data_uri = "data:image/jpeg;base64," + image_input

                                elif re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", image_data):
                                    if (
                                        base64.b64encode(
                                            base64.b64decode(image_data)
                                        ).decode("utf-8")
                                        == image_data
                                    ):
                                        image_input = image_data
                                        image_data_uri = "data:image/jpeg;base64," + image_input

                                elif image_url_check.scheme in (
                                    "http",
                                    "https",
                                    "ftp",
                                ):
                                    temp_image = uuid.uuid4().hex + ".jpg"
                                    urllib.request.urlretrieve(image_data, temp_image)
                                    with open(temp_image, "rb") as image_file:
                                        image_input = base64.b64encode(
                                            image_file.read()
                                        ).decode("utf-8")
                                    os.remove(temp_image)
                                    image_data_uri = "data:image/jpeg;base64," + image_input

                                elif data_uri_pattern.match(image_data):
                                    image_data_uri = image_data

                                else:
                                    raise ValueError(
                                        "Please enter a valid base64 encoded image, image file, image URL, or data URI."
                                    )

                                entry["image_url"]["url"] = image_data_uri
                        elif entry["type"] == "text":
                            continue

        payload_dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stream": stream,
        }

        if input:
            payload_dict["input"] = input
        if output:
            if stream:
                raise ValueError(
                    "Factuality and toxicity checks are not supported when streaming is enabled."
                )
            else:
                payload_dict["output"] = output

        payload = json.dumps(payload_dict)

        if stream:
            return stream_generator(self.url, headers, payload, stream)

        else:
            return return_dict(self.url, headers, payload)

    def list_models(self, capability: Optional[str] = "chat-completion") -> List[str]:
        # Get the list of current models.
        headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Prediction Guard Python Client: " + __version__
                }

        if capability != "chat-completion" and capability != "chat-with-image":
            raise ValueError(
                "Please enter a valid model type (chat-completion or chat-with-image)."
            )
        else:
            model_path = "/models/" + capability

        response = requests.request("GET", self.url + model_path, headers=headers)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list