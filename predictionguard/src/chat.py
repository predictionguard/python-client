import re
import json
import os
import base64

import requests
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.parse
import uuid
from warnings import warn

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
            }
        ]

        result = client.chat.completions.create(
            model="Hermes-2-Pro-Llama-3-8B", messages=messages, max_completion_tokens=500
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
        messages: Union[
            str, List[
                Dict[str, Any]
            ]
        ],
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[
            Dict[str, int]
        ] = None,
        max_completion_tokens: Optional[int] = 100,
        max_tokens: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[
            Union[
                str, List[str]
            ]
        ] = None,
        stream: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        tool_choice: Optional[Union[
            str, Dict[
                str, Dict[str, str]
            ]
        ]] = None,
        tools: Optional[List[Dict[str, Union[str, Dict[str, str]]]]] = None,
        top_p: Optional[float] = 0.99,
        top_k: Optional[float] = 50,
    ) -> Dict[str, Any]:
        """
        Creates a chat request for the Prediction Guard /chat API.

        :param model: The ID(s) of the model to use.
        :param messages: The content of the call, an array of dictionaries containing a role and content.
        :param input: A dictionary containing the PII and injection arguments.
        :param output: A dictionary containing the consistency, factuality, and toxicity arguments.
        :param frequency_penalty: The frequency penalty to use.
        :param logit_bias: The logit bias to use.
        :param max_completion_tokens: The maximum amount of tokens the model should return.
        :param parallel_tool_calls: The parallel tool calls to use.
        :param presence_penalty: The presence penalty to use.
        :param stop: The completion stopping criteria.
        :param stream: Option to stream the API response
        :param temperature: The consistency of the model responses to the same prompt. The higher the more consistent.
        :param tool_choice: The tool choice to use.
        :param tools: Options to pass to the tool choice.
        :param top_p: The sampling for the model to use.
        :param top_k: The Top-K sampling for the model to use.
        :return: A dictionary containing the chat response.
        """

        # Handling max_tokens and returning deprecation message
        if max_tokens is not None:
            max_completion_tokens = max_tokens
            warn("""
            The max_tokens argument is deprecated. 
            Please use max_completion_tokens instead.
            """, DeprecationWarning, stacklevel=2
            )

        # Create a list of tuples, each containing all the parameters for
        # a call to _generate_chat
        args = (
            model,
            messages,
            input,
            output,
            frequency_penalty,
            logit_bias,
            max_completion_tokens,
            parallel_tool_calls,
            presence_penalty,
            stop,
            stream,
            temperature,
            tool_choice,
            tools,
            top_p,
            top_k
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
        frequency_penalty,
        logit_bias,
        max_completion_tokens,
        parallel_tool_calls,
        presence_penalty,
        stop,
        stream,
        temperature,
        tool_choice,
        tools,
        top_p,
        top_k,
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
                # then print out the error field in the json body, and raise an exception.
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

        # TODO: Remove `tool_choice` check when null value available in API
        if tool_choice is None:
            payload_dict = {
                "model": model,
                "messages": messages,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "max_completion_tokens": max_completion_tokens,
                "parallel_tool_calls": parallel_tool_calls,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream,
                "temperature": temperature,
                "tools": tools,
                "top_p": top_p,
                "top_k": top_k,
            }
        else:
            payload_dict = {
                "model": model,
                "messages": messages,
                "frequency_penalty": frequency_penalty,
                "logit_bias": logit_bias,
                "max_completion_tokens": max_completion_tokens,
                "parallel_tool_calls": parallel_tool_calls,
                "presence_penalty": presence_penalty,
                "stop": stop,
                "stream": stream,
                "temperature": temperature,
                "tool_choice": tool_choice,
                "tools": tools,
                "top_p": top_p,
                "top_k": top_k,
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
