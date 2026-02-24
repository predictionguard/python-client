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


class Responses:
    """
    Responses allows for the usage of LLMs intended for agentic usages.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token and url as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"
        os.environ["PREDICTIONGUARD_URL"] = "<url>"

        # Or set your Prediction Guard token and url when initializing the PredictionGuard class.
        client = PredictionGuard(
            api_key="<api_key>",
            url="<url>"
        )

        input = [
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

        result = client.responses.create(
            model="gpt-oss-120b",
            input=input
        )

        print(json.dumps(
            response,
            sort_keys=True,
            indent=4,
            separators=(",", ": ")
        ))
    """

    def __init__(self, api_key, url, timeout):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout

    def create(
        self,
        model: str,
        input: Union[
            str, List[
                Dict[str, Any]
            ]
        ],
        max_output_tokens: Optional[int] = None,
        max_tool_calls: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        reasoning: Optional[Dict[str, str]] = None,
        safeguards: Optional[Dict[str, Any]] = None,
        stream: Optional[bool] = False,
        temperature: Optional[float] = None,
        tool_choice: Optional[Union[
            str, Dict[
                str, Dict[str, str]
            ]
        ]] = None,
        tools: Optional[List[Dict[str, Union[str, Dict[str, str]]]]] = None,
        top_p: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Creates a chat request for the Prediction Guard /chat API.

        :param model: The ID(s) of the model to use.
        :param input: The content of the call, an array of dictionaries containing a role and content.
        :param max_output_tokens: The maximum amount of tokens the model should return.
        :param max_tool_calls: The maximum amount of tool calls the model can perform.
        :param parallel_tool_calls: The parallel tool calls to use.
        :param reasoning: How much effort for model to use for reasoning. Only supported by reasoning models.
        :param safeguards: A dictionary containing the PII, injection, factuality, and toxicity arguments.
        :param stream: Option to stream the API response
        :param temperature: The consistency of the model responses to the same prompt. The higher it is set, the more consistent.
        :param tool_choice: The tool choice to use.
        :param tools: Options to pass to the tool choice.
        :param top_p: The sampling for the model to use.
        :return: A dictionary containing the responses response.
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _generate_response
        args = (
            model,
            input,
            max_output_tokens,
            max_tool_calls,
            parallel_tool_calls,
            reasoning,
            safeguards,
            stream,
            temperature,
            tool_choice,
            tools,
            top_p,
        )

        # Run _generate_response
        output = self._generate_response(*args)

        return output

    def _generate_response(
        self,
        model,
        input,
        max_output_tokens,
        max_tool_calls,
        parallel_tool_calls,
        reasoning,
        safeguards,
        stream,
        temperature,
        tool_choice,
        tools,
        top_p,
    ):
        """
        Function to generate a single responses response.
        """

        def return_dict(url, headers, payload, timeout):
            response = requests.request(
                "POST", url + "/responses", headers=headers, data=payload, timeout=timeout
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
                # Check if there is a JSON body in the response. Read that in,
                # then print out the error field in the JSON body, and raise an exception.
                err = ""
                try:
                    err = response.json()["error"]
                except Exception:
                    pass
                raise ValueError("Could not make prediction. " + err)

        def stream_generator(url, headers, payload, stream, timeout):
            with requests.post(
                url + "/responses",
                headers=headers,
                data=payload,
                stream=stream,
                timeout=timeout,
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

        if type(input) is list:
            for inpt in input:
                if type(inpt["content"]) is list:
                    for entry in inpt["content"]:
                        if entry["type"] == "input_image":
                            image_data = entry["image_url"]
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

                                entry["image_url"] = image_data_uri
                        elif entry["type"] == "input_text":
                            continue

        payload_dict = {
            "model": model,
            "input": input,
            "max_output_tokens": max_output_tokens,
            "max_tool_calls": max_tool_calls,
            "parallel_tool_calls": parallel_tool_calls,
            "reasoning": reasoning,
            "safeguards": safeguards,
            "stream": stream,
            "temperature": temperature,
            "tool_choice": tool_choice,
            "tools": tools,
            "top_p": top_p,
        }

        payload = json.dumps(payload_dict)

        if stream:
            return stream_generator(self.url, headers, payload, stream, self.timeout)

        else:
            return return_dict(self.url, headers, payload, self.timeout)

    def list_models(self, capability: Optional[str] = "responses") -> List[str]:
        # Get the list of current models.
        headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Prediction Guard Python Client: " + __version__
                }

        if capability != "responses" and capability != "responses-with-image":
            raise ValueError(
                "Please enter a valid model type (responses or responses-with-image)."
            )
        else:
            model_path = "/models/" + capability

        response = requests.request("GET", self.url + model_path, headers=headers, timeout=self.timeout)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list
