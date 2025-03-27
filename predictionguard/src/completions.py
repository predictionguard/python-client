import json

import requests
from typing import Any, Dict, List, Optional, Union
from warnings import warn

from ..version import __version__


class Completions:
    """
    OpenAI-compatible completion API
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
        self,
        model: str,
        prompt: Union[str, List[str]],
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        echo: Optional[bool] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[Dict[str, int]] = None,
        max_completion_tokens: Optional[int] = 100,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        stream: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_p: Optional[float] = 0.99,
        top_k: Optional[int] = 50
    ) -> Dict[str, Any]:
        """
        Creates a completion request for the Prediction Guard /completions API.

        :param model: The ID(s) of the model to use.
        :param prompt: The prompt(s) to generate completions for.
        :param input: A dictionary containing the PII and injection arguments.
        :param output: A dictionary containing the consistency, factuality, and toxicity arguments.
        :param echo: A boolean indicating whether to echo the prompt(s) to the output.
        :param frequency_penalty: The frequency penalty to use.
        :param logit_bias: The logit bias to use.
        :param max_completion_tokens: The maximum number of tokens to generate in the completion(s).
        :param presence_penalty: The presence penalty to use.
        :param stop: The completion stopping criteria.
        :param stream: The stream to use for HTTP requests.
        :param temperature: The sampling temperature to use.
        :param top_p: The nucleus sampling probability to use.
        :param top_k: The Top-K sampling for the model to use.
        :return: A dictionary containing the completion response.
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
        # a call to _generate_completion
        args = (
            model,
            prompt,
            input,
            output,
            echo,
            frequency_penalty,
            logit_bias,
            max_completion_tokens,
            presence_penalty,
            stop,
            stream,
            temperature,
            top_p,
            top_k
        )

        # Run _generate_completion
        choices = self._generate_completion(*args)

        return choices

    def _generate_completion(
        self,
        model,
        prompt,
        input,
        output,
        echo,
        frequency_penalty,
        logit_bias,
        max_completion_tokens,
        presence_penalty,
        stop,
        stream,
        temperature,
        top_p,
        top_k
    ):
        """
        Function to generate a single completion.
        """

        def return_dict(url, headers, payload):
            response = requests.request(
                "POST", url + "/completions", headers=headers, data=payload
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
                url + "/completions",
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
                                dict_return["data"]["choices"][0]["text"]
                            except KeyError:
                                pass
                            else:
                                yield dict_return

        # Make a prediction using the proxy.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload_dict = {
            "model": model,
            "prompt": prompt,
            "echo": echo,
            "frequency_penalty": frequency_penalty,
            "logit_bias": logit_bias,
            "max_completion_tokens": max_completion_tokens,
            "presence_penalty": presence_penalty,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
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

    def list_models(self) -> List[str]:
        # Get the list of current models.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        response = requests.request("GET", self.url + "/models/completion", headers=headers)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list
