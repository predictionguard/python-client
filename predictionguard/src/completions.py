import json

import requests
from typing import Any, Dict, List, Optional, Union

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
        max_tokens: Optional[int] = 100,
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
        :param max_tokens: The maximum number of tokens to generate in the completion(s).
        :param temperature: The sampling temperature to use.
        :param top_p: The nucleus sampling probability to use.
        :param top_k: The Top-K sampling for the model to use.
        :return: A dictionary containing the completion response.
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _generate_completion
        args = (model, prompt, input, output, max_tokens, temperature, top_p, top_k)

        # Run _generate_completion
        choices = self._generate_completion(*args)

        return choices

    def _generate_completion(
        self, model, prompt,
        input, output, max_tokens,
        temperature, top_p, top_k
    ):
        """
        Function to generate a single completion.
        """

        # Make a prediction using the proxy.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload_dict = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        if input:
            payload_dict["input"] = input
        if output:
            payload_dict["output"] = output
        payload = json.dumps(payload_dict)

        response = requests.request(
            "POST", self.url + "/completions", headers=headers, data=payload
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
            # Check if there is a json body in the response. Read whether the API response should be streamed in,
            # print out the error field in the json body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except Exception:
                pass
            raise ValueError("Could not make prediction. " + err)

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
