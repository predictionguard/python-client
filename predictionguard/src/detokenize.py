import json

import requests
from typing import Any, Dict, List

from ..version import __version__


class Detokenize:
    """
    Detokenize allows you to generate text with a models internal tokenizer.

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

        response = client.detokenize.create(
            model="Qwen2.5-Coder-14B-Instruct",
            tokens=[896, 686, 77651, 419, 914, 13]
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

    def create(self, model: str, tokens: List[int]) -> Dict[str, Any]:
        """
        Creates a tokenization request in the Prediction Guard /tokenize API.

        :param model: The model to use for generating tokens.
        :param tokens: The tokens to convert into text.
        :return: A dictionary containing the text.
        """

        # Validate models
        if (
                model == "bridgetower-large-itm-mlm-itc" or
                model == "bge-m3" or
                model == "bge-reranker-v2-m3" or
                model == "multilingual-e5-large-instruct"
        ):
            raise ValueError(
                "Model %s is not supported by this endpoint." % model
            )

        # Run _create_tokens
        choices = self._create_tokens(model, tokens)
        return choices

    def _create_tokens(self, model, tokens):
        """
        Function to generate text.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload = {"model": model, "tokens": tokens}

        payload = json.dumps(payload)

        response = requests.request(
            "POST", self.url + "/detokenize", headers=headers, data=payload, timeout=self.timeout
        )

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
            raise ValueError("Could not generate text. " + err)

    def list_models(self):
        # Get the list of current models.
        headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Prediction Guard Python Client: " + __version__
                }

        response = requests.request("GET", self.url + "/models/detokenize", headers=headers, timeout=self.timeout)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list
