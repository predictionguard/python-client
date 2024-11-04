import json

import requests
from typing import Any, Dict

from ..version import __version__


class Tokenize:
    """Tokenize allows you to generate tokens with a models internal tokenizer.

        Usage::

            import os
            import json

            from predictionguard import PredictionGuard

            # Set your Prediction Guard token as an environmental variable.
            os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

            client = PredictionGuard()

            response = client.tokenize.create(
                model="Hermes-3-Llama-3.1-8B",
                input="Tokenize this example."
            )

            print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
        """


    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(self, model: str, input: str) -> Dict[str, Any]:
        """
        Creates a prompt injection check request in the Prediction Guard /injection API.

        :param model: The model to use for generating tokens.
        :param input: The text to convert into tokens.
        :return: A dictionary containing the tokens and token metadata.
        """

        # Validate models
        if model == "llava-1.5-7b-hf" or model == "bridgetower-large-itm-mlm-itc":
            raise ValueError(
                "Model %s is not supported by this endpoint." % model
            )

        # Run _check_injection
        choices = self._create_tokens(model, input)
        return choices

    def _create_tokens(self, model, input):
        """
        Function to generate tokens.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload = {"model": model, "input": input}

        payload = json.dumps(payload)

        response = requests.request(
            "POST", self.url + "/tokenize", headers=headers, data=payload
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
            raise ValueError("Could not generate tokens. " + err)

    def list_models(self):
        # Get the list of current models.
        headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Prediction Guard Python Client: " + __version__
                }

        response = requests.request("GET", self.url + "/models/tokenize", headers=headers)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list
