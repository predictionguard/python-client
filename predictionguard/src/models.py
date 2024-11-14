import requests
from typing import Any, Dict, Optional

from ..version import __version__


class Models:
    """Models lists all the models available in the Prediction Guard Platform.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        response = client.models.list()

        print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def list(self, capability: Optional[str] = "") -> Dict[str, Any]:
        """
        Creates a models list request in the Prediction Guard REST API.

        :param capability: The capability of models to list.
        :return: A dictionary containing the metadata of all the models.
        """

        # Run _check_injection
        choices = self._list_models(capability)
        return choices

    def _list_models(self, capability):
        """
        Function to list available models.
        """

        capabilities = [
            "chat-completion", "chat-with-image", "completion",
            "embedding", "embedding-with-image", "tokenize"
        ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        models_path = "/models"
        if capability != "":
            if capability not in capabilities:
                raise ValueError(
                    "If specifying a capability, please use one of the following: "
                    + ", ".join(capabilities)
                )
            else:
                models_path += "/" + capability

        response = requests.request(
            "GET", self.url + models_path, headers=headers
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
            raise ValueError("Could not check for injection. " + err)
