import json

import requests
from typing import Any, Dict

from ..version import __version__


class Toxicity:
    """
    Toxicity checks the toxicity of a given text.

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

        # Perform the toxicity check.
        result = client.toxicity.check(text="This is a perfectly fine statement.")

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

    def check(self, text: str) -> Dict[str, Any]:
        """
        Creates a toxicity checking request for the Prediction Guard /toxicity API.

        :param text: The text to check for toxicity.
        """

        # Run _generate_score
        choices = self._generate_score(text)
        return choices

    def _generate_score(self, text):
        """
        Function to generate a single toxicity score.
        """

        # Make a prediction using the proxy.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload_dict = {"text": text}
        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", self.url + "/toxicity", headers=headers, data=payload, timeout=self.timeout
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
            raise ValueError("Could not check toxicity. " + err)
