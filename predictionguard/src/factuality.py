import json

import requests
from typing import Any, Dict

from ..version import __version__


class Factuality:
    """Factuality checks the factuality of a given text compared to a reference.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        # Perform the factual consistency check.
        result = client.factuality.check(reference="The sky is blue.", text="The sky is green.")

        print(json.dumps(result, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def check(self, reference: str, text: str) -> Dict[str, Any]:
        """
        Creates a factuality checking request for the Prediction Guard /factuality API.

        :param reference: The reference text used to check for factual consistency.
        :param text: The text to check for factual consistency.
        """

        # Run _generate_score
        choices = self._generate_score(reference, text)
        return choices

    def _generate_score(self, reference, text):
        """
        Function to generate a single factuality score.
        """

        # Make a prediction using the proxy.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload_dict = {"reference": reference, "text": text}
        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", self.url + "/factuality", headers=headers, data=payload
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
            raise ValueError("Could not check factuality. " + err)
