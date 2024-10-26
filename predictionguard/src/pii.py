import json

import requests
from typing import Any, Dict, Optional

from ..version import __version__


class Pii:
    """Pii replaces personal information such as names, SSNs, and emails in a given text.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        response = client.pii.check(
            prompt="Hello, my name is John Doe and my SSN is 111-22-3333.",
            replace=True,
            replace_method="mask",
        )

        print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def check(
        self, prompt: str, replace: bool, replace_method: Optional[str] = "random"
    ) -> Dict[str, Any]:
        """Creates a PII checking request for the Prediction Guard /PII API.

        :param text: The text to check for PII.
        :param replace: Whether to replace PII if it is present.
        :param replace_method: Method to replace PII if it is present.
        """

        # Run _check_pii
        choices = self._check_pii(prompt, replace, replace_method)
        return choices

    def _check_pii(self, prompt, replace, replace_method):
        """Function to check for PII."""

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload_dict = {
            "prompt": prompt,
            "replace": replace,
            "replace_method": replace_method,
        }

        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", self.url + "/PII", headers=headers, data=payload
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
            raise ValueError("Could not check PII. " + err)
