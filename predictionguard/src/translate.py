import json

import requests
from typing import Any, Dict, Optional

from ..version import __version__


class Translate:
    # UNCOMMENT WHEN DEPRECATED
    # """No longer supported.
    # """
    #
    # def __init__(self, api_key, url):
    #     self.api_key = api_key
    #     self.url = url
    #
    # def create(
    #         self,
    #         text: Optional[str],
    #         source_lang: Optional[str],
    #         target_lang: Optional[str],
    #         use_third_party_engine: Optional[bool] = False
    #     ) -> Dict[str, Any]:
    #     """
    #     No longer supported
    #     """
    #
    #     raise ValueError(
    #         "The translate functionality is no longer supported."
    #     )
    """Translate converts text from one language to another.

    Usage::

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        response = client.translate.create(
            text="The sky is blue.",
            source_lang="eng",
            target_lang="fra",
            use_third_party_engine=True
        )

        print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    # REMOVE BELOW HERE FOR DEPRECATION
    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
            self,
            text: str,
            source_lang: str,
            target_lang: str,
            use_third_party_engine: Optional[bool] = False
        ) -> Dict[str, Any]:
        """
        Creates a translate request to the Prediction Guard /translate API.

        :param text: The text to be translated.
        :param source_lang: The language the text is currently in.
        :param target_lang: The language the text will be translated to.
        :param use_third_party_engine: A boolean for enabling translations with third party APIs.
        :result: A dictionary containing the translate response.
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _generate_translation
        args = (text, source_lang, target_lang, use_third_party_engine)

        # Run _generate_translation
        choices = self._generate_translation(*args)
        return choices

    def _generate_translation(self, text, source_lang, target_lang, use_third_party_engine):
        """
        Function to generate a translation response.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload_dict = {
            "text": text,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "use_third_party_engine": use_third_party_engine
        }
        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", self.url + "/translate", headers=headers, data=payload
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
            raise ValueError("Could not make translation. " + err)