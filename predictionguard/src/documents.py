import json
from pyexpat import model

import requests
from typing import Any, Dict, Optional

from ..version import __version__


class Documents:
    """Documents allows you to extract text from various document file types.

    Usage::

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        response = client.documents.extract.create(
            file="sample.pdf"
        )

        print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

        self.extract: DocumentsExtract = DocumentsExtract(self.api_key, self.url)

class DocumentsExtract:
    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
        self,
        file: str
    ) -> Dict[str, Any]:
        """
        Creates a documents request to the Prediction Guard /documents/extract API

        :param file: Document to be parsed
        :result: A dictionary containing the title, content, and length of the document.
        """

        # Run _extract_documents
        choices = self._extract_documents(file)
        return choices

    def _extract_documents(self, file):
        """
        Function to extract a document.
        """

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        with open(file, "rb") as doc_file:
            files = {"file": (file, doc_file)}

            response = requests.request(
                "POST", self.url + "/documents/extract", headers=headers, files=files
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
            raise ValueError("Could not extract document. " + err)