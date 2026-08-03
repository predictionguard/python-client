import contextlib
from typing import Any

import requests

from ..version import __version__


class Documents:
    """
    Documents allows you to extract text from various document file types.

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

        response = client.documents.extract.create(
            file="sample.pdf"
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

        self.extract: DocumentsExtract = DocumentsExtract(self.api_key, self.url, self.timeout)

class DocumentsExtract:
    def __init__(self, api_key, url, timeout):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout

    def create(
        self,
        file: str,
        embed_images: bool | None = False,
        output_format: str | None = None,
        chunk_document: bool | None = False,
        chunk_size: int | None = None,
        enable_ocr: bool | None = True,
        toxicity: bool | None = False,
        pii: str | None = "",
        replace_method: str | None = "",
        entity_list: list[str] | None = "",
        injection: bool | None = False,
    ) -> dict[str, Any]:
        """
        Creates a documents request to the Prediction Guard /documents/extract API

        :param file: Document to be parsed
        :param embed_images: Whether to embed images into documents
        :param output_format: Output format
        :param chunk_document: Whether to chunk documents into chunks
        :param chunk_size: Chunk size
        :param enable_ocr: Whether to enable OCR
        :param toxicity: Whether to check for output toxicity
        :param pii: Whether to check for or replace pii
        :param replace_method: Replace method for any PII that is present.
        :param entity_list: List of entities to ignore in the PII check.
        :param injection: Whether to check for prompt injection
        :result: A dictionary containing the title, content, and length of the document.
        """

        # Run _extract_documents
        choices = self._extract_documents(
            file, embed_images, output_format,
            chunk_document, chunk_size, enable_ocr,
            toxicity, pii, replace_method, entity_list, injection
        )
        return choices

    def _extract_documents(
            self, file, embed_images,
            output_format, chunk_document,
            chunk_size, enable_ocr, toxicity,
            pii, replace_method, entity_list, injection
    ):
        """
        Function to extract a document.
        """

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
            "Toxicity": str(toxicity),
            "Pii": str(pii),
            "Replace-Method": str(replace_method),
            "Entity-List": str(entity_list),
            "Injection": str(injection)
        }

        data = {
            "embedImages": embed_images,
            "outputFormat": output_format,
            "chunkDocument": chunk_document,
            "chunkSize": chunk_size,
            "enableOCR": enable_ocr,
        }

        with open(file, "rb") as doc_file:
            files = {"file": (file, doc_file)}

            response = requests.request(
                "POST", self.url + "/documents/extract",
                headers=headers, files=files, data=data, timeout=self.timeout
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
            with contextlib.suppress(Exception):
                err = response.json()["error"]
            raise ValueError("Could not extract document. " + err)