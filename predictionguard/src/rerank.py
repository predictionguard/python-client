import json

import requests
from typing import Any, Dict, List, Optional

from ..version import __version__


class Rerank:
    """Rerank sorts text inputs by semantic relevance to a specified query.

        Usage::

            import os
            import json

            from predictionguard import PredictionGuard

            # Set your Prediction Guard token as an environmental variable.
            os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

            client = PredictionGuard()

            response = client.rerank.create(
                model="bge-reranker-v2-m3",
                query="What is Deep Learning?",
                documents=[
                    "Deep Learning is pizza.",
                    "Deep Learning is not pizza."
                ],
                return_documents=True
            )

            print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
        """


    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
            self,
            model: str,
            query: str,
            documents: List[str],
            return_documents: Optional[bool] = True
    ) -> Dict[str, Any]:
        """
        Creates a rerank request in the Prediction Guard /rerank API.

        :param model: The model to use for reranking.
        :param query: The query to rank against.
        :param documents: The documents to rank.
        :param return_documents: Whether to return documents with score.
        :return: A dictionary containing the tokens and token metadata.
        """

        # Run _create_rerank
        choices = self._create_rerank(model, query, documents, return_documents)
        return choices

    def _create_rerank(self, model, query, documents, return_documents):
        """
        Function to rank text.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        payload = {
            "model": model,
            "query": query,
            "documents": documents,
            "return_documents": return_documents
        }

        payload = json.dumps(payload)

        response = requests.request(
            "POST", self.url + "/rerank", headers=headers, data=payload
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
            raise ValueError("Could not rank documents. " + err)

    def list_models(self):
        # Get the list of current models.
        headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key,
                "User-Agent": "Prediction Guard Python Client: " + __version__
                }

        response = requests.request("GET", self.url + "/models/rerank", headers=headers)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list
