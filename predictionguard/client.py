import os

import requests
from typing import Optional

from .source.predictionguard.chat import Chat
from .source.predictionguard.completions import Completions
from .source.predictionguard.embeddings import Embeddings
from .source.predictionguard.tokenize import Tokenize
from .source.predictionguard.translate import Translate
from .source.predictionguard.factuality import Factuality
from .source.predictionguard.toxicity import Toxicity
from .source.predictionguard.pii import Pii
from .source.predictionguard.injection import Injection
from .version import __version__

class PredictionGuard:
    """PredictionGuard provides access the Prediction Guard API."""

    def __init__(
        self, api_key: Optional[str] = None, url: Optional[str] = None
    ) -> None:
        """
        :param api_key: api_key represents PG api key.
        :param url: url represents the transport and domain:port
        """

        # Get the access api_key.
        if not api_key:
            api_key = os.environ.get("PREDICTIONGUARD_API_KEY")

        if not api_key:
            raise ValueError(
                "No api_key provided or in environment. "
                "Please provide the api_key as "
                "client = PredictionGuard(api_key=<your_api_key>) "
                "or as PREDICTIONGUARD_API_KEY in your environment."
            )
        self.api_key = api_key

        if not url:
            url = os.environ.get("PREDICTIONGUARD_URL")
        if not url:
            url = "https://api.predictionguard.com"
        self.url = url

        # Connect to Prediction Guard and set the access api_key.
        self._connect_client()

        # Pass Prediction Guard class variables to inner classes
        self.chat: Chat = Chat(self.api_key, self.url)
        """Chat generates chat completions based on a conversation history"""

        self.completions: Completions = Completions(self.api_key, self.url)
        """Completions generates text completions based on the provided input"""

        self.embeddings: Embeddings = Embeddings(self.api_key, self.url)
        """Embedding generates chat completions based on a conversation history."""

        self.translate: Translate = Translate(self.api_key, self.url)
        """Translate converts text from one language to another."""

        self.factuality: Factuality = Factuality(self.api_key, self.url)
        """Factuality checks the factuality of a given text compared to a reference."""

        self.toxicity: Toxicity = Toxicity(self.api_key, self.url)
        """Toxicity checks the toxicity of a given text."""

        self.pii: Pii = Pii(self.api_key, self.url)
        """Pii replaces personal information such as names, SSNs, and emails in a given text."""

        self.injection: Injection = Injection(self.api_key, self.url)
        """Injection detects potential prompt injection attacks in a given prompt."""

        self.tokenize: Tokenize = Tokenize(self.api_key, self.url)
        """Tokenize generates tokens for input text."""

    def _connect_client(self) -> None:

        # Prepare the proper headers.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        # Try listing models to make sure we can connect.
        response = requests.request("GET", self.url + "/completions", headers=headers)

        # If the connection was unsuccessful, raise an exception.
        if response.status_code == 200:
            pass
        elif response.status_code == 401:
            raise ValueError(
                "Could not connect to Prediction Guard API with the given api_key. "
                "Please check your access api_key and try again."
            )
        elif response.status_code == 404:
            raise ValueError(
                "Could not connect to Prediction Guard API with given url. "
                "Please check url specified, if no url specified, "
                "Please contact support."
            )
