import os

import requests
from typing import Optional, Union

from .src.audio import Audio
from .src.chat import Chat
from .src.completions import Completions
from .src.detokenize import Detokenize
from .src.documents import Documents
from .src.embeddings import Embeddings
from .src.rerank import Rerank
from .src.tokenize import Tokenize
from .src.translate import Translate
from .src.factuality import Factuality
from .src.toxicity import Toxicity
from .src.pii import Pii
from .src.injection import Injection
from .src.models import Models
from .version import __version__

__all__ = [
    "PredictionGuard", "Chat", "Completions", "Embeddings",
    "Audio", "Documents", "Rerank", "Tokenize", "Translate",
    "Detokenize", "Factuality", "Toxicity", "Pii", "Injection",
    "Models"
]

class PredictionGuard:
    """PredictionGuard provides access the Prediction Guard API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        timeout: Optional[Union[int, float]] = None
    ) -> None:
        """
        :param api_key: api_key represents PG api key.
        :param url: url represents the transport and domain:port
        :param timeout: request timeout in seconds.
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

        if not timeout:
            timeout = os.environ.get("TIMEOUT")
        if not timeout:
            timeout = None
        if timeout:
            try:
                timeout = float(timeout)
            except ValueError:
                raise ValueError(
                    "Timeout must be of type integer or float, not %s." % (type(timeout).__name__,)
                )
            except TypeError:
                raise TypeError(
                    "Timeout should be of type integer or float, not %s." % (type(timeout).__name__,)
                )
        self.timeout = timeout

        # Connect to Prediction Guard and set the access api_key.
        self._connect_client()

        # Pass Prediction Guard class variables to inner classes
        self.chat: Chat = Chat(self.api_key, self.url, self.timeout)
        """Chat generates chat completions based on a conversation history"""

        self.completions: Completions = Completions(self.api_key, self.url, self.timeout)
        """Completions generates text completions based on the provided input"""

        self.embeddings: Embeddings = Embeddings(self.api_key, self.url, self.timeout)
        """Embedding generates chat completions based on a conversation history."""

        self.audio: Audio = Audio(self.api_key, self.url, self.timeout)
        """Audio allows for the transcription of audio files."""

        self.documents: Documents = Documents(self.api_key, self.url, self.timeout)
        """Documents allows you to extract text from various document file types."""

        self.rerank: Rerank = Rerank(self.api_key, self.url, self.timeout)
        """Rerank sorts text inputs by semantic relevance to a specified query."""

        self.translate: Translate = Translate(self.api_key, self.url, self.timeout)
        """Translate converts text from one language to another."""

        self.factuality: Factuality = Factuality(self.api_key, self.url, self.timeout)
        """Factuality checks the factuality of a given text compared to a reference."""

        self.toxicity: Toxicity = Toxicity(self.api_key, self.url, self.timeout)
        """Toxicity checks the toxicity of a given text."""

        self.pii: Pii = Pii(self.api_key, self.url, self.timeout)
        """Pii replaces personal information such as names, SSNs, and emails in a given text."""

        self.injection: Injection = Injection(self.api_key, self.url, self.timeout)
        """Injection detects potential prompt injection attacks in a given prompt."""

        self.tokenize: Tokenize = Tokenize(self.api_key, self.url, self.timeout)
        """Tokenize generates tokens for input text."""

        self.detokenize: Detokenize = Detokenize(self.api_key, self.url, self.timeout)
        """Detokenizes generates text for input tokens."""

        self.models: Models = Models(self.api_key, self.url, self.timeout)
        """Models lists all of the models available in the Prediction Guard API."""

    def _connect_client(self) -> None:

        # Prepare the proper headers.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        # Try listing models to make sure we can connect.
        response = requests.request("GET", self.url + "/completions", headers=headers, timeout=self.timeout)

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
