import json

import requests
from typing import Any, Dict, Optional

from ..version import __version__


class Translate:
    """No longer supported.
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
            self,
            text: Optional[str],
            source_lang: Optional[str],
            target_lang: Optional[str],
            use_third_party_engine: Optional[bool] = False
        ) -> Dict[str, Any]:
        """
        No longer supported
        """

        raise ValueError(
            "The translate functionality is no longer supported."
        )