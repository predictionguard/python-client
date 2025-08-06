from typing import Any, Dict, Optional


class Translate:
    """No longer supported."""

    def __init__(self, api_key, url, timeout):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout

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