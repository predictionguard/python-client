from typing import Any


class Translate:
    """No longer supported."""

    def __init__(self, api_key, url, timeout):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout

    def create(
            self,
            text: str | None,
            source_lang: str | None,
            target_lang: str | None,
            use_third_party_engine: bool | None = False
        ) -> dict[str, Any]:
        """
        No longer supported
        """

        raise ValueError(
            "The translate functionality is no longer supported."
        )