import requests
from typing import Any, Dict, Optional

from ..version import __version__


class MCPTools:
    """
    MCPTools lists all the MCP tools available in the Prediction Guard API.

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

        response = client.mcp_tools.list()

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

    def list(self) -> Dict[str, Any]:
        """
        Creates a mcp_tools list request in the Prediction Guard REST API.

        :return: A dictionary containing the metadata of all the MCP tools.
        """

        # Run _list_mcp_tools
        choices = self._list_mcp_tools()
        return choices

    def _list_mcp_tools(self):
        """
        Function to list available MCP tools.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        response = requests.request(
            "GET", self.url + "/mcp_tools", headers=headers, timeout=self.timeout
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
            # Check if there is a JSON body in the response. Read that in,
            # print out the error field in the JSON body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except Exception:
                pass
            raise ValueError("Could not check for injection. " + err)
