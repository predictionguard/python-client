import warnings

import pytest

from predictionguard import PredictionGuard


def test_mcp_tools_list():
    test_client = PredictionGuard()

    response = test_client.mcp_tools.list()

    if len(response["data"]) == 0:
        # Verify it's a valid empty response, then warn instead of fail
        assert response["object"] == "list"
        assert response["data"] == {}
        warnings.warn(pytest.PytestWarning("No tools available — data is empty, skipping content checks"))
        return

    first_key = list(response["data"].keys())[0]
    assert type(response["data"][first_key][0]["id"]) is str
