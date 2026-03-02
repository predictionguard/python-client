import warnings

import pytest

from predictionguard import PredictionGuard


def test_mcp_servers_list():
    test_client = PredictionGuard()

    response = test_client.mcp_servers.list()

    if len(response["data"]) == 0:
        assert response["object"] == "list"
        assert response["data"] == []
        warnings.warn(pytest.PytestWarning("No servers available — data is empty, skipping content checks"))
        return

    assert type(response["data"][0]["server_label"]) is str
