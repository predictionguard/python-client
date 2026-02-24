from predictionguard import PredictionGuard


def test_mcp_servers_list():
    test_client = PredictionGuard()

    response = test_client.mcp_servers.list()

    assert len(response["data"]) > 0
    assert type(response["data"][0]["server_label"]) is str
