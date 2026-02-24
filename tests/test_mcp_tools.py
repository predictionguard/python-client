from predictionguard import PredictionGuard


def test_mcp_tools_list():
    test_client = PredictionGuard()

    response = test_client.mcp_tools.list()

    assert len(response["data"]) > 0
    first_key = list(response["data"].keys())[0]
    assert type(response["data"][first_key][0]["id"]) is str
