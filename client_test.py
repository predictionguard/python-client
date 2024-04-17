import os
import json

import pytest

from predictionguard import PredictionGuard


def test_specified_var():
    test_api_key = "predictionguard_api_key"
    test_url = "https://test.api.com/"

    assert ret_api_key == test_api_key
    assert ret_url == test_url

def test_env_var():
    test_api_key = "predictionguard_api_key"
    test_url = "https://test.api.com/"

    assert ret_api_key == test_api_key
    assert ret_url == test_url