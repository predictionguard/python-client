import os
import json
import re

import pytest

from predictionguard import PredictionGuard


# tests for api_key auth
def test_specified_var():
    test_api_key = os.environ["PREDICTIONGUARD_API_KEY"]
    test_url = os.environ["PREDICTIONGUARD_URL"]

    test_client = PredictionGuard(
        api_key=test_api_key,
        url=test_url
    )

    ret_api_key, ret_url = test_client.connect_client()

    assert ret_api_key == test_api_key
    assert ret_url == test_url


def test_env_var():
    test_client = PredictionGuard()

    ret_api_key, ret_url = test_client.connect_client()

    assert ret_api_key == os.environ["PREDICTIONGUARD_API_KEY"]
    assert ret_url == os.environ["PREDICTIONGUARD_URL"]


def test_fail_var():
    invalid_key_error = """
Could not connect to Prediction Guard API with the given api_key. 
Please check your access api_key and try again.
""".replace("\n", "")

    with pytest.raises(ValueError, match=invalid_key_error):
        PredictionGuard(
            api_key="i-will-fail-this-test"
        )


def test_fail_url():
    invalid_url_error = """
Could not connect to Prediction Guard API with given url. 
Please check url specified, if no url specified, 
Please contact support.
""".replace("\n", "")

    with pytest.raises(ValueError, match=invalid_url_error):
        PredictionGuard(
            api_key=os.environ["PREDICTIONGUARD_API_KEY"],
            url="https://www.google.com/"
            )
        
    
# deletes api_key from env, keep at bottom of tests
def test_no_key():
    del os.environ["PREDICTIONGUARD_API_KEY"]

    no_key_error = """
No api_key provided or in environment. 
Please provide the api_key as client = PredictionGuard(api_key=<your_api_key>) 
or as PREDICTIONGUARD_API_KEY in your environment.
""".replace("\n", "")
    
    no_key_error = re.escape(no_key_error)

    with pytest.raises(ValueError, match=no_key_error):
        PredictionGuard()