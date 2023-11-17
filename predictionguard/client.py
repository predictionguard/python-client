import os
import json
import base64
from typing import Any, Dict, List, Optional, Union

import requests


# The main Prediction Guard API URL.
if os.environ.get("PREDICTIONGUARD_URL") == None or os.environ.get("PREDICTIONGUARD_URL") == "":
    url = "https://api.predictionguard.com"
else:
    url = os.environ.get("PREDICTIONGUARD_URL")
    

# The main Prediction Guard client class.
class Client:
    def __init__(self, token: str = None) -> None:
        """
        Initialize the client.
        Args:
           * token (str): The user token associated with your Prediction Guard account.
        """

        # Get the access token.
        if token:
            self.token = token

            # Cache the token locally.
            config_path = os.path.join(os.path.expanduser("~"), ".predictionguard")
            with open(config_path, "w") as config_file:
                token_encoded = base64.b64encode(token.encode("utf-8")).decode("utf-8")
                config = {"token": token_encoded}
                json.dump(config, config_file)

        else:
            # Try and get the token from the environment variables.
            self.token = os.environ.get("PREDICTIONGUARD_TOKEN")

            # If the token is not in the environment variables,
            # try to get the creds from a local config file.
            if not self.token:
                # Get the path to the config file.
                config_path = os.path.join(os.path.expanduser("~"), ".predictionguard")

                # If the config file does not exist, raise an error.
                if not os.path.exists(config_path):
                    raise ValueError(
                        "No access token provided and no predictionguard"
                        " config file found. Please provide the access token as "
                        "arguments or set the environment variable PREDICTIONGUARD_TOKEN."
                    )
                
                else:
                    # Read the JSON config file.
                    with open(config_path, "r") as config_file:
                        config = json.load(config_file)

                    # Get the token from the config file.
                    if "token" not in config:
                        raise ValueError(
                            "Imporperly formatted predictionguard config "
                            "file at ~/.predictionguard."
                        )
                    else:
                        self.token = base64.b64decode(config["token"]).decode("utf-8")

        # Connect to Prediction Guard and set the access token.
        self.connect_client()

    def connect_client(self) -> None:

        # Prepare the proper headers.
        headers = {
                "Content-Type": "application/json",
                "x-api-key": self.token
                }
        params = {"token": self.token}

        # Try listing models to make sure we can connect.
        response = requests.request(
            "GET", url + "/completions", headers=headers, params=params
        )

        # If the connection was unsuccessful, raise an exception.
        if response.status_code == 200:
            pass
        else:
            raise ValueError(
                "Could not connect to Prediction Guard API with the given token. "
                "Please check your access token and try again."
            ) 
        return str(self.token)

    def get_token(self) -> str:
        return self.token


class Completion():
    """
    OpenAI-compatible completion API
    """

    @classmethod
    def _connect(self) -> None:
        """
        Initialize a Prediction Guard client to check access.
        """

        client = Client()
        self.token = client.get_token()

    @classmethod
    def create(self, model: str, prompt: Union[str, List[str]],
                                output: Optional[Dict[str, Any]] = None,
                                max_tokens: Optional[int] = 100,
                                temperature: Optional[float] = 0.75,
                                top_p: Optional[float] = 1.0,
                                top_k: Optional[int] = 50) -> Dict[str, Any]:
        """
        Creates a completion request for the Prediction Guard /completions API.

        :param model: The ID(s) of the model to use.
        :param prompt: The prompt(s) to generate completions for.
        :param max_tokens: The maximum number of tokens to generate in the completion(s).
        :param temperature: The sampling temperature to use.
        :param top_p: The nucleus sampling probability to use.
        :param n: The number of completions to generate.
        :return: A dictionary containing the completion response.
        """

        # Make sure we can connect to prediction guard.
        self._connect()

        # Create a list of tuples, each containing all the parameters for 
        # a call to _generate_completion
        args = (model, prompt, output, max_tokens, temperature, top_p, top_k)

        # Run _generate_completion
        choices = self._generate_completion(*args)
        return choices
    
    @classmethod
    def _generate_completion(self, model, prompt, output, max_tokens, temperature, top_p, top_k):
        """
        Function to generate a single completion. 
        """

        # Make a prediction using the proxy.
        headers = {
                "x-api-key": "" + self.token,
                "Authorization": "Bearer " + self.token
                }
        if isinstance(model, list):
            model_join = ",".join(model)
        else:
            model_join = model
        if "openai" in model_join.lower():
            if os.environ.get("OPENAI_API_KEY") != "":
                headers["OpenAI-ApiKey"] = os.environ.get("OPENAI_API_KEY")
            else:
                raise ValueError("OpenAI API key not set. Please set the environment variable OPENAI_API_KEY.")
        payload_dict = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k
        }
        if output:
            payload_dict["output"] = output
        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", url + "/completions", headers=headers, data=payload
        )

        # If the request was successful, print the proxies.
        if response.status_code == 200:
            ret = response.json()
            return ret
        else:
            # Check if there is a json body in the response. Read that in,
            # print out the error field in the json body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except:
                pass
            raise ValueError("Could not make prediction. " + err)

    @classmethod
    def list_models(self) -> List[str]:

        # Make sure we can connect to prediction guard.
        self._connect()

        # Get the list of current models.
        headers = {
                "x-api-key": self.token,
                "Authorization": "Bearer " + self.token
                }
 
        response = requests.request("GET", url + "/completions", headers=headers)

        return list(response.json())
    

class Factuality():

    @classmethod
    def _connect(self) -> None:
        """
        Initialize a Prediction Guard client to check access.
        """
        client = Client()
        self.token = client.get_token()

    @classmethod
    def check(self, reference: str,
                    text: str) -> Dict[str, Any]:
        """
        Creates a factuality checking request for the Prediction Guard /factuality API.

        :param reference: The reference text used to check for factual consistency.
        :param text: The text to check for factual consistency.
        """

        # Make sure we can connect to prediction guard.
        self._connect()

        # Run _generate_score
        choices = self._generate_score(reference, text)
        return choices
    
    @classmethod
    def _generate_score(self, reference, text):
        """
        Function to generate a single factuality score. 
        """

        # Make a prediction using the proxy.
        headers = {
                "x-api-key": self.token,
                "Authorization": "Bearer " + self.token
                }

        payload_dict = {
            "reference": reference,
            "text": text
        }
        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", url + "/factuality", headers=headers, data=payload
        )

        # If the request was successful, print the proxies.
        if response.status_code == 200:
            ret = response.json()
            return ret
        else:
            # Check if there is a json body in the response. Read that in,
            # print out the error field in the json body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except:
                pass
            raise ValueError("Could not check factuality. " + err)
        

class Toxicity():

    @classmethod
    def _connect(self) -> None:
        """
        Initialize a Prediction Guard client to check access.
        """
        client = Client()
        self.token = client.get_token()

    @classmethod
    def check(self, text: str) -> Dict[str, Any]:
        """
        Creates a toxicity checking request for the Prediction Guard /toxicity API.

        :param text: The text to check for toxicity.
        """

        # Make sure we can connect to prediction guard.
        self._connect()

        # Run _generate_score
        choices = self._generate_score(text)
        return choices
    
    @classmethod
    def _generate_score(self, text):
        """
        Function to generate a single toxicity score. 
        """

        # Make a prediction using the proxy.
        headers = {
                "x-api-key": self.token,
                "Authorization": "Bearer " + self.token
                }

        payload_dict = {"text": text}
        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", url + "/toxicity", headers=headers, data=payload
        )

        # If the request was successful, print the proxies.
        if response.status_code == 200:
            ret = response.json()
            return ret
        else:
            # Check if there is a json body in the response. Read that in,
            # print out the error field in the json body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except:
                pass
            raise ValueError("Could not check toxicity. " + err)
