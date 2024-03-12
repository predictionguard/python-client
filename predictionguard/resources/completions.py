import os
import json
from typing import Any, Dict, List, Optional, Union
import requests

class Completions():
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
                                input: Optional[Dict[str, Any]] = None,
                                output: Optional[Dict[str, Any]] = None,
                                max_tokens: Optional[int] = 100,
                                temperature: Optional[float] = 0.75,
                                top_p: Optional[float] = 1.0,
                                stream: Optional[bool] = False
                                ) -> Dict[str, Any]:
        """
        Creates a completion request for the Prediction Guard /completions API.

        :param model: The ID(s) of the model to use.
        :param prompt: The prompt(s) to generate completions for.
        :param input: A dictionary containing the PII and injection arguments.
        :param output: A dictionary containing the consistency, factuality, and toxicity arguments.
        :param max_tokens: The maximum number of tokens to generate in the completion(s).
        :param temperature: The sampling temperature to use.
        :param top_p: The nucleus sampling probability to use.
        :param stream: A boolean enabled or disabling streaming model output.
        :return: A dictionary containing the completion response.
        """

        # Make sure we can connect to prediction guard.
        self._connect()

        # Create a list of tuples, each containing all the parameters for 
        # a call to _generate_completion
        args = (model, prompt, input, output, max_tokens, temperature, top_p, stream)

        # Run _generate_completion
        choices = self._generate_completion(*args)
        return choices
    
    @classmethod
    def _generate_completion(self, model, prompt, input, output, max_tokens, temperature, top_p, stream):
        """
        Function to generate a single completion. 
        """

        # Make a prediction using the proxy.
        headers = {
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
            "stream": stream
        }
        if input:
            payload_dict["input"] = input
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
        headers = {"Authorization": "Bearer " + self.token}
 
        response = requests.request("GET", url + "/completions", headers=headers)

        return list(response.json())