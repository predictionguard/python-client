import json
from typing import Any, Dict, List, Optional

import requests


class Chat():
    """
    Chat API
    """

    @classmethod
    def _connect(self) -> None:
        """
        Initialize a Prediction Guard client to check access
        """
        client = Client()
        self.token = client.get_token()

    @classmethod
    def create(
        self, 
        model: str,
        messages: List[Dict[str, str]],
        input: Optional[Dict[str, Any]] = None,
        output: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = 100,
        temperature: Optional[float] = 0.75,
        top_p: Optional[float] = 1.0,
        stream: Optional[bool] = False
        ) -> Dict[str, Any]:
        """
        Creates a chat request for the Prediction Guard /chat API.

        :param model: The ID(s) of the model to use.
        :param messages: The content of the call, an array of dictionaries containing a role and content.
        :param input: A dictionary containing the PII and injection arguments.
        :param output: A dictionary containing the consistency, factuality, and toxicity arguments.
        :param max_tokens: The maximum amount of tokens the model should return.
        :param temperature: The consistency of the model responses to the same prompt. The higher the more consistent.
        :param top_p: The sampling for the model to use.
        :param stream: A boolean enabled or disabling streaming model output.
        :return: A dictionary containing the chat response.
        """

        # Make sure we can connect to prediction guard.
        self._connect()

        # Create a list of tuples, each containing all the parameters for 
        # a call to _generate_chat
        args = (model, messages, input, output, max_tokens, temperature, top_p, stream)

        # Run _generate_chat
        choices = self._generate_chat(*args)
        return choices

    @classmethod
    def _generate_chat(self, model, messages, input, output, max_tokens, temperature, top_p, streaming):
        """
        Function to generate a single chat response.
        """
        
        headers = {
            "Authorization": "Bearer " + self.token,
            "Accept": "text/event-stream"
            }

        payload_dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        
        if input:
            payload_dict["input"] = input
        if output:
            payload_dict["output"] = output

        payload = json.dumps(payload_dict)
        with requests.request("POST", url=url, stream=streaming, headers=headers, data=payload) as stream:
                for chunk in stream.iter_lines():
                    if 'data' in str(chunk):
                        yield json.loads(chunk.decode('utf-8').split('data:')[1].strip())
                    else:
                        # Check if there is a json body in the response. Read that in,
                        # print out the error field in the json body, and raise an exception.
                        err = ""
                        try:
                            err = chunk.json()["error"]
                        except:
                            pass
                        raise ValueError("Could not make prediction. " + err)
                    
    @classmethod
    def list_models(self) -> List[str]:
        # Commented out parts are there for easier fix when
        # functionality for this call on chat endpoint added
        model_list = [
            "deepseek-coder-6.7b-instruct", 
            "Neural-Chat-7B", 
            "Nous-Hermes-2-SOLAR_10.7B", 
            "Yi-34B-Chat"
            ]

        # Make sure we can connect to prediction guard.
        # self._connect()

        # Get the list of current models.
        # headers = {
        #         "x-api-key": self.token
        #         }
 
        # response = requests.request("GET", url + "/chat", headers=headers)

        # return list(response.json())
        return model_list 