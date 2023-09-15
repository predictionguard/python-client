import os
import json
import base64
import time
import sys
import itertools
import threading
from typing import Any, Dict, List, Optional, Union

import requests
from tabulate import tabulate


# The main Prediction Guard API URL.
if os.environ.get("PREDICTIONGUARD_URL") == None or os.environ.get("PREDICTIONGUARD_URL") == "":
    url = "https://api.predictionguard.com"
else:
    url = os.environ.get("PREDICTIONGUARD_URL")
    if url[0:8] != "https://" or url[-4::] != ".com":
        if url[0:8] != "https://":
            url = "https://" + url
        #if url[-4::] != ".com":
        #    url = url + ".com"
#url = "http://localhost:8080"

# The animation for the loading spinner.
done = True


def animate():
    for c in itertools.cycle(["|", "/", "-", "\\"]):
        if done:
            break
        sys.stdout.write(
            "\rCreating the proxy endpoint. Evaluating a" " bunch of SOTA models! " + c
        )
        sys.stdout.flush()
        time.sleep(0.1)
    #sys.stdout.write("\r\nProxy ready. 🎉     ")


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

            # Cache the email and password in a local config file.
            config_path = os.path.join(os.path.expanduser("~"), ".predictionguard")
            with open(config_path, "w") as config_file:
                # Write the email and password to the config file after base64 encoding.
                token_encoded = base64.b64encode(token.encode("utf-8")).decode("utf-8")
                config = {"token": token_encoded}
                json.dump(config, config_file)
        else:
            # Try and get the email and password from the environment variables.
            self.token = os.environ.get("PREDICTIONGUARD_TOKEN")

            # If the email and password are not in the environment variables,
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

                    # Get the email and password from the config file.
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
        # Connect to the Prediction Guard API.
        headers = {"Content-Type": "application/json"}
        params = {"token": self.token}

        response = requests.request(
            "GET", url + "/user", headers=headers, params=params
        )

        # If the connection was successful, set the access token.
        if response.status_code == 200:
            pass
        else:
            raise ValueError(
                "Could not connect to Prediction Guard API. Please check "
                "your access token and try again."
            )

    def list_proxies(self, print_table: bool = True):
        """
        List all proxies associated with your Prediction Guard account.
        """

        # Get the proxies associated with the user.
        headers = {"Authorization": "Bearer " + self.token}
        response = requests.request("GET", url + "/proxy", headers=headers)

        # If the request was successful, print the proxies.
        if response.status_code == 200:
            proxies = response.json()
            if not proxies:
                if print_table:
                    print("No proxies found. Please create a proxy first.")
                return []
            if not print_table:
                return proxies
            table = []
            for proxy in proxies:
                table.append(
                    [
                        proxy["name"],
                        proxy["task"],
                        proxy["status"],
                        proxy["created_at"],
                        proxy["updated_at"],
                        proxy["failure_rate"],
                        proxy["gpus"],
                    ]
                )
            print(
                tabulate(
                    table,
                    headers=[
                        "Name",
                        "Task",
                        "Status",
                        "Created At",
                        "Updated At",
                        "Failure Rate",
                        "GPUs",
                    ],
                )
            )
        else:
            err = ""
            try:
                err = response.json()["error"]
            except:
                pass
            raise ValueError("Could not list proxies. " + err)

    def create_proxy(self, task: str, name: str, examples: list, wait: bool = True):
        """
        Create a new proxy.
        Args:
           * task (str): The task to create the proxy for.
           * name (str): The name of the proxy.
           * examples (list): A list of examples to use to create the proxy.
           * wait (bool): Whether or not to wait for the proxy to be created.
        """

        # Check the task name.
        tasks = [
            "text-gen",
            "sentiment",
            "qa",
            "mt",
            "fact",
            "toxicity"
        ]
        if task not in tasks:
            raise ValueError(
                "Invalid task name. Please choose one of the following: {}".format(
                    ", ".join(tasks)
                )
            )

        # Use list_proxies to determine if one already exists with the given name.
        proxies = self.list_proxies(print_table=False)
        for proxy in proxies:
            if proxy["name"] == name:
                raise ValueError(
                    "A proxy with the name {} already exists. Please choose a "
                    "different name.".format(name)
                )

        # Change the global done variable to facilitate animation.
        global done
        done = False

        # Create the proxy.
        headers = {"Authorization": "Bearer " + self.token}
        payload = json.dumps(examples)

        # Put the task and name in the query string as parameters.
        params = {"task": task, "name": name}
        response = requests.request(
            "POST", url + "/proxy", headers=headers, data=payload, params=params
        )

        # If the request was successful, move on to wait for the proxy
        # to be ready for predictions.
        if response.status_code == 200:
            pass
        else:
            print(response)
            raise ValueError("Could not create proxy. Please try again.")
        
        # If the user does not want to wait for the proxy to be ready,
        # return.
        if not wait:
            print("Proxy created successfully! (use client.list_proxies() to see status)")
            return
        
        t = threading.Thread(target=animate)
        t.start()
        while not done:
            # Wait for the proxy to be ready.
            headers = {"Authorization": "Bearer " + self.token}
            response = requests.request("GET", url + "/proxy", headers=headers)

            # If the request was successful, print the proxy info.
            if response.status_code == 200:
                proxies = response.json()
                for proxy in proxies:
                    if proxy["name"] == name:
                        if proxy["status"] == "available":
                            done = True
                        else:
                            time.sleep(0.2)

        print("\n\nProxy created successfully!")
        print("---------------------------")
        print("Name: " + proxy["name"])
        print("Task: " + proxy["task"])
        print("Status: " + proxy["status"])
        print("Failure Rate: " + str(proxy["failure_rate"]))

    def delete_proxy(self, name: str):
        """
        Delete a proxy.
        Args:
           * name (str): The name of the proxy to delete.
        """

        # Use list_proxies to determine if one already exists with the given name.
        proxies = self.list_proxies(print_table=False)
        names = []
        for proxy in proxies:
            names.append(proxy["name"])
        if name not in names:
            raise ValueError(
                "Can't find a proxy with the name {}.".format(name)
            )

        # Delete the proxy.
        headers = {"Authorization": "Bearer " + self.token}
        params = {"name": name}
        response = requests.request("DELETE", url + "/proxy", headers=headers, params=params)

        # If the request was successful, print the proxy info.
        if response.status_code == 200:
            print("Proxy deleted successfully!")
        else:
            err = ""
            try:
                err = response.json()["error"]
            except:
                pass
            raise ValueError("Could not delete proxy. " + err)

    def predict(self, name: str, data: dict):
        """
        Make a prediction using a proxy.
        Args:
           * name (str): The name of the proxy to make the prediction with.
           * data (dict): A input sample to submit for inference.
        """

        # Make a prediction using the proxy.
        headers = {"Authorization": "Bearer " + self.token}
        payload = json.dumps(data)
        params = {"name": name}
        response = requests.request(
            "POST", url + "/predict", headers=headers, data=payload, params=params
        )

        # If the request was successful, print the proxies.
        if response.status_code == 200:
            prediction = response.json()
            return prediction
        else:
            # Check if there is a json body in the response. Read that in,
            # print out the error field in the json body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except:
                pass
            raise ValueError("Could not make prediction. " + err)
        

class Completion():
    """
    OpenAI-compatible completion API
    """

    @classmethod
    def _connect(self) -> None:
        """
        Initialize a Prediction Guard client to check access.
        Args:
           * token (str): The user token associated with your Prediction Guard account.
        """

        # Try and get the access token.
        self.token = os.environ.get("PREDICTIONGUARD_TOKEN")

        # If the email and password are not in the environment variables,
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

                # Get the email and password from the config file.
                if "token" not in config:
                    raise ValueError(
                        "Imporperly formatted predictionguard config "
                        "file at ~/.predictionguard."
                    )
                else:
                    self.token = base64.b64decode(config["token"]).decode("utf-8")

        # Connect to Prediction Guard and set the access token.
        try:
            Client(token=self.token)
        except:
            if "api.predictionguard.com" in url:
                raise ValueError("Could not connect to Prediction Guard with the provided token.")

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
        if "api.predictionguard.com" in url:
            headers = {"Authorization": "Bearer " + self.token}
        else:
            headers = {"x-api-key": self.token}
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
        if "api.predictionguard.com" in url:
            headers = {"Authorization": "Bearer " + self.token}
        else:
            headers = {"x-api-key": self.token}
 
        response = requests.request("GET", url + "/completions", headers=headers)

        return list(response.json())
    

class Factuality():

    @classmethod
    def _connect(self) -> None:
        """
        Initialize a Prediction Guard client to check access.
        Args:
           * token (str): The user token associated with your Prediction Guard account.
        """

        # Try and get the access token.
        self.token = os.environ.get("PREDICTIONGUARD_TOKEN")

        # If the email and password are not in the environment variables,
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

                # Get the email and password from the config file.
                if "token" not in config:
                    raise ValueError(
                        "Imporperly formatted predictionguard config "
                        "file at ~/.predictionguard."
                    )
                else:
                    self.token = base64.b64decode(config["token"]).decode("utf-8")

        # Connect to Prediction Guard and set the access token.
        try:
            Client(token=self.token)
        except:
            if "api.predictionguard.com" in url:
                raise ValueError("Could not connect to Prediction Guard with the provided token.")

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
        if "api.predictionguard.com" in url:
            headers = {"Authorization": "Bearer " + self.token}
        else:
            headers = {"x-api-key": self.token}
 

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
        Args:
           * token (str): The user token associated with your Prediction Guard account.
        """

        # Try and get the access token.
        self.token = os.environ.get("PREDICTIONGUARD_TOKEN")

        # If the email and password are not in the environment variables,
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

                # Get the email and password from the config file.
                if "token" not in config:
                    raise ValueError(
                        "Imporperly formatted predictionguard config "
                        "file at ~/.predictionguard."
                    )
                else:
                    self.token = base64.b64decode(config["token"]).decode("utf-8")

        # Connect to Prediction Guard and set the access token.
        try:
            Client(token=self.token)
        except:
            if "api.predictionguard.com" in url:
                raise ValueError("Could not connect to Prediction Guard with the provided token.")

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
        if "api.predictionguard.com" in url:
            headers = {"Authorization": "Bearer " + self.token}
        else:
            headers = {"x-api-key": self.token}
 

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
