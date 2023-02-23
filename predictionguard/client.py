import os
import json
import base64
import time
import sys
import itertools
import threading

import requests
from tabulate import tabulate


# The main Prediction Guard API URL.
url = "https://api.predictionguard.com"


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
           * email (str): The email associated with your Prediction Guard account.
           * password (str): The password associated with your Prediction Guard account.
        """

        # Get the email and password from the provided arguments.
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
            raise ValueError("Could not list proxies. Please try again.")

    def create_proxy(self, task: str, name: str, examples: list):
        """
        Create a new proxy.
        Args:
           * task (str): The task to create the proxy for.
           * name (str): The name of the proxy.
        """

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

        # Delete the proxy.
        headers = {"Authorization": "Bearer " + self.token}
        params = {"name": name}
        response = requests.request("DELETE", url + "/proxy", headers=headers, params=params)

        # If the request was successful, print the proxy info.
        if response.status_code == 200:
            print("Proxy deleted successfully!")
        else:
            raise ValueError("Could not delete proxy. Please try again.")

    def predict(self, name: str, data: dict):
        """
        Make a prediction using a proxy.
        Args:
           * name (str): The name of the proxy to make the prediction with.
           * data (dict): A sample to submit for inference.
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
            raise ValueError("Could not make prediction. Please try again.")
