import re
import json
import os
import base64

import requests
from typing import Any, Dict, List, Union, Optional
import urllib.request
import urllib.parse
import uuid

from ..version import __version__


class Embeddings:
    """Embedding generates chat completions based on a conversation history.

    Usage::

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        response = client.embeddings.create(
            model="multilingual-e5-large-instruct",
            input="This is how you generate embeddings with Prediction Guard"
        )

        print(json.dumps(response, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
        self,
        model: str,
        input: Union[
            str,
            List[Union[
                str,
                int,
                List[int],
                Dict[str, str]
                ]
            ]
        ],
        truncate: bool = False,
        truncation_direction: str = "right",
    ) -> Dict[str, Any]:
        """
        Creates an embeddings request to the Prediction Guard /embeddings API

        :param model: Model to use for embeddings
        :param input: String, list of strings, or list of dictionaries containing input data with text and image keys.
        :param truncate: Whether to truncate input text.
        :param truncation_direction: Direction to truncate input text.
        :result:
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _generate_translation
        args = (model, input, truncate, truncation_direction)

        # Run _generate_embeddings
        choices = self._generate_embeddings(*args)
        return choices

    def _generate_embeddings(self, model, input, truncate, truncation_direction):
        """
        Function to generate an embeddings response.
        """

        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        if type(input) is list and type(input[0]) is dict:
            inputs = []
            for item in input:
                item_dict = {}
                if "text" in item.keys():
                    item_dict["text"] = item["text"]
                if "image" in item.keys():
                    image_url_check = urllib.parse.urlparse(item["image"])
                    data_uri_pattern = re.compile(
                        r'^data:([a-zA-Z0-9!#$&-^_]+/[a-zA-Z0-9!#$&-^_]+)?(;base64)?,.*$'
                    )

                    if os.path.exists(item["image"]):
                        with open(item["image"], "rb") as image_file:
                            image_input = base64.b64encode(image_file.read()).decode(
                                "utf-8"
                            )

                    elif re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", item["image"]):
                        if (
                            base64.b64encode(base64.b64decode(item["image"])).decode(
                                "utf-8"
                            )
                            == item["image"]
                        ):
                            image_input = item["image"]

                    elif image_url_check.scheme in ("http", "https", "ftp"):
                        temp_image = uuid.uuid4().hex + ".jpg"
                        urllib.request.urlretrieve(item["image"], temp_image)
                        with open(temp_image, "rb") as image_file:
                            image_input = base64.b64encode(image_file.read()).decode(
                                "utf-8"
                            )
                        os.remove(temp_image)

                    elif data_uri_pattern.match(item["image"]):
                        #process data_uri
                        comma_find = item["image"].rfind(',')
                        image_input = item["image"][comma_find + 1:]

                    else:
                        raise ValueError(
                            "Please enter a valid base64 encoded image, image file, image URL, or data URI."
                        )

                    item_dict["image"] = image_input

                inputs.append(item_dict)

        else:
            inputs = input

        if truncation_direction == "right":
            truncation_direction = "Right"
        elif truncation_direction == "left":
            truncation_direction = "Left"
        else:
            raise ValueError(
                "Please enter either 'Right' or 'Left' for the truncation_direction value."
            )

        payload_dict = {
            "model": model,
            "input": inputs,
            "truncate": truncate,
            "truncation_direction": truncation_direction
        }

        payload = json.dumps(payload_dict)
        response = requests.request(
            "POST", self.url + "/embeddings", headers=headers, data=payload
        )

        # If the request was successful, print the proxies.
        if response.status_code == 200:
            ret = response.json()
            return ret
        elif response.status_code == 429:
            raise ValueError(
                "Could not connect to Prediction Guard API. "
                "Too many requests, rate limit or quota exceeded."
            )
        else:
            # Check if there is a json body in the response. Read that in,
            # print out the error field in the json body, and raise an exception.
            err = ""
            try:
                err = response.json()["error"]
            except Exception:
                pass
            raise ValueError("Could not generate embeddings. " + err)

    def list_models(self, capability: Optional[str] = "embedding") -> List[str]:
        # Get the list of current models.
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        if capability != "embedding" and capability != "embedding-with-image":
            raise ValueError(
                "Please enter a valid models type "
                "(embedding or embedding-with-image)."
            )
        else:
            model_path = "/models/" + capability

        response = requests.request("GET", self.url + model_path, headers=headers)

        response_list = []
        for model in response.json()["data"]:
            response_list.append(model["id"])

        return response_list
