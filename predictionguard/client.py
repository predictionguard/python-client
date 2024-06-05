import base64
import json
import os
import re
import requests
from typing import Any, Dict, List, Optional, Union
import urllib.request
import urllib.parse


# The main Prediction Guard client class.
class PredictionGuard:
    def __init__(
            self, 
            api_key: str = None,
            url: str = None
            ) -> None:
        """
        Initialize the client.
        Args:
        * api_key (str): The user api_key associated with your Prediction Guard account.
        * url (str): The url for the Prediction Guard API of your choice.
        """

        # Get the access api_key.
        if api_key:
            self.api_key = api_key
 
        # Try and get the api_key from the environment variables
        elif "PREDICTIONGUARD_API_KEY" in os.environ:
            self.api_key = os.environ.get("PREDICTIONGUARD_API_KEY")

        else:
            raise ValueError(
                "No api_key provided or in environment. "
                "Please provide the api_key as "
                "client = PredictionGuard(api_key=<your_api_key>) "
                "or as PREDICTIONGUARD_API_KEY in your environment."
            )

        if url:
            self.url = url

        else:
            if "PREDICTIONGUARD_URL" in os.environ:
                self.url = os.environ["PREDICTIONGUARD_URL"]
        
            else:
                self.url = "https://api.predictionguard.com"

        # Connect to Prediction Guard and set the access api_key.
        self.connect_client()

        # Pass Prediction Guard class variables to inner classes
        self.completions = self.Completions(self.api_key, self.url)
        self.chat = self.Chat(self.api_key, self.url)
        self.embeddings = self.Embeddings(self.api_key, self.url)
        self.translate = self.Translate(self.api_key, self.url)
        self.factuality = self.Factuality(self.api_key, self.url)
        self.toxicity = self.Toxicity(self.api_key, self.url)
        self.pii = self.Pii(self.api_key, self.url)
        self.injection = self.Injection(self.api_key, self.url)


    def connect_client(self) -> None:

        # Prepare the proper headers.
        headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

        # Try listing models to make sure we can connect.
        response = requests.request(
            "GET", self.url + "/completions", headers=headers
        )

        # If the connection was unsuccessful, raise an exception.
        if response.status_code == 200:
            pass
        elif response.status_code == 403:
            raise ValueError(
                "Could not connect to Prediction Guard API with the given api_key. "
                "Please check your access api_key and try again."
            )
        elif response.status_code == 404:
            raise ValueError(
                "Could not connect to Prediction Guard API with given url. "
                "Please check url specified, if no url specified, "
                "Please contact support."
            )
        
        return str(self.api_key), str(self.url)


    class Completions:
        """
        OpenAI-compatible completion API
        """
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def create(self, model: str, prompt: Union[str, List[str]],
                                    input: Optional[Dict[str, Any]] = None,
                                    output: Optional[Dict[str, Any]] = None,
                                    max_tokens: Optional[int] = 100,
                                    temperature: Optional[float] = 0.75,
                                    top_p: Optional[float] = 1.0
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
            :return: A dictionary containing the completion response.
            """

            # Create a list of tuples, each containing all the parameters for 
            # a call to _generate_completion
            args = (model, prompt, input, output, max_tokens, temperature, top_p)

            # Run _generate_completion
            choices = self._generate_completion(*args)

            return choices
        
        def _generate_completion(self, model, prompt, input, output, max_tokens, temperature, top_p):
            """
            Function to generate a single completion. 
            """

            # Make a prediction using the proxy.
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            payload_dict = {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            if input:
                payload_dict["input"] = input
            if output:
                payload_dict["output"] = output
            payload = json.dumps(payload_dict)

            response = requests.request(
                "POST", self.url + "/completions", headers=headers, data=payload
            )
            # If the request was successful, print the proxies.
            if response.status_code == 200:
                ret = response.json()
                return ret
            else:
                # Check if there is a json body in the response. Read thhether the API response should be streamedat in,
                # print out the error field in the json body, and raise an exception.
                err = ""
                try:
                    err = response.json()["error"]
                except:
                    pass
                raise ValueError("Could not make prediction. " + err)

        def list_models(self) -> List[str]:
            # Get the list of current models.
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }
    
            response = requests.request("GET", self.url + "/completions", headers=headers)

            return list(response.json())
        

    class Chat:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

            self.completions = self.Completions(self.api_key, self.url)

        class Completions:
            """
            Chat API
            """
            def __init__(self, api_key, url):
                self.api_key = api_key
                self.url = url

            def create(
                self, 
                model: str,
                messages: List[Dict[str, Any]],
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
                :param stream: Option to stream the API response 
                :return: A dictionary containing the chat response.
                """

                # Create a list of tuples, each containing all the parameters for 
                # a call to _generate_chat
                args = (model, messages, input, output, max_tokens, temperature, top_p, stream)

                # Run _generate_chat
                choices = self._generate_chat(*args)

                return choices

            def _generate_chat(self, model, messages, input, output, max_tokens, temperature, top_p, stream):
                """
                Function to generate a single chat response.
                """
                
                def return_dict(url, headers, payload):
                    response = requests.request(
                        "POST", url + "/chat/completions", headers=headers, data=payload
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

                def stream_generator(url, headers, payload, stream):
                    with requests.post(
                        url + "/chat/completions", headers=headers, data=payload, stream=stream
                    ) as response:
                        response.raise_for_status()

                        for line in response.iter_lines():
                            if line:
                                decoded_line = line.decode("utf-8")
                                formatted_return = "{" + (decoded_line.replace('data', '"data"', 1)) + "}"
                                try:
                                    dict_return = json.loads(formatted_return)
                                except json.decoder.JSONDecodeError:
                                    pass
                                else:
                                    try:
                                        dict_return["data"]["choices"][0]["delta"]["content"]
                                    except KeyError:
                                        pass
                                    else:
                                        yield dict_return

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.api_key
                    }

                for message in messages:
                    if type(message["content"]) == list:
                        for entry in message["content"]:
                            if entry["type"] == "image_url":
                                image_data = entry["image_url"]["url"]
                                if stream == True:
                                    raise ValueError("Streaming is not currently supported when using vision.")
                                else:
                                    image_url_check = urllib.parse.urlparse(image_data)
                                    if os.path.exists(image_data):
                                        with open(image_data, "rb") as image_file:
                                            image_input = base64.b64encode(image_file.read()).decode("utf-8")

                                    elif re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", image_data):
                                        if base64.b64encode(base64.b64decode(image_data)).decode('utf-8') == image_data:
                                            image_input = image_data

                                    elif image_url_check.scheme in ("http", "https", "ftp"):
                                        urllib.request.urlretrieve(image_data, "temp.jpg")
                                        temp_image = "temp.jpg"
                                        with open(temp_image, "rb") as image_file:
                                            image_input = base64.b64encode(image_file.read()).decode("utf-8")
                                        os.remove(temp_image)
                                    else:
                                        raise ValueError("Please enter a valid base64 encoded image, image file, or image url.")
                                    
                                    image_data_uri = "data:image/jpeg;base64," + image_input
                                    entry["image_url"]["url"] = image_data_uri
                            elif entry["type"] == "text":
                                continue   

                payload_dict = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "stream": stream
                }
                
                if input:
                    payload_dict["input"] = input
                if output:
                    if stream == True:
                        raise ValueError("Factuality and toxicity checks are not supported when streaming is enabled.")
                    else:
                        payload_dict["output"] = output

                payload = json.dumps(payload_dict)

                if stream == True:
                    return stream_generator(self.url, headers, payload, stream)

                else:
                    return return_dict(self.url, headers, payload)

            def list_models(self) -> Dict[str, List[str]]:
                # Commented out parts are there for easier fix when
                # functionality for this call on chat endpoint added
                model_list = {
                    "Chat Models": [
                        "deepseek-coder-6.7b-instruct", 
                        "Hermes-2-Pro-Mistral-7B",
                        "Hermes-2-Pro-Llama-3-8B",
                        "llama-3-sqlcoder-8b",
                        "Neural-Chat-7B",
                    ],
                    "Vision Models": [
                        "llava-1.5-7b-hf"
                    ]
                }

                # Get the list of current models.
                # headers = {
                #         "Content-Type": "application/json",
                #         "Authorization": "Bearer " + self.api_key
                #         }
        
                # response = requests.request("GET", self.url + "/chat/completions", headers=headers)

                # return list(response.json())
                return model_list
        

    class Embeddings:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def create(
                self,
                model: str,
                input: List[Dict[str, str]],
        ) -> Dict[str, Any]:
            
            """
            Creates an embeddings request to the Prediction Guard /embeddings API
            
            :param model: Model to use for embeddings
            :param input: List of dictionaries containing input data with text and image keys.
            :result: 
            """

            # Create a list of tuples, each containing all the parameters for 
            # a call to _generate_translation
            args = (model, input)

            # Run _generate_embeddings
            choices = self._generate_embeddings(*args)
            return choices

        def _generate_embeddings(self, model, input):

            """
            Function to generate an embeddings response.
            """

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            inputs = []
            for item in input:
                item_dict = {}
                if "text" in item.keys():
                    item_dict["text"] = item["text"]
                if "image" in item.keys():
                    image_url_check = urllib.parse.urlparse(item["image"])

                    if os.path.exists(item["image"]):
                        with open(item["image"], "rb") as image_file:
                            image_input = base64.b64encode(image_file.read()).decode("utf-8")

                    elif re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", item["image"]):
                        if base64.b64encode(base64.b64decode(item["image"])).decode('utf-8') == item["image"]:
                            image_input = item["image"]

                    elif image_url_check.scheme in ("http", "https", "ftp"):
                        urllib.request.urlretrieve(item["image"], "temp.jpg")
                        temp_image = "temp.jpg"
                        with open(temp_image, "rb") as image_file:
                            image_input = base64.b64encode(image_file.read()).decode("utf-8")
                        os.remove(temp_image)

                    else:
                        raise ValueError("Please enter a valid base64 encoded image, image file, or image url.")

                    item_dict["image"] = image_input
                
                inputs.append(item_dict)

            payload_dict = {
                "model": model,
                "input": inputs
            }

            payload = json.dumps(payload_dict)
            response = requests.request(
                "POST", self.url + "/embeddings", headers=headers, data=payload
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
                raise ValueError("Could not generate embeddings. " + err)
            
        def list_models(self) -> Dict[str, List[str]]:

            model_list = [
                "bridgetower-large-itm-mlm-itc"
            ]

            return model_list


    class Translate:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def create(
                self,
                text: str,
                source_lang: str,
                target_lang: str
        ) -> Dict[str, Any]:

            """
            Creates a translate request to the Prediction Guard /translate API.

            :param text: The text to be translated.
            :param source_lang: The language the text is currently in.
            :param target_lang: The language the text will be translated to.
            :result: A dictionary containing the translate response.
            """

            # Create a list of tuples, each containing all the parameters for 
            # a call to _generate_translation
            args = (text, source_lang, target_lang)

            # Run _generate_translation
            choices = self._generate_translation(*args)
            return choices

        def _generate_translation(self, text, source_lang, target_lang):
            """
            Function to generate a translation response.
            """

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            payload_dict = {
                "text": text,
                "source_lang": source_lang,
                "target_lang": target_lang
            }
            payload = json.dumps(payload_dict)
            response = requests.request(
                "POST", self.url + "/translate", headers=headers, data=payload
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
                raise ValueError("Could not make translation. " + err)
            

    class Factuality:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def check(self, reference: str,
                        text: str) -> Dict[str, Any]:
            """
            Creates a factuality checking request for the Prediction Guard /factuality API.

            :param reference: The reference text used to check for factual consistency.
            :param text: The text to check for factual consistency.
            """

            # Run _generate_score
            choices = self._generate_score(reference, text)
            return choices
        
        def _generate_score(self, reference, text):
            """
            Function to generate a single factuality score. 
            """

            # Make a prediction using the proxy.
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            payload_dict = {
                "reference": reference,
                "text": text
            }
            payload = json.dumps(payload_dict)
            response = requests.request(
                "POST", self.url + "/factuality", headers=headers, data=payload
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
            

    class Toxicity:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def check(self, text: str) -> Dict[str, Any]:
            """
            Creates a toxicity checking request for the Prediction Guard /toxicity API.

            :param text: The text to check for toxicity.
            """

            # Run _generate_score
            choices = self._generate_score(text)
            return choices
        
        def _generate_score(self, text):
            """
            Function to generate a single toxicity score. 
            """

            # Make a prediction using the proxy.
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            payload_dict = {"text": text}
            payload = json.dumps(payload_dict)
            response = requests.request(
                "POST", self.url + "/toxicity", headers=headers, data=payload
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
                

    class Pii:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def check(self, prompt: str, replace: bool, replace_method: Optional[str] = "random") -> Dict[str, Any]:
            """
            Creates a PII checking request for the Prediction Guard /PII API.

            :param text: The text to check for PII.
            :param replace: Whether to replace PII if it is present.
            :param replace_method: Method to replace PII if it is present.
            """

            # Run _check_pii
            choices = self._check_pii(prompt, replace, replace_method)
            return choices

        def _check_pii(self, prompt, replace, replace_method):
            """
            Function to check for PII.
            """

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            payload_dict = {
                "prompt": prompt,
                "replace": replace,
                "replace_method": replace_method
                }

            payload = json.dumps(payload_dict)
            response = requests.request(
                "POST", self.url + "/PII", headers=headers, data=payload
            )

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
                raise ValueError("Could not check PII. " + err)
            

    class Injection:
        def __init__(self, api_key, url):
            self.api_key = api_key
            self.url = url

        def check(self, prompt: str, detect: bool) -> Dict[str, Any]:
            """
            Creates a prompt injection check request in the Prediction Guard /injection API.

            :param prompt: Prompt to test for injection.
            :param detect: Whether to detect the prompt for injections.
            """

            # Run _check_injection
            choices = self._check_injection(prompt, detect)
            return choices
        
        def _check_injection(self, prompt, detect):
            """
            Function to check if prompt is a prompt injection.
            """

            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self.api_key
                }

            payload = {
                "prompt": prompt,
                "detect": detect
            }

            payload = json.dumps(payload)

            response = requests.request(
                "POST", self.url + "/injection", headers=headers, data=payload
            )

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
                raise ValueError("Could not check for injection. " + err)