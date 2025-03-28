import json

import requests
from typing import Any, Dict, Optional

from ..version import __version__


class Audio:
    """Audio generates a response based on audio data.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"

        client = PredictionGuard()

        result = client.audio.transcriptions.create(
            model="whisper-3-large-instruct", file=sample_audio.wav
        )

        print(json.dumps(result, sort_keys=True, indent=4, separators=(",", ": ")))
    """

    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

        self.transcriptions: AudioTranscriptions = AudioTranscriptions(self.api_key, self.url)

class AudioTranscriptions:
    def __init__(self, api_key, url):
        self.api_key = api_key
        self.url = url

    def create(
        self,
        model: str,
        file: str,
        language: Optional[str] = "auto",
        temperature: Optional[float] = 0.0,
        prompt: Optional[str] = "",
    ) -> Dict[str, Any]:
        """
        Creates a audio transcription request to the Prediction Guard /audio/transcriptions API

        :param model: The model to use
        :param file: Audio file to be transcribed
        :param language: The language of the audio file
        :param temperature: The temperature parameter for model transcription
        :param prompt: A prompt to assist in transcription styling
        :result: A dictionary containing the transcribed text.
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _transcribe_audio
        args = (model, file, language, temperature, prompt)

        # Run _transcribe_audio
        choices = self._transcribe_audio(*args)
        return choices

    def _transcribe_audio(self, model, file, language, temperature, prompt):
        """
        Function to transcribe an audio file.
        """

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
        }

        with open(file, "rb") as audio_file:
            files = {"file": (file, audio_file, "audio/wav")}
            data = {
                "model": model,
                "language": language,
                "temperature": temperature,
                "prompt": prompt,
                }

            response = requests.request(
                "POST", self.url + "/audio/transcriptions", headers=headers, files=files, data=data
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
            raise ValueError("Could not transcribe the audio file. " + err)