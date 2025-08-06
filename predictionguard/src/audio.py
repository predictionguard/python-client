from typing import Any, Dict, List, Optional

import requests

from ..version import __version__


class Audio:
    """
    Audio generates a response based on audio data.

    Usage::

        import os
        import json

        from predictionguard import PredictionGuard

        # Set your Prediction Guard token and url as an environmental variable.
        os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"
        os.environ["PREDICTIONGUARD_URL"] = "<url>"

        # Or set your Prediction Guard token and url when initializing the PredictionGuard class.
        client = PredictionGuard(
            api_key="<api_key>",
            url="<url>"
        )

        result = client.audio.transcriptions.create(
            model="base",
            file="sample_audio.wav"
        )

        print(json.dumps(
            response,
            sort_keys=True,
            indent=4,
            separators=(",", ": ")
        ))
    """

    def __init__(self, api_key, url, timeout):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout

        self.transcriptions: AudioTranscriptions = AudioTranscriptions(self.api_key, self.url, self.timeout)

class AudioTranscriptions:
    def __init__(self, api_key, url, timeout):
        self.api_key = api_key
        self.url = url
        self.timeout = timeout

    def create(
        self,
        model: str,
        file: str,
        language: Optional[str] = "auto",
        temperature: Optional[float] = 0.0,
        prompt: Optional[str] = "",
        timestamp_granularities: Optional[List[str]] = None,
        diarization: Optional[bool] = False,
        response_format: Optional[str] = "json",
        toxicity: Optional[bool] = False,
        pii: Optional[str] = "",
        replace_method: Optional[str] = "",
        injection: Optional[bool] = False,
    ) -> Dict[str, Any]:
        """
        Creates an audio transcription request to the Prediction Guard /audio/transcriptions API

        :param model: The model to use
        :param file: Audio file to be transcribed
        :param language: The language of the audio file
        :param temperature: The temperature parameter for model transcription
        :param prompt: A prompt to assist in transcription styling
        :param timestamp_granularities: The timestamp granularities to populate for this transcription
        :param diarization: Whether to diarize the audio
        :param response_format: The response format to use
        :param toxicity: Whether to check for output toxicity
        :param pii: Whether to check for or replace pii
        :param replace_method: Replace method for any PII that is present.
        :param injection: Whether to check for prompt injection
        :result: A dictionary containing the transcribed text.
        """

        # Create a list of tuples, each containing all the parameters for
        # a call to _transcribe_audio
        args = (
            model,
            file,
            language,
            temperature,
            prompt,
            timestamp_granularities,
            diarization,
            response_format,
            pii,
            replace_method,
            injection,
            toxicity,
        )

        # Run _transcribe_audio
        choices = self._transcribe_audio(*args)
        return choices

    def _transcribe_audio(
            self,
            model,
            file,
            language,
            temperature,
            prompt,
            timestamp_granularities,
            diarization,
            response_format,
            pii,
            replace_method,
            injection,
            toxicity,
    ):
        """
        Function to transcribe an audio file.
        """

        headers = {
            "Authorization": "Bearer " + self.api_key,
            "User-Agent": "Prediction Guard Python Client: " + __version__,
            "Toxicity": str(toxicity),
            "Pii": str(pii),
            "Replace-Method": str(replace_method),
            "Injection": str(injection)
        }

        if timestamp_granularities:
            if diarization and "segment" in timestamp_granularities:
                raise ValueError(
                    "Timestamp granularities cannot be set to "
                    "`segments` when using diarization."
                )

            if response_format != "verbose_json":
                raise ValueError(
                    "Response format must be set to `verbose_json` "
                    "when using timestamp granularities."
                )

        if diarization and response_format != "verbose_json":
            raise ValueError(
                "Response format must be set to `verbose_json` "
                "when using diarization."
            )

        with open(file, "rb") as audio_file:
            files = {"file": (file, audio_file, "audio/wav")}
            data = {
                "model": model,
                "language": language,
                "temperature": temperature,
                "prompt": prompt,
                "timestamp_granularities[]": timestamp_granularities,
                "diarization": str(diarization).lower(),
                "response_format": response_format,
            }

            response = requests.request(
                "POST", self.url + "/audio/transcriptions", headers=headers, files=files, data=data, timeout=self.timeout
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
