Quick Start
=================

To use this library, you must have an api key. You can set it two ways: as an environment variable name `PREDICTIONGUARD_API_KEY` or when creating the client object. API Keys can be acquired [here](https://mailchi.mp/predictionguard/getting-started). This is a basic example that:

1. Instantiates a Prediction Guard client
2. Defines some example model input/ output
3. Creates a request to the Prediction Guard API
4. Formats the API response in a clean, readable way.

.. code-block:: python

   import json
   import os

   from predictionguard import PredictionGuard


    # Set your Prediction Guard token and url as an environmental variable.
    os.environ["PREDICTIONGUARD_API_KEY"] = "<api key>"
    os.environ["PREDICTIONGUARD_URL"] = "<url>"

    # Or set your Prediction Guard token and url when initializing the PredictionGuard class.
    client = PredictionGuard(
        api_key=<api_key>,
        url=<url>
    )

   messages = [
       {
           "role": "system",
           "content": "You are a helpful chatbot that helps people learn."
       },
       {
           "role": "user",
           "content": "What is a good way to learn to code?"
       }
   ]

   result = client.chat.completions.create(
       model="Hermes-3-Llama-3.1-8B",
       messages=messages
   )

   print(json.dumps(
       result,
       sort_keys=True,
       indent=4,
       separators=(',', ': ')
   ))

