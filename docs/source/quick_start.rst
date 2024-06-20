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


   # You can set you Prediction Guard API Key as an env variable,
   # or when creating the client object
   os.environ["PREDICTIONGUARD_API_KEY"]

   client = PredictionGuard(
       api_key="<your Prediction Guard API Key>"
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
       model="Hermes-2-Pro-Llama-3-8B",
       messages=messages,
       max_tokens=100
   )

   print(json.dumps(
       result,
       sort_keys=True,
       indent=4,
       separators=(',', ': ')
   ))

