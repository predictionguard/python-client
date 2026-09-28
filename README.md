# Prediction Guard - Python Client

> [!WARNING]
> **This SDK is deprecated and no longer maintained.** Some features are broken or missing, and no further updates will be released. Existing releases remain installable from PyPI, but you should migrate to an OpenAI-compatible or Anthropic-compatible SDK as described below.

## Migrating

The Prediction Guard API is compatible with both OpenAI-style and Anthropic-style clients. Use whichever SDK matches the functionality you need, pointed at the Prediction Guard API with your existing API key.

### OpenAI-compatible SDK

```bash
pip install openai
```

```python
import os

from openai import OpenAI

client = OpenAI(
    api_key=os.environ["PREDICTIONGUARD_API_KEY"],
    base_url="https://api.predictionguard.com",
)

response = client.chat.completions.create(
    model="<model-name>",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(response.choices[0].message.content)
```

### Anthropic-compatible SDK

```bash
pip install anthropic
```

```python
import os

from anthropic import Anthropic

client = Anthropic(
    api_key=os.environ["PREDICTIONGUARD_API_KEY"],
    base_url="https://api.predictionguard.com",
)

message = client.messages.create(
    model="<model-name>",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(message.content[0].text)
```

For the full list of endpoints and models, see the [Prediction Guard documentation](https://docs.predictionguard.com).

## Legacy documentation

The [API documentation](https://predictionguard.github.io/python-client/) for this SDK remains available for reference but will not be updated.
