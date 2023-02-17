# Prediction Guard - Python Client

This package provides functionality developed to simplify interfacing with [Prediction Guard](https://www.predictionguard.com/) in Python 3.

## Documentation

See the [API documentation](https://docs.predictionguard.com/python).

## Installation

The package can be installed with `pip`:

```bash
pip install --upgrade predictionguard
```

### Requirements

- Python 3.6+

## Quick Start

To use this library, you must have an access token and specify it as a string when creating the `pg.Client` object. Access tokens can be created through the Prediction Guard platform (link coming soon). This is a basic example that:

1. Instantiates a Prediction Guard client
2. Defines some example model input/ output
3. Creates a prediction proxy endpoint
4. Uses the endpoint to make a prediction

```python
import predictionguard as pg

# Initialize a Prediction Guard client.
client = Client(token=<your access token>)

# Create some examples illustrating the kind of predictions you
# want to make (domain/ use case specific).
examples = [
 {
   "input": {
     "phrase": "I'm so excited about Prediction Guard. It's gonna be awesome!"
   },
   "output": {
     "sentiment": "POS"
   }
 },
 {
   "input": {
     "phrase": "AI development without Prediction Guard is bad. It's really terrible."
   },
   "output": {
     "sentiment": "NEG"
   }
 }
]

# Create a prediction "proxy." This proxy will save your examples, evaluate
# SOTA models to find the best one for your use case, and expose the best model
# at an endpoint corresponding to the proxy.
client.create_proxy(task='sentiment', name='my-sentiment-proxy', examples=examples)

# Now your ready to start getting reliable, future proof predictions. No fuss!
client.predict(name='test-client-sentiment4', data={
 "phrase": "Isn't this great! I'm so happy I'm using Prediction Guard"
})
```