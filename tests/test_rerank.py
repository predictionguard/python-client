import os

from predictionguard import PredictionGuard


def test_rerank_create():
    test_client = PredictionGuard()

    response = test_client.rerank.create(
        model=os.environ["TEST_RERANK_MODEL"],
        query="What is Deep Learning?",
        documents=[
            "Deep Learning is pizza.",
            "Deep Learning is not pizza."
        ],
        return_documents=True,
    )

    assert len(response) > 0
    assert type(response["results"][0]["index"]) is int
    assert type(response["results"][0]["relevance_score"]) is float
    assert type(response["results"][0]["text"]) is str


def test_rerank_list():
    test_client = PredictionGuard()

    response = test_client.rerank.list_models()

    assert len(response) > 0
    assert type(response[0]) is str
