import os

from predictionguard import PredictionGuard


def test_documents_extract_success():
    test_client = PredictionGuard()

    response = test_client.documents.extract.create(
        file="fixtures/test_pdf.pdf"
    )

    assert len(response["contents"]) > 0