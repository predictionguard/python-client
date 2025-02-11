import os

import pytest

from predictionguard import PredictionGuard

## TODO: finalize fail tests
def test_documents_extract_docling_success():
    test_client = PredictionGuard()

    response = test_client.documents.extract(
        file="fixtures/test_pdf.pdf",
        engine="docling"
    )

    assert len(response["content"]) > 0


def test_documents_extract_markitdown_success():
    test_client = PredictionGuard()

    response = test_client.documents.extract(
        file="fixtures/test_pdf.pdf",
        engine="markitdown"
    )

    assert len(response["content"]) > 0


def test_documents_extract_failure():
    test_client = PredictionGuard()

    file_error = "Factuality and toxicity checks are not supported when streaming is enabled.".replace(
        "\n", ""
    )

    with pytest.raises(ValueError, match=file_error):
        test_client.documents.extract(
            file="fixtures/test_csv.csv",
            engine="docling"
        )


def test_documents_extract_engine_failure():
    test_client = PredictionGuard()

    engine_error = ""

    with pytest.raises(ValueError, match=engine_error):
        test_client.documents.extract(
                file="fixtures/test_pdf.pdf",
                engine="pizza_pie"
        )