from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "crochet-pattern-converter",
    }


def test_valid_pdf_returns_mock_conversion(client: TestClient, valid_pdf: bytes) -> None:
    response = client.post(
        "/api/convert",
        files={"file": ("chart.pdf", valid_pdf, "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["mock"] is True
    assert payload["title"] == "Simple Scalloped Coaster"
    assert len(payload["instructions"]) == 5
    assert payload["pages"][0]["detectedSymbols"][0]["label"] == "ch"
    assert payload["pages"][0]["detectedSymbols"][0]["pageNumber"] == 1
    assert "not" in payload["message"].lower()


def test_missing_file_is_rejected(client: TestClient) -> None:
    response = client.post("/api/convert")

    assert response.status_code == 400
    assert "required" in response.json()["detail"].lower()


def test_non_pdf_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/convert",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )

    assert response.status_code == 415
    assert "only pdf" in response.json()["detail"].lower()


def test_invalid_pdf_signature_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/convert",
        files={"file": ("chart.pdf", b"not really a pdf", "application/pdf")},
    )

    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


def test_oversized_pdf_is_rejected(small_limit_client: TestClient) -> None:
    oversized_pdf = b"%PDF-1.4\n" + (b"x" * 64)
    response = small_limit_client.post(
        "/api/convert",
        files={"file": ("large.pdf", oversized_pdf, "application/pdf")},
    )

    assert response.status_code == 413
    assert "maximum upload size" in response.json()["detail"].lower()


def test_response_contains_frontend_contract(client: TestClient, valid_pdf: bytes) -> None:
    response = client.post(
        "/api/convert",
        files={"file": ("chart.pdf", valid_pdf, "application/pdf")},
    )

    payload = response.json()
    assert set(payload) == {
        "status",
        "mock",
        "title",
        "instructions",
        "pages",
        "averageConfidence",
        "message",
    }
    assert set(payload["instructions"][0]) == {"id", "row", "text", "confidence"}
    assert set(payload["pages"][0]) == {"pageNumber", "width", "height", "detectedSymbols"}
    assert set(payload["pages"][0]["detectedSymbols"][0]) == {
        "id",
        "label",
        "confidence",
        "boundingBox",
        "pageNumber",
    }


def test_cors_allows_local_frontend(client: TestClient) -> None:
    response = client.options(
        "/api/convert",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
