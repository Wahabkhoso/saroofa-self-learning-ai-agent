def test_health_endpoint(api_client):
    response = api_client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_calculation_end_to_end(api_client):
    response = api_client.post("/api/chat", json={"message": "Calculate 5 * 6"})
    assert response.status_code == 200
    data = response.json()
    assert data["evaluation"]["status"] == "SUCCESS"
    assert data["tool_calls"][0]["output"] == 30.0


def test_task_appears_in_task_list(api_client):
    api_client.post("/api/chat", json={"message": "Calculate 3 + 3"})
    response = api_client.get("/api/tasks")
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) >= 1
    assert any(t["user_input"] == "Calculate 3 + 3" for t in tasks)


def test_get_single_task(api_client):
    chat_resp = api_client.post("/api/chat", json={"message": "Calculate 9 - 4"})
    task_id = chat_resp.json()["task_id"]

    response = api_client.get(f"/api/tasks/{task_id}")
    assert response.status_code == 200
    assert response.json()["id"] == task_id


def test_get_nonexistent_task_returns_404(api_client):
    response = api_client.get("/api/tasks/does-not-exist")
    assert response.status_code == 404


def test_memory_populated_after_chat(api_client):
    api_client.post("/api/chat", json={"message": "Calculate 10 / 2"})
    response = api_client.get("/api/memory")
    assert response.status_code == 200
    experiences = response.json()
    assert len(experiences) >= 1


def test_feedback_endpoint(api_client):
    chat_resp = api_client.post("/api/chat", json={"message": "Calculate 7 + 7"})
    task_id = chat_resp.json()["task_id"]
    memory = api_client.get("/api/memory").json()
    experience_id = memory[0]["id"]

    response = api_client.post(
        "/api/feedback",
        json={"task_id": task_id, "feedback_type": "THUMBS_UP", "experience_id": experience_id},
    )
    assert response.status_code == 200
    assert response.json()["feedback_type"] == "THUMBS_UP"


def test_feedback_for_unknown_task_returns_404(api_client):
    response = api_client.post(
        "/api/feedback", json={"task_id": "nope", "feedback_type": "THUMBS_UP"}
    )
    assert response.status_code == 404


def test_stats_endpoint_reflects_activity(api_client):
    api_client.post("/api/chat", json={"message": "Calculate 1 + 1"})
    response = api_client.get("/api/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_tasks"] >= 1
    assert data["memory"]["total_experiences"] >= 1


def test_knowledge_upload_and_list(api_client):
    files = {"file": ("notes.txt", b"The sky is blue. Water boils at 100C.", "text/plain")}
    response = api_client.post("/api/knowledge/upload", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["document"]["chunk_count"] >= 1

    listing = api_client.get("/api/knowledge")
    assert listing.status_code == 200
    assert any(d["filename"] == "notes.txt" for d in listing.json())


def test_knowledge_upload_rejects_bad_extension(api_client):
    files = {"file": ("virus.exe", b"MZ", "application/octet-stream")}
    response = api_client.post("/api/knowledge/upload", files=files)
    assert response.status_code == 415


def test_rate_limit_engages_after_threshold(api_client, monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    api_client.post("/api/chat", json={"message": "Calculate 1 + 1"})
    api_client.post("/api/chat", json={"message": "Calculate 1 + 2"})
    third = api_client.post("/api/chat", json={"message": "Calculate 1 + 3"})
    assert third.status_code == 429
