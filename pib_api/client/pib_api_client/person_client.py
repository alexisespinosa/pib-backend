from urllib.request import Request
from typing import Any
from pib_api_client import send_request, URL_PREFIX
import json

PERSONS_URL = URL_PREFIX + "/person"
PERSON_URL = URL_PREFIX + "/person/%s"
EMBEDDINGS_URL = URL_PREFIX + "/person/%s/embedding"


def get_all_persons() -> (bool, dict[str, Any]):
    request = Request(PERSONS_URL, method="GET")
    return send_request(request)


def create_person(name: str) -> (bool, dict[str, Any]):
    request = Request(
        PERSONS_URL,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"name": name}).encode("UTF-8"),
    )
    return send_request(request)


def get_embeddings(person_id: str) -> (bool, dict[str, Any]):
    request = Request(EMBEDDINGS_URL % person_id, method="GET")
    return send_request(request)


def add_embedding(person_id: str, embedding: list[float]) -> (bool, dict[str, Any]):
    request = Request(
        EMBEDDINGS_URL % person_id,
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps({"embedding": json.dumps(embedding)}).encode("UTF-8"),
    )
    return send_request(request)
