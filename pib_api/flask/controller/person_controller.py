from service import person_service
from schema.person_schema import (
    person_schema,
    persons_schema,
    create_person_schema,
    face_embedding_schema,
    face_embeddings_schema,
    create_embedding_schema,
)
from flask import jsonify, request, Blueprint

bp = Blueprint("person_controller", __name__)


@bp.route("", methods=["GET"])
def get_all_persons():
    persons = person_service.get_all_persons()
    return jsonify({"persons": persons_schema.dump(persons)})


@bp.route("", methods=["POST"])
def create_person():
    person_dto = create_person_schema.load(request.json)
    person = person_service.create_person(person_dto)
    return person_schema.dump(person), 201


@bp.route("/<string:person_id>", methods=["GET"])
def get_person(person_id: str):
    person = person_service.get_person(person_id)
    return person_schema.dump(person)


@bp.route("/<string:person_id>", methods=["DELETE"])
def delete_person(person_id: str):
    person_service.delete_person(person_id)
    return "", 204


@bp.route("/<string:person_id>/embedding", methods=["GET"])
def get_embeddings(person_id: str):
    embeddings = person_service.get_embeddings(person_id)
    return jsonify({"embeddings": face_embeddings_schema.dump(embeddings)})


@bp.route("/<string:person_id>/embedding", methods=["POST"])
def add_embedding(person_id: str):
    embedding_dto = create_embedding_schema.load(request.json)
    embedding = person_service.add_embedding(person_id, embedding_dto)
    return face_embedding_schema.dump(embedding), 201


@bp.route("/<string:person_id>/embedding/<string:embedding_id>", methods=["DELETE"])
def delete_embedding(person_id: str, embedding_id: str):
    person_service.delete_embedding(person_id, embedding_id)
    return "", 204
