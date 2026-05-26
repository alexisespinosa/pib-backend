from typing import Any, List
from model.person_model import Person, FaceEmbedding
from app.app import db


MAX_EMBEDDINGS_PER_PERSON = 20


def get_all_persons() -> List[Person]:
    return Person.query.all()


def get_person(person_id: str) -> Person:
    return Person.query.filter(Person.person_id == person_id).one()


def get_person_by_name(name: str) -> Person:
    return Person.query.filter(Person.name == name).one()


def create_person(person_dto: Any) -> Person:
    person = Person(name=person_dto["name"])
    db.session.add(person)
    db.session.flush()
    return person


def delete_person(person_id: str) -> None:
    db.session.delete(get_person(person_id))
    db.session.flush()


def get_embeddings(person_id: str) -> List[FaceEmbedding]:
    person = get_person(person_id)
    return FaceEmbedding.query.filter(
        FaceEmbedding.person_id == person.person_id
    ).all()


def add_embedding(person_id: str, embedding_dto: Any) -> FaceEmbedding:
    person = get_person(person_id)
    existing = FaceEmbedding.query.filter(
        FaceEmbedding.person_id == person.person_id
    ).order_by(FaceEmbedding.id.asc()).all()
    while len(existing) >= MAX_EMBEDDINGS_PER_PERSON:
        db.session.delete(existing.pop(0))
    embedding = FaceEmbedding(
        embedding=embedding_dto["embedding"],
        person_id=person.person_id,
    )
    db.session.add(embedding)
    db.session.flush()
    return embedding


def delete_embedding(person_id: str, embedding_id: str) -> None:
    embedding = FaceEmbedding.query.filter(
        FaceEmbedding.embedding_id == embedding_id,
        FaceEmbedding.person_id == person_id,
    ).one()
    db.session.delete(embedding)
    db.session.flush()
