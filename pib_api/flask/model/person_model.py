from app.app import db
from model.util import generate_uuid


class Person(db.Model):

    __tablename__ = "person"

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(
        db.String(255), nullable=False, default=generate_uuid, unique=True
    )
    name = db.Column(db.String(255), nullable=False, unique=True)
    embeddings = db.relationship(
        "FaceEmbedding", backref="person", lazy=True, cascade="all,delete"
    )


class FaceEmbedding(db.Model):

    __tablename__ = "face_embedding"

    id = db.Column(db.Integer, primary_key=True)
    embedding_id = db.Column(
        db.String(255), nullable=False, default=generate_uuid, unique=True
    )
    embedding = db.Column(db.Text, nullable=False)
    person_id = db.Column(
        db.String(255), db.ForeignKey("person.person_id"), nullable=False
    )
