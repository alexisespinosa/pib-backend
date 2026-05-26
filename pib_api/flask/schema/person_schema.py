from model.person_model import Person, FaceEmbedding
from schema.sql_auto_with_camel_case_schema import SQLAutoWithCamelCaseSchema


class PersonSchemaSQLAutoWith(SQLAutoWithCamelCaseSchema):
    class Meta:
        model = Person
        exclude = ("id",)


class FaceEmbeddingSchemaSQLAutoWith(SQLAutoWithCamelCaseSchema):
    class Meta:
        model = FaceEmbedding
        exclude = ("id", "person_id")


person_schema = PersonSchemaSQLAutoWith()
persons_schema = PersonSchemaSQLAutoWith(many=True)
create_person_schema = PersonSchemaSQLAutoWith(only=("name",))

face_embedding_schema = FaceEmbeddingSchemaSQLAutoWith()
face_embeddings_schema = FaceEmbeddingSchemaSQLAutoWith(many=True)
create_embedding_schema = FaceEmbeddingSchemaSQLAutoWith(only=("embedding",))
