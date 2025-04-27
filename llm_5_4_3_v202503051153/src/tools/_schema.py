from pydantic import BaseModel, Field
from typing import Annotated


class RetrievalQAToolSchema(BaseModel):
    input: Annotated[
        str,
        Field(
            description="Should be a detailed search query in natural language to be answered by a retriever."
        ),
    ]
