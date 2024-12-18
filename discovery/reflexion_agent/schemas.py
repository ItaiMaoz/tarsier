from typing import List
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    """Reflection on the answer."""
    missing_information: List[str] = Field(description="The missing information that is needed to improve the answer.")
    superflous_information: List[str] = Field(description="The superflous information that is not needed to improve the answer.")
    improvement_suggestions: List[str] = Field(description="The improvement suggestions to improve the answer.")


class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="~250 words detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the inital answer.")
    search_queries: List[str] = Field(description="1-3 search queries to research information and improve the answer and adress the reflection.")
