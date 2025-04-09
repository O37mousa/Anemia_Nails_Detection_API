from pydantic import BaseModel, Field
from typing import Literal

class Request(BaseModel):
    Age: int = Field(description="Age of the person")
    Gender: str = Field(description="Gender of the person (e.g., 'M' or 'F')")
