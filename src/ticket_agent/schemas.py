from pydantic import BaseModel, ConfigDict, Field
from typing import Literal



class PredictionRequest(BaseModel):
    title: str
    description: str


class PredictionResponse(BaseModel):
    category: str
    confidence: float = Field(ge=0, le=1)
    model_version: str


class AgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["reply", "ask_clarification", "escalate"] = Field(
        description="Recommended next action for the support workflow.",
    )
    draft_reply: str = Field(
        description="Customer-facing draft in the customer's language.",
    )
    source_ids: list[Literal["KB-001", "KB-002"]] = Field(
        description=(
            "Exact IDs of relevant articles retrieved for this ticket. "
            "Use bare IDs without brackets or titles. "
            "Use an empty list when no retrieved article is relevant."
        )
    )
    reason: str = Field(
        description="One-sentence operational explanation for the support team.",
    )