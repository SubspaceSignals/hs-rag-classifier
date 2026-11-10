"""Request and response models for POST /classify and GET /health."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_DESCRIPTION_CHARS = 10_000  # refused above this; the pipeline itself truncates at 1,000
REVIEW_REASONS = ("operating_point", "out_of_distribution")
# Invented for the docs; not taken from any data split.
EXAMPLE_DESCRIPTION = (
    "Child's wooden rocking horse with a padded fabric saddle and rope reins, for indoor play"
)


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"description": EXAMPLE_DESCRIPTION}]}
    )

    description: str = Field(min_length=1, max_length=MAX_DESCRIPTION_CHARS)

    @field_validator("description")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("description is empty or whitespace only")
        return v


class Candidate(BaseModel):
    cn8: str
    confidence: float


class OOD(BaseModel):
    similarity: float
    threshold: float
    flagged: bool
    nearest_ruling_ref: str
    message: str | None = None


class Citation(BaseModel):
    """The pipeline's citation check of the rationale's quote against the chosen line."""

    model_config = ConfigDict(populate_by_name=True)

    quote: str | None = None
    passed: bool = Field(serialization_alias="pass", validation_alias="pass")


class Precedent(BaseModel):
    """The nearest train ruling (full index) to the description."""

    ref: str
    cn8: str
    similarity: float
    agrees_with_prediction: bool = Field(
        description="True when the nearest ruling's cn8 equals the predicted top-1 cn8",
        examples=[True],
    )


class Provenance(BaseModel):
    config_hash: str
    operating_point_rule_sha: str
    ood_rule_sha: str
    git_commit: str


class ClassifyResponse(BaseModel):
    request_id: str
    cn8: str | None
    confidence: float
    top3: list[Candidate]
    needs_review: bool
    review_reasons: list[Literal["operating_point", "out_of_distribution"]]
    ood: OOD
    rationale: str | None
    citation: Citation | None
    precedent: Precedent
    provenance: Provenance
    advisory: str


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None = None
