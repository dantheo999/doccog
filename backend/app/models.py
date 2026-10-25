import hashlib
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic_core import PydanticCustomError


NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExampleExtraction(StrictModel):
    extraction_class: Annotated[NonEmpty, Field(max_length=100)]
    extraction_text: str = Field(min_length=1, max_length=12_000)
    attributes: dict[str, JsonValue] = Field(default_factory=dict, max_length=30)


class ExampleSource(StrictModel):
    name: str = Field(min_length=1, max_length=1_024)
    document_id: Annotated[NonEmpty, Field(max_length=100)]
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    verified: bool = Field(strict=True)

    @model_validator(mode="after")
    def check_interval(self):
        if self.end <= self.start:
            raise ValueError("Source interval must end after its start.")
        return self


class Example(StrictModel):
    text: str = Field(min_length=1, max_length=500_000)
    extractions: list[ExampleExtraction] = Field(min_length=1, max_length=100)
    source: ExampleSource | None = None

    @model_validator(mode="after")
    def check_spans(self):
        if self.source and self.source.verified:
            if (self.source.end - self.source.start != len(self.text)
                    or self.source.text_sha256 != hashlib.sha256(self.text.encode()).hexdigest()):
                raise ValueError("Verified source metadata must match the unchanged example text.")
        for index, extraction in enumerate(self.extractions):
            if extraction.extraction_text not in self.text:
                raise PydanticCustomError(
                    "extraction_text_not_in_example",
                    "extractions[{index}].extraction_text must be an exact substring of "
                    "its example text, including spaces and line breaks",
                    {"index": index},
                )
        return self


class GeneratedRecipe(StrictModel):
    instructions: Annotated[NonEmpty, Field(max_length=16_000)]
    examples: list[Example] = Field(min_length=1, max_length=5)


class Recipe(GeneratedRecipe):
    id: Annotated[NonEmpty, Field(max_length=100)]
    version: int = Field(ge=1)
    goal: Annotated[NonEmpty, Field(max_length=4_000)]
    source_name: str = Field(max_length=1_024)
    created_at: str = Field(max_length=100)
    source_characters_used: int = Field(default=0, ge=0)
    source_characters_total: int = Field(default=0, ge=0)


class GenerateRequest(StrictModel):
    document_id: NonEmpty | None = None
    document_ids: list[NonEmpty] | None = Field(default=None, min_length=1, max_length=5)
    goal: Annotated[NonEmpty, Field(max_length=4_000)]

    @model_validator(mode="after")
    def check_sources(self):
        if (self.document_id is None) == (self.document_ids is None):
            raise ValueError("Provide document_id or document_ids, not both.")
        if self.document_ids is not None and len(set(self.document_ids)) != len(self.document_ids):
            raise ValueError("Exemplar document IDs must be distinct.")
        return self

    @property
    def source_ids(self) -> list[str]:
        if self.document_ids is not None:
            return self.document_ids
        assert self.document_id is not None
        return [self.document_id]


class SelectedExample(StrictModel):
    passage_ids: list[NonEmpty] = Field(min_length=1, max_length=1)
    extractions: list[ExampleExtraction] = Field(min_length=1, max_length=100)


class SelectedRecipe(StrictModel):
    instructions: Annotated[NonEmpty, Field(max_length=16_000)]
    examples: list[SelectedExample] = Field(min_length=1, max_length=5)


class SelectedOptimization(SelectedRecipe):
    examples: list[SelectedExample] = Field(min_length=0, max_length=5)
    reasons: list[Annotated[NonEmpty, Field(max_length=2_000)]] = Field(min_length=1, max_length=10)


class ExtractionOptions(StrictModel):
    chunk_size: int = Field(default=6000, ge=500, le=20_000, strict=True)
    max_workers: int = Field(default=2, ge=1, le=16, strict=True)
    extraction_passes: int = Field(default=1, ge=1, le=5, strict=True)
    context_window_chars: int = Field(default=0, ge=0, le=6000, strict=True)


class ExtractRequest(StrictModel):
    document_id: NonEmpty
    recipe: Recipe
    options: ExtractionOptions = Field(default_factory=ExtractionOptions)


class ExtractionItem(StrictModel):
    id: str
    extraction_class: str
    extraction_text: str
    attributes: dict[str, JsonValue]
    start: int | None
    end: int | None
    alignment_status: str | None
    grounding: str


class ExtractionResult(StrictModel):
    document_id: str
    text: str
    recipe_id: str
    recipe_version: int
    extractions: list[ExtractionItem]
    elapsed_seconds: float
    options: ExtractionOptions


class OptimizeRequest(StrictModel):
    document_id: NonEmpty
    recipe: Recipe
    result: ExtractionResult


class OptimizationProposal(StrictModel):
    base_recipe_id: NonEmpty
    base_recipe_version: int = Field(ge=1, strict=True)
    candidate: Recipe
    reasons: list[Annotated[NonEmpty, Field(max_length=2_000)]] = Field(min_length=1, max_length=10)
