from dataclasses import dataclass
import hashlib
import logging
import re

from app.documents import ParsedDocument
from app.models import Example, ExampleExtraction, ExampleSource, SelectedExample


logger = logging.getLogger(__name__)


def text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class SourcePassage:
    text: str
    source: ExampleSource

    def reference(self, passage_id: str) -> dict:
        return {"passage_id": passage_id, "text": self.text,
                "start": self.source.start, "end": self.source.end}


def catalog_document(document: ParsedDocument, prefix: str) -> dict[str, SourcePassage]:
    digest = text_digest(document.text)
    return {f"{prefix}:0": SourcePassage(document.text, ExampleSource(
        name=document.name, document_id=document.id or f"local:{prefix}",
        document_sha256=digest, start=0, end=len(document.text),
        text_sha256=digest, verified=True,
    ))}


def _normalised_source(text: str) -> tuple[str, list[int], list[int]]:
    characters = []
    starts = []
    ends = []
    for offset, character in enumerate(text):
        if character.isspace():
            if characters and characters[-1] == " ":
                ends[-1] = offset + 1
                continue
            character = " "
        characters.append(character)
        starts.append(offset)
        ends.append(offset + 1)
    return "".join(characters), starts, ends


def materialize_examples(choices: list[SelectedExample], catalog: dict[str, SourcePassage]) -> list[Example]:
    examples = []
    for example_index, choice in enumerate(choices):
        try:
            passage = catalog[choice.passage_ids[0]]
        except KeyError as error:
            raise ValueError("The model selected an unknown source passage.") from error
        normalised = None
        starts = []
        ends = []
        extractions = []
        for extraction_index, extraction in enumerate(choice.extractions):
            if extraction.extraction_text not in passage.text:
                if normalised is None:
                    normalised, starts, ends = _normalised_source(passage.text)
                needle = re.sub(r"\s+", " ", extraction.extraction_text)
                start = normalised.find(needle)
                if needle.strip() and start >= 0 and normalised.find(needle, start + 1) == -1:
                    literal = passage.text[starts[start]:ends[start + len(needle) - 1]]
                    extraction = ExampleExtraction.model_validate({
                        **extraction.model_dump(), "extraction_text": literal,
                    })
                    logger.info("Restored source whitespace for example %s annotation %s",
                                example_index + 1, extraction_index + 1)
            extractions.append(extraction)
        examples.append(Example(text=passage.text, extractions=extractions, source=passage.source))
    return examples
