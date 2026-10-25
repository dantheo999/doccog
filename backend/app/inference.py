import json
import logging
import threading
from typing import Literal

from langextract.core import exceptions
from langextract.core.format_handler import FormatHandler
from langextract.providers.openai import OpenAILanguageModel
from langextract.resolver import Resolver, ResolverParsingError


logger = logging.getLogger(__name__)

OUTPUT_FORMAT = (
    'Return only a JSON object with an "extractions" array, using the A: format '
    'demonstrated below. Each annotation uses its class as a key with a literal '
    'source-text string value, and "<class>_attributes" for its attributes object. '
    'Do not use extraction_class/extraction_text/attributes record fields, nested '
    'entity objects, or include example text. Use {"extractions": []} if this target '
    'chunk contains no matches. This output format overrides conflicting formatting '
    'instructions, not the extraction goal.'
)


class InvalidExtractionOutput(exceptions.InferenceRuntimeError):
    def __init__(self, reason: Literal["invalid_json", "invalid_structure", "invalid_annotation", "truncated"]):
        self.reason = reason
        super().__init__(f"Invalid extraction output ({reason}).")


def output_failure_reason(error: ResolverParsingError) -> Literal["invalid_json", "invalid_structure", "invalid_annotation"]:
    if isinstance(error.__cause__, ValueError):
        return "invalid_annotation"
    if isinstance(getattr(error.__cause__, "__cause__", None), json.JSONDecodeError):
        return "invalid_json"
    return "invalid_structure"


class RetryingOpenAIModel(OpenAILanguageModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._usage_lock = threading.Lock()
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _response_to_scored_output(self, response):
        usage = getattr(response, "usage", None)
        with self._usage_lock:
            self.usage["calls"] += 1
            self.usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            self.usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        if response.choices and response.choices[0].finish_reason == "length":
            raise InvalidExtractionOutput("truncated")
        return OpenAILanguageModel._response_to_scored_output(response)

    def _process_single_prompt(self, prompt, config):
        resolver = Resolver(format_handler=FormatHandler(use_fences=False))
        for attempt in range(2):
            request_prompt = prompt if attempt == 0 else (
                "The previous response was invalid or incomplete. Re-extract the "
                "same target, returning ALL its matches in the required format.\n"
                f"{OUTPUT_FORMAT}\n\n{prompt}"
            )
            try:
                output = super()._process_single_prompt(request_prompt, config)
                resolver.resolve(output.output, suppress_parse_errors=False)
                return output
            except (ResolverParsingError, InvalidExtractionOutput) as error:
                if attempt == 1:
                    if isinstance(error, InvalidExtractionOutput):
                        raise
                    raise InvalidExtractionOutput(output_failure_reason(error)) from error
                reason = error.reason if isinstance(error, InvalidExtractionOutput) else output_failure_reason(error)
                logger.warning("Retrying one extraction chunk after invalid model output (%s)", reason)

    def close(self):
        self._client.close()
