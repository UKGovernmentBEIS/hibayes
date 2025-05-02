import logging
from abc import ABC, abstractmethod
from typing import Any, Dict

from inspect_ai.log import EvalLog, EvalPlan, EvalSample


class MetadataExtractor(ABC):
    """Base class for all metadata extractors"""

    @abstractmethod
    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract metadata from a sample and eval log.

        Args:
            sample: The evaluation sample to extract data from
            eval_log: The complete evaluation log minus the samples

        Returns:
            Dictionary of extracted metadata
        """
        pass


class BaseMetadataExtractor(MetadataExtractor):
    """Default extractor providing basic metadata from all samples"""

    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        model_name = eval_log.eval.model
        model_name = model_name.split("/")[-1] if "/" in model_name else model_name

        return {
            "score": self._normalise_score(next(iter(sample.scores.values())).value),
            "target": str(sample.target),
            "model": model_name,
            "dataset": eval_log.eval.dataset.name,
            "task": str(sample.id),
            "epoch": sample.epoch,
            "num_messages": len(sample.messages),
        }

    def _normalise_score(self, score: Any) -> float:
        if score == "I":
            return 0.0
        if score == "C":
            return 1.0
        return float(score)


class CyberMetadataExtractor(MetadataExtractor):
    """Extracts cybersecurity domain expertise information"""

    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        metadata = sample.metadata.get("challenge_metadata", {})
        if not metadata:
            return {}

        attributes = {
            "max_messages": int(metadata.get("max_messages")),
            "category": str(metadata.get("category")),
            "split": str(metadata.get("split")),
            "full_name": str(metadata.get("full_name", sample.id)),
            "description": str(metadata.get("description")),
            "source": str(metadata.get("source")),
        }

        domains = {}
        capabilities = metadata.get("capabilities", [])
        if capabilities:
            domains = {
                domain["name"]: str(domain["level"])
                for domain in capabilities
                if "name" in domain and "level" in domain
            }

        return {k: v for k, v in {**attributes, **domains}.items() if v is not None}


class ToolsExtractor(MetadataExtractor):
    """Extracts information about tools used in evaluation"""

    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        tools = []
        for step in getattr(eval_log, "plan", EvalPlan()).steps:
            if isinstance(step.params, dict):
                tool_params = step.params.get("tools", [])
                if tool_params:
                    tools.extend(
                        [
                            str(tool.get("name"))
                            for tool in tool_params
                            if "name" in tool
                        ]
                    )
        return {"tools": tools or [None]}


class TokenExtractor(MetadataExtractor):
    """Extracts token usage information"""

    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        model_usage = getattr(sample, "model_usage", {}) or {}
        if not model_usage:
            return {}

        # Helper function to safely extract usage data
        def safe_sum(attr_name):
            return sum(
                getattr(usage, attr_name, 0) or 0 for usage in model_usage.values()
            )

        return {
            "total_tokens": safe_sum("total_tokens"),
            "input_tokens": safe_sum("input_tokens"),
            "output_tokens": safe_sum("output_tokens"),
            "cache_write_tokens": safe_sum("input_tokens_cache_write"),
            "cache_read_tokens": safe_sum("input_tokens_cache_read"),
        }


# class Extractor:
#     pass


# def extractor(cls):
#     """Decorator to register a metadata extractor class"""
#     if not issubclass(cls, MetadataExtractor):
#         raise TypeError("Extractor must be a subclass of MetadataExtractor")


# @extractor
# def custom_extractor() -> Extractor:
#     """Build custom extractor for specific use cases."""

#     def extract(
#             sample: EvalSample, eval_log: EvalLog
#             ) -> Dict[str, Any]:
#         # Custom extraction logic For example, extracting
#         # the number of messages in a sample
#         return {"num_messages": len(sample.messages)}

#     return extract
