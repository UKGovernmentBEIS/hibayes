from __future__ import annotations

from typing import Any, Dict, List

from inspect_ai.log import EvalLog, EvalPlan, EvalSample

from ._extractor import Extractor, extractor


@extractor
def base_extractor() -> Extractor:
    """
    Extract basic metadata from evaluation samples.

    Returns:
        An Extractor function that extracts basic metadata.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract basic metadata from a sample."""
        model_name_raw = eval_log.eval.model
        model_name = (
            model_name_raw.split("/")[-1] if "/" in model_name_raw else model_name_raw
        )

        # Normalize score
        score_value = next(iter(sample.scores.values())).value
        if score_value == "I":
            score = 0.0
        elif score_value == "C":
            score = 1.0
        else:
            score = float(score_value)

        # Record the log file path this sample came from
        log_path = eval_log.location

        return {
            "score": score,
            "target": str(sample.target),
            "model": model_name,
            "model_raw": model_name_raw,
            "dataset": eval_log.eval.dataset.name,
            "task": str(sample.id),
            "epoch": sample.epoch,
            "num_messages": len(sample.messages),
            "log_path": log_path,
        }

    return extract


@extractor
def cyber_extractor() -> Extractor:
    """
    Extract cybersecurity domain expertise information.

    Returns:
        An Extractor function that extracts cyber metadata.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract cybersecurity metadata from a sample."""
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

    return extract


@extractor
def tools_extractor() -> Extractor:
    """
    Extract information about tools used in evaluation.

    Returns:
        An Extractor function that extracts tool usage information.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract tool usage information from a sample."""
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

    return extract


@extractor
def token_extractor() -> Extractor:
    """
    Extract token usage information.

    Returns:
        An Extractor function that extracts token usage data.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract token usage information from a sample."""
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

    return extract


@extractor
def message_count_extractor(
    include_system: bool = True,
    include_user: bool = True,
    include_assistant: bool = True,
) -> Extractor:
    """
    Extract message count information by role.

    Args:
        include_system: Whether to count system messages.
        include_user: Whether to count user messages.
        include_assistant: Whether to count assistant messages.

    Returns:
        An Extractor function that counts messages by role.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract message counts by role."""
        result = {}

        if include_system:
            result["system_messages"] = sum(
                1 for msg in sample.messages if msg.role == "system"
            )

        if include_user:
            result["user_messages"] = sum(
                1 for msg in sample.messages if msg.role == "user"
            )

        if include_assistant:
            result["assistant_messages"] = sum(
                1 for msg in sample.messages if msg.role == "assistant"
            )

        result["total_messages"] = len(sample.messages)

        return result

    return extract


@extractor
def metadata_field_extractor(
    field_path: str,
    default_value: Any = None,
    rename_to: str | None = None,
) -> Extractor:
    """
    Extract a specific field from sample metadata.

    Args:
        field_path: Dot-separated path to the metadata field (e.g., "challenge.difficulty").
        default_value: Default value if field is not found.
        rename_to: Optional name to use for the extracted field in the result.

    Returns:
        An Extractor function that extracts a specific metadata field.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract a specific metadata field."""
        # Navigate through nested metadata using the field path
        value = sample.metadata
        for field in field_path.split("."):
            if isinstance(value, dict):
                value = value.get(field)
                if value is None:
                    value = default_value
                    break
            else:
                value = default_value
                break

        # Use the field path as key unless rename_to is specified
        key = rename_to or field_path.replace(".", "_")
        return {key: value}

    return extract


@extractor
def score_normalizer(
    score_field: str = "score",
    normalize_to_01: bool = True,
    custom_mapping: Dict[Any, float] | None = None,
) -> Extractor:
    """
    Extract and normalize scores with custom mappings.

    Args:
        score_field: Name of the score field to extract.
        normalize_to_01: Whether to normalize scores to [0, 1] range.
        custom_mapping: Optional custom mapping of score values to floats.

    Returns:
        An Extractor function that normalizes scores.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract and normalize score."""
        score_value = next(iter(sample.scores.values())).value

        # Apply custom mapping if provided
        if custom_mapping and score_value in custom_mapping:
            normalized_score = custom_mapping[score_value]
        # Default mappings for common cases
        elif score_value == "I" or score_value == "INCORRECT":
            normalized_score = 0.0
        elif score_value == "C" or score_value == "CORRECT":
            normalized_score = 1.0
        else:
            try:
                normalized_score = float(score_value)
            except (ValueError, TypeError):
                normalized_score = 0.0

        # Normalize to [0, 1] if requested
        if normalize_to_01 and normalized_score > 1.0:
            # Assume percentage score
            normalized_score = normalized_score / 100.0

        return {score_field: normalized_score}

    return extract
