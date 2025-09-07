from typing import Any, Dict

from inspect_ai.log import EvalLog, EvalSample

from hibayes.load import Extractor, extractor

DOMAINS = {
    "inspect_evals/mbpp": "coding",
    "DS-1000": "coding",
    "inspect_evals/boolq": "reasoning",
    "inspect_evals/race_h": "reasoning",
}

SUB_DOMAINS = {
    "inspect_evals/mbpp": "easy",
    "DS-1000": "hard",
    "inspect_evals/boolq": "easy",
    "inspect_evals/race_h": "hard",
}


@extractor
def domains_extractor(
    default_domain: str = "other",
    default_sub_domain: str = "other",
) -> Extractor:
    """
    Extract domain categorisation from evaluation logs.

    Args:
        default_domain: Default value if domain is not found.
        default_sub_domain: Default value if sub-domain is not found.

    Returns:
        An Extractor function that categorises tasks by domain.
    """

    def extract(sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        """Extract domain information from the evaluation log."""
        task_name = eval_log.eval.task if hasattr(eval_log.eval, "task") else ""

        return {
            "dataset": task_name,
            "domain": DOMAINS.get(task_name, default_domain),
            "sub_domain": SUB_DOMAINS.get(task_name, default_sub_domain),
        }

    return extract
