from typing import Any, Dict

from inspect_ai.log import EvalLog, EvalSample

from hibayes.load import MetadataExtractor

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


class Domains(MetadataExtractor):
    def extract(self, sample: EvalSample, eval_log: EvalLog) -> Dict[str, Any]:
        return {
            "dataset": eval_log.eval.task,
            "domain": DOMAINS.get(eval_log.eval.task, "other"),
            "sub_domain": SUB_DOMAINS.get(eval_log.eval.task, "other"),
            # "sub_domain": eval_log.eval.task,
        }
