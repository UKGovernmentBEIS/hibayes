from typing import List, Tuple

from arviz import InferenceData


def drop_not_present_vars(
    vars: List[str],
    inference_data: InferenceData,
) -> Tuple[List[str], List[str]]:
    """
    Drop variables that are not present in the inference data. Also return the
    list of variables which were dropped.
    """
    present_vars = [var for var in vars if var in inference_data.posterior.data_vars]
    dropped_vars = [var for var in vars if var not in present_vars]
    return present_vars, dropped_vars
