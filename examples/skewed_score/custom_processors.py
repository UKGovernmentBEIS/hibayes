from typing import Dict

from hibayes.analysis import AnalysisState
from hibayes.process import DataProcessor, process
from hibayes.ui import ModellingDisplay


@process
def add_categorical_column(
    new_column: str,
    source_column: str,
    mapping_rules: Dict[str, str],
    default_value: str = "unknown",
    case_sensitive: bool = False,
) -> DataProcessor:
    """
    Add a new categorical column to the processed data based on mapping rules (does the value contain the pattern).
    """

    def processor(
        state: AnalysisState, display: ModellingDisplay | None = None
    ) -> AnalysisState:
        def categorize(value):
            str_value = str(value)
            if not case_sensitive:
                str_value = str_value.lower()

            for pattern, category in mapping_rules.items():
                search_pattern = pattern if case_sensitive else pattern.lower()
                if search_pattern in str_value:
                    return category
            return default_value

        state.processed_data[new_column] = state.processed_data[source_column].apply(
            categorize
        )
        return state

    return processor
