import argparse
import pathlib

from ..analysis import AnalysisConfig, load_data
from ..ui import ModellingDisplay


def run_load(args, display=None):
    config = AnalysisConfig.from_yaml(args.config)
    if display is None:
        display = ModellingDisplay()
    analysis_state = load_data(config=config.data_loader, display=display)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    analysis_state.save(path=out)


def main():
    parser = argparse.ArgumentParser(description="Load data hibayes.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the configuration file (YAML format). See examples/*/*.yaml for examples",
    )
    parser.add_argument(
        "--out", required=True, help="Where to write the DVC tracking files"
    )

    parser.add_argument(
        "--no-tui",
        dest="use_tui",
        action="store_false",
        help="Use classic Rich display instead of interactive TUI",
    )
    parser.set_defaults(use_tui=True)

    args = parser.parse_args()
    if args.use_tui:
        from ..ui.textual.app import run_with_tui
        run_with_tui(run_load, args)
    else:
        run_load(args)


if __name__ == "__main__":
    main()
