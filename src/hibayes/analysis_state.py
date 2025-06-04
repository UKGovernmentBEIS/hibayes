from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from arviz import InferenceData

from .utils import init_logger

if TYPE_CHECKING:
    from .model.models import BaseModel, Coords, Dims, Features, ModelConfig

logger = init_logger()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _dump_json(obj: Dict[str, Any], fname: Path) -> None:
    if is_dataclass(obj):
        obj = asdict(obj)
    with fname.open("w", encoding="utf-8") as fp:
        json.dump(obj, fp, indent=2, default=str)


def _load_json(fname: Path) -> Dict[str, Any]:
    with fname.open("r", encoding="utf-8") as fp:
        return json.load(fp)


class ModelAnalysisState:
    """State of the model analysis. This class is used to store model builder and1 extracted features for the model,"""

    def __init__(
        self,
        model_name: str,  # name of the statistical model
        model_builder: "BaseModel",  # statistical model builder
        inference_data: Optional[
            InferenceData
        ] = None,  # inference data re the statistical model fit
        diagnostics: Optional[
            Dict[str, Any]
        ] = None,  # outcomes of the checkers e.g. rhat
        is_fitted: bool = False,
    ) -> None:
        self._model_name: str = model_name
        self._model_builder: "BaseModel" = model_builder
        self._inference_data: InferenceData = (
            inference_data
            if inference_data
            else InferenceData()  # now that checks can happenbefore fitting, smoother to merge inferencedata rather than override
        )
        self._diagnostics: Dict[str, Any] = diagnostics or {}
        self._is_fitted: bool = is_fitted

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @model_name.setter
    def model_name(self, model_name: str) -> None:
        """Set the model name."""
        self._model_name = model_name

    @property
    def model(self) -> Callable:
        """Get the callable model function."""
        return self._model_builder.build_model()

    @property
    def model_config(self) -> "ModelConfig":
        """Get the model configuration."""
        return self._model_builder.config

    @model_config.setter
    def model_config(self, model_config: "ModelConfig") -> None:
        """Set the model configuration."""
        self._model_builder.config = model_config

    @property
    def model_builder(self) -> "BaseModel":
        """Get the model builder."""
        return self._model_builder

    @property
    def inference_data(self) -> InferenceData | None:
        """Get the inference_data."""
        return self._inference_data

    @inference_data.setter
    def inference_data(self, inference_data: InferenceData) -> None:
        """Set the inference_data."""
        self._inference_data = inference_data

    @property
    def diagnostics(self) -> Dict[str, Any] | None:
        """Get the diagnostics."""
        return self._diagnostics

    @diagnostics.setter
    def diagnostics(self, diagnostics: Dict[str, Any]) -> None:
        """Set the diagnostics."""
        self._diagnostics = diagnostics

    def add_diagnostic(self, name: str, diagnostic: Any) -> None:
        """Add a diagnostic."""
        if self._diagnostics is None:
            self._diagnostics = {}
        self._diagnostics[name] = diagnostic

    def diagnostic(self, var: str) -> Any:
        """Get a specific result."""
        return self._diagnostics.get(var, None)

    @property
    def is_fitted(self) -> bool:
        """Get the fitted status."""
        return self._is_fitted

    @is_fitted.setter
    def is_fitted(self, is_fitted: bool) -> None:
        """Set the fitted status."""
        self._is_fitted = is_fitted

    @property
    def link_function(self) -> Optional[Callable]:
        """Get the link function."""
        return self.model_config.link_function

    def save(self, path: Path) -> None:
        """
        save the model state

        Folder layout:

            <path>/
              ├── metadata.json
              ├── model_config.json
              ├── model_builder.pkl
              ├── features.json
              ├── coords.json
              ├── diagnostics.json
              └── inference_data.nc notive netcdf - see arviz for more info
        """
        _ensure_dir(path)

        _dump_json(
            {
                "model_name": self.model_name,
                "is_fitted": self.is_fitted,
            },
            path / "metadata.json",
        )

        _dump_json(self.model_config, path / "model_config.json")

        with (path / "model_builder.pkl").open("wb") as fp:
            pickle.dump(self.model_builder, fp)

        if self.diagnostics:
            _dump_json(self.diagnostics, path / "diagnostics.json")
            # if any figures save them as pngs:
            for name, obj in self.diagnostics.items():
                if not os.path.exists(path / "diagnostic_plots"):
                    os.makedirs(path / "diagnostic_plots")
                if isinstance(obj, plt.Figure):
                    obj.savefig(
                        path / "diagnostic_plots" / f"{name}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close(obj)

        if self.inference_data is not None:
            target = path / "inference_data.nc"
            tmp = target.with_suffix(
                ".tmp.nc"
            )  # arviz lazily load the inf data so it remains open. This seems to be the best approach to saving the file
            self.inference_data.to_netcdf(tmp)
            tmp.replace(target)

    @classmethod
    def load(cls, path: Path) -> "ModelAnalysisState":
        """Recreate ModelAnalysisState from path."""
        if not path.exists():
            raise FileNotFoundError(path)

        meta = _load_json(path / "metadata.json")
        model_name: str = meta["model_name"]
        is_fitted: bool = meta.get("is_fitted", False)

        # Builder (contains config + build_model implementation)
        with (path / "model_builder.pkl").open("rb") as fp:
            model_builder: "BaseModel" = pickle.load(fp)

        diagnostics = None
        if (path / "diagnostics.json").exists():
            diagnostics = _load_json(path / "diagnostics.json")

        inference_data = None
        if (path / "inference_data.nc").exists():
            inference_data = InferenceData.from_netcdf(path / "inference_data.nc")

        return cls(
            model_name=model_name,
            model_builder=model_builder,
            inference_data=inference_data,
            diagnostics=diagnostics,
            is_fitted=is_fitted,
        )


class AnalysisState:
    """State of the analysis. This class is used to store, data, features, model,
    results and configuration of the analysis.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        processed_data: Optional[pd.DataFrame] = None,  # processed data
        features: Optional[
            "Features"
        ] = None,  # extracted features and to be shared with the models
        coords: Optional[
            "Coords"
        ] = None,  # map dimensinos to coordinates - used for nice plotting vars
        dims: Optional[
            "Dims"
        ] = None,  # variable names to coordinates - used for nice plotting vars
        models: List[ModelAnalysisState] = [],
        communicate: Dict[str, plt.Figure | pd.DataFrame] = {},
    ) -> None:
        self._data: pd.DataFrame = (
            data  # extracted data from inspect eval logs see hibayes.load for details
        )
        self._processed_data: pd.DataFrame | None = (
            processed_data  # processed data, e.g. grouping, filtering etc
        )
        self._features: "Features" = features if features is not None else {}
        self._coords: "Coords" | None = coords
        self._dims: "Dims" | None = dims
        self._models: List[ModelAnalysisState] = models
        self._communicate: Dict[
            str, plt.Figure | pd.DataFrame
        ] | None = communicate  # plots of findings

    @property
    def data(self) -> pd.DataFrame:
        """Get the data."""
        return self._data

    @data.setter
    def data(self, data: pd.DataFrame) -> None:
        """Set the data."""
        self._data = data

    @property
    def processed_data(self) -> pd.DataFrame | None:
        """Get the processed data."""
        return self._processed_data

    @processed_data.setter
    def processed_data(self, processed_data: pd.DataFrame) -> None:
        """Set the processed data."""
        self._processed_data = processed_data

    @property
    def features(self) -> Dict[str, Any]:
        """Get the features."""
        return self._features

    @property
    def prior_features(self) -> Dict[str, Optional[Any]]:
        """Get the prior features. Bascally all bar observables."""
        features: Dict[str, Optional[Any]] = {
            k: v for k, v in self._features.items() if "obs" not in k
        }
        features["obs"] = None  # add obs as None for numpyro

        return features

    @features.setter
    def features(self, features: "Features") -> None:
        """Set the features."""
        self._features = features

    def feature(self, feature_name: str) -> Any:
        """Get a specific feature."""
        return self._features.get(feature_name, None)

    @property
    def coords(self) -> "Coords" | None:
        """Get the coordinates."""
        return self._coords

    @coords.setter
    def coords(self, coords: "Coords") -> None:
        """Set the coordinates."""
        self._coords = coords

    def coord(self, coord_name: str) -> list | None:
        """Get a specific coordinate."""
        return self._coords[coord_name] if self._coords else None

    @property
    def dims(self) -> "Dims" | None:
        """Get the dimensions."""
        return self._dims

    @dims.setter
    def dims(self, dims: "Dims") -> None:
        """Set the dimensions."""
        self._dims = dims

    def dim(self, dim_name: str) -> list | None:
        """Get a specific dimension."""
        return self._dims[dim_name] if self._dims else None

    @property
    def communicate(self) -> Dict[str, plt.Figure | pd.DataFrame] | None:
        """Get the communicate."""
        return self._communicate

    def communicate_item(self, item_name: str) -> plt.Figure | pd.DataFrame:
        """Get a specific communicate item."""
        return self._communicate[item_name]

    def add_plot(self, plot: plt.Figure, plot_name: str) -> None:
        """Add a plot to the communicate."""
        if self._communicate is None:
            self._communicate = {}
        self._communicate[plot_name] = plot

    def add_table(self, table: pd.DataFrame, table_name: str) -> None:
        """Add a table to the communicate."""
        if self._communicate is None:
            self._communicate = {}
        self._communicate[table_name] = table

    @property
    def models(self) -> List[ModelAnalysisState]:
        """Get the models."""
        return self._models

    @models.setter
    def models(self, models: List[ModelAnalysisState]) -> None:
        """Set the models."""
        self._models = models

    def add_model(self, model: ModelAnalysisState) -> None:
        """Add a model to the analysis state."""
        self._models.append(model)

    def get_model(self, model_name: str) -> ModelAnalysisState:
        """Get a model by name."""
        for model in self.models:
            if model.model_name == model_name:
                return model
        raise ValueError(f"Model {model_name} not found in analysis state.")

    def get_best_model(
        self, with_respect_to: str = "elpd_waic", minimum: bool = False
    ) -> ModelAnalysisState:
        """
        Get the best model based on a diagnostic metric (lower is better).

        Args:
            with_respect_to (str): The diagnostic metric to use for comparison. This needs
            to be an attribute calculated by the checkers and added to diagnoistics.
            minimum (bool): If True, the model with the minimum value of the diagnostic is returned.

        """
        # Collect models that have the specified diagnostic
        candidates: List[Tuple[float, ModelAnalysisState]] = []
        for model in self.models:
            val = model.diagnostic(with_respect_to)
            if val is not None:
                candidates.append((val, model))

        if not candidates:
            raise ValueError(f"No models have diagnostic '{with_respect_to}'.")

        # Select the model with best metric
        best_value, best_model = (
            min(candidates, key=lambda x: x[0])
            if minimum
            else max(candidates, key=lambda x: x[0])
        )
        return best_model

    def save(self, path: Path) -> None:
        """
        save the analysis state

        Folder layout:

            <path>/
              ├── data.parquet
              ├── communicate/
              │     ├── <name>.png
              │     └── <name>.parquet
              └── models/
                    └── <model_name>/ then see ModelAnalysisState.save
        """
        _ensure_dir(path)

        self.data.to_parquet(
            path / "data.parquet",
            engine="pyarrow",  # auto might result in different engines in different setups)
            compression="snappy",
        )
        if self.features:
            with (path / "features.pkl").open("wb") as fp:
                pickle.dump(self.features, fp)

        if self.coords is not None:
            _dump_json(self.coords, path / "coords.json")

        if self.dims is not None:
            _dump_json(self.dims, path / "dims.json")
        if self._communicate:
            comm_path = path / "communicate"
            _ensure_dir(comm_path)

            for name, obj in self._communicate.items():
                if isinstance(obj, plt.Figure):
                    obj.savefig(comm_path / f"{name}.png", dpi=300, bbox_inches="tight")
                    plt.close(obj)  # free memory in long pipelines
                elif isinstance(obj, pd.DataFrame):
                    obj.to_csv(
                        comm_path / f"{name}.csv",
                    )
                else:
                    raise TypeError(
                        f"Unsupported communicate object type for '{name}': {type(obj)}"
                    )

        models_root = path / "models"
        _ensure_dir(models_root)

        for model_state in self._models:
            model_state.save(models_root / model_state.model_name)

    @classmethod
    def load(cls, path: Path) -> "AnalysisState":
        """load AnalysisState from path."""
        if not path.exists():
            raise FileNotFoundError(path)

        data = pd.read_parquet(path / "data.parquet")

        # Features
        if (path / "features.pkl").exists():
            with (path / "features.pkl").open("rb") as fp:
                features: "Features" = pickle.load(fp)

        if (path / "coords.json").exists():
            coords: "Coords" = _load_json(path / "coords.json")
        else:
            coords = None
        if (path / "dims.json").exists():
            dims: "Dims" = _load_json(path / "dims.json")
        else:
            dims = None

        # figures and tables!
        communicate: Dict[str, plt.Figure | pd.DataFrame] = {}
        comm_path = path / "communicate"
        if comm_path.exists():
            for p in comm_path.iterdir():
                stem, suffix = p.stem, p.suffix.lower()
                if suffix == ".png":
                    img = plt.imread(p)  # so that they can be matplotlib figures again.
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(img)
                    ax.axis("off")
                    communicate[stem] = fig
                elif suffix == ".csv":
                    communicate[stem] = pd.read_csv(p)
                else:
                    logger.warning(
                        f"Unsupported file type in communicate directory: {p}. Supported types are .png and .parquet."
                    )
                    continue

        models_root = path / "models"
        models: List[ModelAnalysisState] = []
        if models_root.exists():
            for model_dir in models_root.iterdir():
                if model_dir.is_dir():
                    models.append(ModelAnalysisState.load(model_dir))

        return cls(
            data=data,
            models=models,
            features=features,
            coords=coords,
            dims=dims,
            communicate=communicate,
        )
