from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict
from unittest.mock import ANY, MagicMock, patch

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from hibayes.analysis_state import ModelAnalysisState
from hibayes.model import (
    Model,
    ModelConfig,
    ModelsToRunConfig,
    fit,
    model,
    two_level_group_binomial,
)
from hibayes.model.model_config import INIT_FUNCTION_MAP, LINK_FUNCTION_MAP, FitConfig
from hibayes.model.models import check_features
from hibayes.model.utils import (
    cloglog_to_prob,
    link_to_key,
    logit_to_prob,
    probit_to_prob,
)
from hibayes.process import Features
from hibayes.registry import RegistryInfo, registry_get, registry_info


class TestModelProtocol:
    """Test the Model protocol and decorator."""

    def test_model_decorator_registration(self):
        """Test that @model decorator registers models correctly."""

        @model
        def test_model_builder() -> Model:
            def model_impl(features: Features) -> None:
                pass

            return model_impl

        # Should be registered and callable
        model_instance = test_model_builder()
        assert callable(model_instance)

        # Should have registry info
        from hibayes.registry import registry_info

        info = registry_info(model_instance)
        assert info.type == "model"
        assert info.name == "test_model_builder"

    def test_model_decorator_enforces_obs_interface(self):
        """Test that model decorator enforces 'obs' in features."""

        @model
        def test_model_builder() -> Model:
            def model_impl(features: Features) -> None:
                pass

            return model_impl

        model_instance = test_model_builder()

        # Should raise error when 'obs' not in features
        with pytest.raises(ValueError, match=r".*Model must have 'obs' in features.*"):
            model_instance({"other_feature": jnp.array([1, 2, 3])})

    def test_model_decorator_passes_args_correctly(self):
        """Test that decorator correctly passes args to model builder."""
        build_calls = []

        @model
        def test_parametric_model(param1: str = "default", param2: int = 42) -> Model:
            build_calls.append({"param1": param1, "param2": param2})

            def model_impl(features: Features) -> None:
                pass

            return model_impl

        # Create model with custom parameters
        model_instance = test_parametric_model(param1="custom", param2=100)

        # Should have called builder with correct args
        assert len(build_calls) == 1
        assert build_calls[0]["param1"] == "custom"
        assert build_calls[0]["param2"] == 100

        # Model should work with valid features
        features = {"obs": jnp.array([1, 2, 3])}
        result = model_instance(features)
        assert result is None  # Our test model returns None

    def test_model_decorator_preserves_function_metadata(self):
        """Test that decorator preserves original function metadata."""

        @model
        def well_documented_model(important_param: float = 1.0) -> Model:
            """This is a well documented model.

            Args:
                important_param: A very important parameter
            """

            def model_impl(features: Features) -> None:
                pass

            return model_impl

        # Should preserve docstring and name
        assert "well documented model" in well_documented_model.__doc__
        assert well_documented_model.__name__ == "well_documented_model"


class TestModelBuilders:
    """Test built-in model builders."""

    def test_two_level_group_binomial_registration(self):
        """Test that two_level_group_binomial is properly registered."""
        # Should be able to create instance
        model_instance = two_level_group_binomial()
        assert callable(model_instance)

        # Should be registered in registry
        info = RegistryInfo(type="model", name="two_level_group_binomial")
        registry_model = registry_get(info)
        assert registry_model is two_level_group_binomial

    def test_two_level_group_binomial_requires_features(self):
        """Test that two_level_group_binomial validates required features."""
        model_instance = two_level_group_binomial()

        # Missing required features should raise error
        incomplete_features = {"obs": jnp.array([1, 2, 3])}

        with pytest.raises(ValueError, match=r".*Missing required features.*"):
            model_instance(incomplete_features)

    def test_two_level_group_binomial_with_valid_features(self):
        """Test two_level_group_binomial with valid features."""
        model_instance = two_level_group_binomial()

        # Create valid features
        features = {
            "obs": jnp.array([5, 3, 7, 2]),
            "num_group": 2,
            "group_index": jnp.array([0, 0, 1, 1]),
            "n_total": jnp.array([10, 8, 12, 6]),
        }

        # Should not raise error with valid features
        # Note: We can't easily test the actual numpyro sampling without mocking
        # but we can test that it doesn't immediately fail
        try:
            with patch("numpyro.sample"), patch("numpyro.deterministic"):
                result = model_instance(features)
                assert result is None  # numpyro models return None
        except Exception as e:
            # If it fails, it should be a numpyro-related issue, not our validation
            assert "Missing required features" not in str(e)

    def test_check_features_utility(self):
        """Test the check_features utility function."""
        features = {"a": 1, "b": 2, "c": 3}

        # Should pass with all required features present
        check_features(features, ["a", "b"])

        # Should fail with missing features
        with pytest.raises(ValueError, match=r".*Missing required features: d, e.*"):
            check_features(features, ["a", "d", "e"])

        # Error message should include current features
        with pytest.raises(
            ValueError, match=r".*Current features: \['a', 'b', 'c'\].*"
        ):
            check_features(features, ["missing"])


class TestCustomModelBuilders:
    """Test custom model builder functionality."""

    def test_custom_model_with_parameters(self):
        """Test creating custom models with parameters."""

        @model
        def custom_linear_model(slope: float = 1.0, intercept: float = 0.0) -> Model:
            def model_impl(features: Features) -> None:
                # Mock implementation that uses the parameters
                assert slope == 2.5
                assert intercept == 1.0

            return model_impl

        # Create model with custom parameters
        model_instance = custom_linear_model(slope=2.5, intercept=1.0)

        # Test it works
        features = {"obs": jnp.array([1, 2, 3])}
        model_instance(features)  # Should not raise

    def test_custom_model_with_complex_features(self):
        """Test custom model that processes complex features."""

        @model
        def custom_hierarchical_model(
            groups: int = 2, prior_scale: float = 1.0
        ) -> Model:
            def model_impl(features: Features) -> None:
                check_features(features, ["obs", "group_idx", "num_groups"])
                assert features["num_groups"] == groups
                # Could add numpyro sampling here

            return model_impl

        model_instance = custom_hierarchical_model(groups=3, prior_scale=0.5)

        features = {
            "obs": jnp.array([1, 0, 1, 1, 0]),
            "group_idx": jnp.array([0, 0, 1, 2, 2]),
            "num_groups": 3,
        }

        model_instance(features)  # Should not raise

    def test_custom_model_error_handling(self):
        """Test error handling in custom models."""

        @model
        def failing_model() -> Model:
            def model_impl(features: Features) -> None:
                raise RuntimeError("Custom model error")

            return model_impl

        model_instance = failing_model()
        features = {"obs": jnp.array([1, 2, 3])}

        with pytest.raises(RuntimeError, match="Custom model error"):
            model_instance(features)


class TestModelConfig:
    """Test ModelConfig functionality."""

    def test_default_model_config(self):
        """Test default ModelConfig values."""
        config = ModelConfig()

        assert isinstance(config.fit, FitConfig)
        assert config.main_effect_params is None
        assert config.tag is None
        assert config.link_function == logit_to_prob
        assert config.extra_kwargs == {}

    def test_model_config_custom_values(self):
        """Test ModelConfig with custom values."""
        fit_config = FitConfig(samples=1000, chains=2)

        config = ModelConfig(
            fit=fit_config,
            main_effect_params=["param1", "param2"],
            tag="v1.0",
            link_function=probit_to_prob,
            extra_kwargs={"custom_param": "value"},
        )

        assert config.fit.samples == 1000
        assert config.fit.chains == 2
        assert config.main_effect_params == ["param1", "param2"]
        assert config.tag == "v1.0"
        assert config.link_function == probit_to_prob
        assert config.extra_kwargs == {"custom_param": "value"}

    def test_model_config_from_dict_default(self):
        """Test ModelConfig.from_dict with empty dict."""
        config = ModelConfig.from_dict({})

        assert isinstance(config.fit, FitConfig)
        assert config.link_function == logit_to_prob

    def test_model_config_from_dict_with_values(self):
        """Test ModelConfig.from_dict with custom values."""
        config_dict = {
            "fit": {"samples": 2000, "chains": 3},
            "main_effect_params": ["effect1", "effect2"],
            "tag": "experiment_v2",
            "link_function": "probit",
            "custom_param": "custom_value",
        }

        config = ModelConfig.from_dict(config_dict)

        assert config.fit.samples == 2000
        assert config.fit.chains == 3
        assert config.main_effect_params == ["effect1", "effect2"]
        assert config.tag == "experiment_v2"
        assert config.link_function == probit_to_prob
        assert config.extra_kwargs == {"custom_param": "custom_value"}

    def test_model_config_link_function_string(self):
        """Test link function specification by string."""
        for link_name, link_func in LINK_FUNCTION_MAP.items():
            config = ModelConfig.from_dict({"link_function": link_name})
            assert config.link_function == link_func

    def test_model_config_invalid_link_function(self):
        """Test error for invalid link function."""
        with pytest.raises(
            ValueError, match=r".*Link function invalid_link not recognised.*"
        ):
            ModelConfig.from_dict({"link_function": "invalid_link"})

    def test_model_config_link_function_callable(self):
        """Test link function specification by callable."""
        config = ModelConfig.from_dict({"link_function": probit_to_prob})
        assert config.link_function == probit_to_prob

    def test_model_config_invalid_callable_link_function(self):
        """Test error for invalid callable link function."""

        def custom_link(x):
            return x

        with pytest.raises(
            ValueError, match=r".*Link function must be one of the predefined.*"
        ):
            ModelConfig.from_dict({"link_function": custom_link})

    def test_model_config_invalid_link_function_type(self):
        """Test error for invalid link function type."""
        with pytest.raises(
            ValueError, match=r".*Link function must be a string or a callable.*"
        ):
            ModelConfig.from_dict({"link_function": 123})

    def test_model_config_get_plot_params(self):
        """Test get_plot_params method."""
        config1 = ModelConfig(main_effect_params=["param1", "param2"])
        assert config1.get_plot_params() == ["param1", "param2"]

        config2 = ModelConfig()
        assert config2.get_plot_params() is None

    def test_model_config_from_yaml(self, tmp_path: Path):
        """Test ModelConfig.from_yaml."""
        yaml_content = {
            "fit": {"samples": 3000, "chains": 4},
            "main_effect_params": ["alpha", "beta"],
            "tag": "yaml_test",
            "link_function": "cloglog",
            "extra_param": "extra_value",
        }

        yaml_file = tmp_path / "model_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ModelConfig.from_yaml(str(yaml_file))

        assert config.fit.samples == 3000
        assert config.fit.chains == 4
        assert config.main_effect_params == ["alpha", "beta"]
        assert config.tag == "yaml_test"
        assert config.link_function == cloglog_to_prob
        assert config.extra_kwargs == {"extra_param": "extra_value"}

    def test_model_config_save(self, tmp_path: Path):
        """Test ModelConfig.save method."""
        config = ModelConfig(
            fit=FitConfig(samples=1500),
            main_effect_params=["param1"],
            tag="save_test",
            link_function=probit_to_prob,
            extra_kwargs={"saved_param": "saved_value"},
        )

        save_path = tmp_path / "saved_config.json"

        print(config)
        config.save(save_path)

        assert save_path.exists()

        # Verify saved content
        with open(save_path, "r") as f:
            loaded_config_data = json.load(f)

        config = ModelConfig.from_dict(loaded_config_data)
        print(config)

        assert config.fit.samples == 1500
        assert config.main_effect_params == ["param1"]
        assert config.tag == "save_test"
        assert config.link_function == probit_to_prob
        assert config.extra_kwargs["saved_param"] == "saved_value"


class TestFitConfig:
    """Test FitConfig functionality."""

    def test_default_fit_config(self):
        """Test default FitConfig values."""
        config = FitConfig()

        assert config.method == "NUTS"
        assert config.samples == 2000
        assert config.warmup == 1000
        assert config.chains == 4
        assert config.seed == 0
        assert config.progress_bar is True
        assert config.target_accept == 0.95
        assert config.max_tree_depth == 10
        assert config.init_strategy == "median"

    def test_fit_config_custom_values(self):
        """Test FitConfig with custom values."""
        config = FitConfig(
            method="HMC",
            samples=2000,
            warmup=1000,
            chains=2,
            seed=42,
            progress_bar=False,
            target_accept=0.9,
            max_tree_depth=15,
        )

        assert config.method == "HMC"
        assert config.samples == 2000
        assert config.warmup == 1000
        assert config.chains == 2
        assert config.seed == 42
        assert config.progress_bar is False
        assert config.target_accept == 0.9
        assert config.max_tree_depth == 15

    def test_fit_config_merged(self):
        """Test FitConfig.merged method."""
        original = FitConfig(samples=1000, chains=2)
        updated = original.merged(samples=2000, seed=42)

        # Original should be unchanged
        assert original.samples == 1000
        assert original.chains == 2
        assert original.seed == 0

        # Updated should have new values
        assert updated.samples == 2000
        assert updated.chains == 2  # Unchanged
        assert updated.seed == 42

    @pytest.mark.parametrize("init_strategy", ["median", "mean", "uniform"])
    def test_fit_config_init_strategy_valid_values(self, init_strategy):
        """Test FitConfig with all valid init_strategy values."""
        config = FitConfig(init_strategy=init_strategy)
        assert config.init_strategy == init_strategy
        # Verify the strategy maps to a valid function
        assert init_strategy in INIT_FUNCTION_MAP

    def test_fit_config_init_strategy_from_dict(self):
        """Test init_strategy is properly parsed from ModelConfig.from_dict."""
        config = ModelConfig.from_dict({"fit": {"init_strategy": "uniform"}})
        assert config.fit.init_strategy == "uniform"

    def test_fit_config_init_strategy_invalid_raises_error(self):
        """Test that invalid init_strategy raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ModelConfig.from_dict({"fit": {"init_strategy": "invalid_strategy"}})

        error_msg = str(exc_info.value)
        assert "invalid_strategy" in error_msg
        assert "median" in error_msg
        assert "mean" in error_msg
        assert "uniform" in error_msg

    def test_fit_config_init_strategy_merged(self):
        """Test FitConfig.merged preserves init_strategy."""
        original = FitConfig(init_strategy="uniform")
        updated = original.merged(samples=500)
        assert updated.init_strategy == "uniform"

        # And can update init_strategy
        updated2 = original.merged(init_strategy="mean")
        assert updated2.init_strategy == "mean"


class TestModelsToRunConfig:
    """Test ModelsToRunConfig functionality."""

    def test_default_models_to_run_config(self):
        """Test default ModelsToRunConfig values."""
        config = ModelsToRunConfig()

        # Should use default models
        assert len(config.enabled_models) == 1
        model_instance, model_config = config.enabled_models[0]
        assert callable(model_instance)
        assert isinstance(model_config, ModelConfig)

        model_info = registry_info(model_instance)

        assert model_info.name == "two_level_group_binomial"

    @patch("hibayes.model.model_config.registry_get")
    def test_models_to_run_config_custom_models(self, mock_registry_get):
        """Test ModelsToRunConfig with custom models."""
        # Mock registry to return a test model
        mock_model_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_builder.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_builder

        config = ModelsToRunConfig(
            enabled_models=[(mock_model_instance, ModelConfig(tag="custom"))]
        )

        assert len(config.enabled_models) == 1
        model_instance, model_config = config.enabled_models[0]
        assert model_instance == mock_model_instance
        assert model_config.tag == "custom"

    @patch("hibayes.model.model_config.registry_get")
    def test_models_to_run_config_from_dict_list_format(self, mock_registry_get):
        """Test ModelsToRunConfig.from_dict with list format."""
        # Mock registry
        mock_model_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_builder.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_builder

        config_dict = {
            "models": [
                "TestModel1",
                {
                    "name": "TestModel2",
                    "config": {
                        "fit": {"samples": 1000},
                        "tag": "test_v1",
                    },
                },
            ]
        }

        config = ModelsToRunConfig.from_dict(config_dict)

        assert len(config.enabled_models) == 2

        # Check calls to registry
        assert mock_registry_get.call_count == 2
        assert mock_model_builder.call_count == 2

    @patch("hibayes.model.model_config.registry_get")
    def test_models_to_run_config_from_dict_dict_format(self, mock_registry_get):
        """Test ModelsToRunConfig.from_dict with dict format."""
        mock_model_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_builder.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_builder

        config_dict = {
            "models": {
                "test_model_1": {
                    "fit": {"samples": 2000},
                    "custom_param": "custom_value",
                },
                "test_model_2": {
                    "tag": "v2.0",
                },
            }
        }

        config = ModelsToRunConfig.from_dict(config_dict)

        assert len(config.enabled_models) == 2
        assert mock_registry_get.call_count == 2

    @patch("hibayes.model.model_config.registry_get")
    @patch("hibayes.model.model_config._import_path")
    def test_models_to_run_config_with_custom_path(
        self, mock_import_path, mock_registry_get
    ):
        """Test ModelsToRunConfig with custom model paths."""
        mock_model_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_builder.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_builder

        config_dict = {"path": "/custom/models.py", "models": ["custom_model"]}

        config = ModelsToRunConfig.from_dict(config_dict)

        mock_import_path.assert_called_once_with("/custom/models.py")
        assert len(config.enabled_models) == 1

    @patch("hibayes.model.model_config.registry_get")
    @patch("hibayes.model.model_config._import_path")
    def test_models_to_run_config_with_multiple_custom_paths(
        self, mock_import_path, mock_registry_get
    ):
        """Test ModelsToRunConfig with multiple custom paths."""
        mock_model_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_builder.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_builder

        config_dict = {
            "path": ["/custom/models1.py", "/custom/models2.py"],
            "models": ["custom_model_1"],
        }

        _ = ModelsToRunConfig.from_dict(config_dict)

        assert mock_import_path.call_count == 2
        mock_import_path.assert_any_call("/custom/models1.py")
        mock_import_path.assert_any_call("/custom/models2.py")

    @patch("hibayes.model.model_config.registry_get")
    def test_models_to_run_config_with_extra_kwargs(self, mock_registry_get):
        """Test that extra_kwargs are passed to model builders."""
        mock_model_builder = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_builder.return_value = mock_model_instance
        mock_registry_get.return_value = mock_model_builder

        config_dict = {
            "models": {
                "test_model": {
                    "fit": {"samples": 1000},
                    "custom_param1": "value1",
                    "custom_param2": 42,
                }
            }
        }

        _ = ModelsToRunConfig.from_dict(config_dict)

        # Check that model builder was called with extra kwargs
        mock_model_builder.assert_called_once_with(
            custom_param1="value1", custom_param2=42
        )

    def test_models_to_run_config_from_yaml(self, tmp_path: Path):
        """Test ModelsToRunConfig.from_yaml."""
        yaml_content = {
            "models": [
                "test_model_1",
                {"name": "test_model_2", "config": {"tag": "yaml_test"}},
            ]
        }

        yaml_file = tmp_path / "models_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        with patch("hibayes.model.model_config.registry_get") as mock_registry_get:
            mock_model_builder = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_builder.return_value = mock_model_instance
            mock_registry_get.return_value = mock_model_builder

            config = ModelsToRunConfig.from_yaml(str(yaml_file))

            assert len(config.enabled_models) == 2

    def test_models_to_run_config_from_none(self):
        """Test ModelsToRunConfig.from_dict with None input."""
        config = ModelsToRunConfig.from_dict(None)

        # Should use default models
        assert len(config.enabled_models) == 1


class TestFitFunction:
    """Test the fit function."""

    @patch("hibayes.model.fit.MCMC")
    @patch("hibayes.model.fit.NUTS")
    @patch("hibayes.model.fit.jax.random.PRNGKey")
    @patch("hibayes.model.fit.az.from_numpyro")
    def test_fit_nuts_default(
        self, mock_from_numpyro, mock_prng_key, mock_nuts, mock_mcmc
    ):
        """Test fit function with NUTS (default)."""
        # Setup mocks
        mock_kernel = MagicMock()
        mock_nuts.return_value = mock_kernel
        mock_mcmc_instance = MagicMock()
        mock_mcmc.return_value = mock_mcmc_instance
        mock_inference_data = MagicMock()
        mock_from_numpyro.return_value = mock_inference_data

        # Create test model analysis state
        model_state = self._create_test_model_analysis_state()

        # Run fit
        fit(model_state)

        # Verify NUTS kernel was created with correct parameters
        mock_nuts.assert_called_once_with(
            model_state.model,
            target_accept_prob=0.95,
            max_tree_depth=10,
            init_strategy=ANY,  # init_strategy is a callable
        )

        # Verify MCMC was configured correctly
        mock_mcmc.assert_called_once_with(
            mock_kernel,
            num_samples=2000,
            num_warmup=1000,
            num_chains=4,
            progress_bar=True,
            chain_method="parallel",
        )

        # Verify MCMC was run
        mock_mcmc_instance.run.assert_called_once()
        call_args = mock_mcmc_instance.run.call_args
        assert call_args[0][1] == model_state.features  # features passed
        assert call_args[1]["extra_fields"] == ("potential_energy", "energy")

        # Verify inference data was created and assigned
        assert model_state.inference_data == mock_inference_data
        assert model_state.is_fitted is True

    @patch("hibayes.model.fit.MCMC")
    @patch("hibayes.model.fit.HMC")
    @patch("hibayes.model.fit.jax.random.PRNGKey")
    @patch("hibayes.model.fit.az.from_numpyro")
    def test_fit_hmc(self, mock_from_numpyro, mock_prng_key, mock_hmc, mock_mcmc):
        """Test fit function with HMC."""
        # Setup mocks
        mock_kernel = MagicMock()
        mock_hmc.return_value = mock_kernel
        mock_mcmc_instance = MagicMock()
        mock_mcmc.return_value = mock_mcmc_instance
        mock_inference_data = MagicMock()
        mock_from_numpyro.return_value = mock_inference_data

        # Create test model analysis state with HMC config
        model_state = self._create_test_model_analysis_state(method="HMC")

        # Run fit
        fit(model_state)

        # Verify HMC kernel was created
        mock_hmc.assert_called_once_with(model_state.model)

        # Verify the rest of the process
        assert model_state.is_fitted is True

    def test_fit_invalid_method(self):
        """Test fit function with invalid method."""
        model_state = self._create_test_model_analysis_state(method="INVALID")

        with pytest.raises(
            ValueError, match=r".*Unsupported inference method: INVALID.*"
        ):
            fit(model_state)

    def _create_test_model_analysis_state(self, method="NUTS"):
        """Helper to create test ModelAnalysisState."""
        # Create a mock model
        mock_model = MagicMock()

        # Create features
        features = {
            "obs": jnp.array([1, 0, 1, 0]),
            "num_group": 2,
            "group_index": jnp.array([0, 0, 1, 1]),
            "n_total": jnp.array([5, 5, 5, 5]),
        }

        # Create model config
        fit_config = FitConfig(method=method)
        model_config = ModelConfig(fit=fit_config)

        # Create model analysis state
        model_state = ModelAnalysisState(
            model=mock_model,
            model_config=model_config,
            features=features,
            coords={},
            dims={},
        )

        return model_state


class TestModelUtils:
    """Test model utility functions."""

    def test_logit_to_prob(self):
        """Test logit_to_prob function."""
        # Test known values
        assert logit_to_prob(0) == pytest.approx(0.5)
        assert logit_to_prob(-np.inf) == pytest.approx(0.0)
        assert logit_to_prob(np.inf) == pytest.approx(1.0)

        # Test array input
        x = np.array([-2, -1, 0, 1, 2])
        result = logit_to_prob(x)
        expected = 1 / (1 + np.exp(-x))
        np.testing.assert_array_almost_equal(result, expected)

    def test_probit_to_prob(self):
        """Test probit_to_prob function."""
        # Test known values
        assert probit_to_prob(0) == pytest.approx(0.5)
        assert probit_to_prob(-np.inf) == pytest.approx(0.0)
        assert probit_to_prob(np.inf) == pytest.approx(1.0)

        # Test array input
        x = np.array([-2, -1, 0, 1, 2])
        result = probit_to_prob(x)
        from scipy.stats import norm

        expected = norm.cdf(x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_cloglog_to_prob(self):
        """Test cloglog_to_prob function."""
        # Test known values
        x = np.array([-2, -1, 0, 1, 2])
        result = cloglog_to_prob(x)
        expected = 1.0 - np.exp(-np.exp(x))
        np.testing.assert_array_almost_equal(result, expected)

    def test_link_to_key(self):
        """Test link_to_key function."""
        # Test with function objects
        assert link_to_key(logit_to_prob, LINK_FUNCTION_MAP) == "logit"
        assert link_to_key(probit_to_prob, LINK_FUNCTION_MAP) == "probit"

        # Test with string input
        assert link_to_key("logit", LINK_FUNCTION_MAP) == "logit"

        # Test with unknown function
        def unknown_func(x):
            return x

        with pytest.raises(ValueError, match=r".*Unknown link function.*"):
            link_to_key(unknown_func, LINK_FUNCTION_MAP)


class TestModelIntegration:
    """Integration tests for model functionality."""

    def test_full_model_pipeline(self):
        """Test complete model pipeline from registration to fitting."""

        @model
        def test_integration_model(scale: float = 1.0) -> Model:
            def model_impl(features: Features) -> None:
                check_features(features, ["obs"])
                # Simple model implementation
                import numpyro
                import numpyro.distributions as dist

                p = numpyro.sample("p", dist.Beta(1, 1))
                numpyro.sample(
                    "obs", dist.Binomial(total_count=1, probs=p), obs=features["obs"]
                )

            return model_impl

        # Create model instance
        model_instance = test_integration_model(scale=2.0)

        # Test model with features
        features = {"obs": jnp.array([1, 0, 1, 1, 0])}

        # This should not raise an error (actual execution would need numpyro context)
        with patch("numpyro.sample"), patch("numpyro.distributions"):
            result = model_instance(features)
            assert result is None

    @patch("hibayes.model.model_config.registry_get")
    def test_models_to_run_config_integration(self, mock_registry_get):
        """Test integration of ModelsToRunConfig with actual model builders."""

        # Create a test model builder
        @model
        def integration_test_model(param1: str = "default") -> Model:
            def model_impl(features: Features) -> None:
                pass

            return model_impl

        mock_registry_get.return_value = integration_test_model

        # Create config that uses the model with parameters
        config_dict = {
            "models": {
                "integration_test_model": {
                    "fit": {"samples": 1000},
                    "param1": "custom_value",
                }
            }
        }

        config = ModelsToRunConfig.from_dict(config_dict)

        assert len(config.enabled_models) == 1
        model_instance, model_config = config.enabled_models[0]

        # Verify model config
        assert model_config.fit.samples == 1000
        assert model_config.extra_kwargs == {"param1": "custom_value"}

    def test_model_config_yaml_roundtrip(self, tmp_path: Path):
        """Test ModelConfig YAML save/load roundtrip."""
        original_config = ModelConfig(
            fit=FitConfig(samples=1234, chains=3),
            main_effect_params=["alpha", "beta"],
            tag="roundtrip_test",
            link_function=cloglog_to_prob,
            extra_kwargs={"param1": "value1", "param2": 42},
        )

        json_path = tmp_path / "config.json"
        original_config.save(json_path)

        # Load back and verify
        import json

        with open(json_path) as f:
            saved_data = json.load(f)

        recreated_config = ModelConfig.from_dict(saved_data)

        assert recreated_config.fit.samples == 1234
        assert recreated_config.fit.chains == 3
        assert recreated_config.main_effect_params == ["alpha", "beta"]
        assert recreated_config.tag == "roundtrip_test"
        assert recreated_config.link_function == cloglog_to_prob
        assert recreated_config.extra_kwargs == {"param1": "value1", "param2": 42}


class TestModelRegistryIntegration:
    """Test model integration with registry system."""

    def test_model_registration_and_retrieval(self):
        """Test that models are properly registered and can be retrieved."""

        @model
        def registry_test_model(test_param: int = 10) -> Model:
            def model_impl(features: Features) -> None:
                check_features(features, ["obs"])
                assert test_param == 20  # Will be set when creating instance

            return model_impl

        # Should be able to retrieve from registry
        from hibayes.registry import RegistryInfo, registry_get

        info = RegistryInfo(type="model", name="registry_test_model")
        retrieved_builder = registry_get(info)

        assert retrieved_builder == registry_test_model

        # Should be able to create instance with parameters
        model_instance = retrieved_builder(test_param=20)

        # Test the model works
        features = {"obs": jnp.array([1, 0, 1])}
        model_instance(features)  # Should not raise

    def test_builtin_models_registration(self):
        """Test that built-in models are properly registered."""
        from hibayes.registry import RegistryInfo, registry_get

        # two_level_group_binomial should be registered
        info = RegistryInfo(type="model", name="two_level_group_binomial")
        retrieved_builder = registry_get(info)

        assert retrieved_builder == two_level_group_binomial

        # Should be able to create instance
        model_instance = retrieved_builder()
        assert callable(model_instance)

    @patch("hibayes.model.model_config._import_path")
    def test_custom_model_path_loading(self, mock_import_path):
        """Test loading custom models from file paths."""

        # Simulate importing custom models
        def mock_import_side_effect(path):
            # Simulate registering a custom model when path is imported
            @model
            def custom_imported_model() -> Model:
                def model_impl(features: Features) -> None:
                    pass

                return model_impl

            # The import would normally register the model
            pass

        mock_import_path.side_effect = mock_import_side_effect

        config_dict = {
            "path": "/fake/custom/models.py",
            "models": ["custom_imported_model"],
        }

        # This would normally work if the model was actually imported and registered
        with patch("hibayes.model.model_config.registry_get") as mock_registry_get:
            mock_model_builder = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_builder.return_value = mock_model_instance
            mock_registry_get.return_value = mock_model_builder

            config = ModelsToRunConfig.from_dict(config_dict)

            # Verify import was called
            mock_import_path.assert_called_once_with("/fake/custom/models.py")

            # Verify model was retrieved and instantiated
            mock_registry_get.assert_called_once()
            assert len(config.enabled_models) == 1


class TestModelErrorHandling:
    """Test error handling in model functionality."""

    def test_model_with_invalid_features(self):
        """Test model error handling with invalid features."""

        @model
        def strict_model() -> Model:
            def model_impl(features: Features) -> None:
                check_features(features, ["obs", "required_param"])
                # This model requires specific features

            return model_impl

        model_instance = strict_model()

        # Missing 'obs' should trigger decorator validation
        with pytest.raises(ValueError, match=r".*Model must have 'obs' in features.*"):
            model_instance({"other": jnp.array([1, 2, 3])})

        # Missing other required features should trigger model validation
        with pytest.raises(
            ValueError, match=r".*Missing required features: required_param.*"
        ):
            model_instance({"obs": jnp.array([1, 2, 3])})

    def test_model_builder_exceptions(self):
        """Test handling of exceptions in model builders."""

        @model
        def failing_builder(should_fail: bool = False) -> Model:
            if should_fail:
                raise ValueError("Builder failed!")

            def model_impl(features: Features) -> None:
                pass

            return model_impl

        # Should work when not failing
        model_instance = failing_builder(should_fail=False)
        assert callable(model_instance)

        # Should raise when builder fails
        with pytest.raises(ValueError, match="Builder failed!"):
            failing_builder(should_fail=True)

    def test_fit_with_invalid_model_analysis_state(self):
        """Test fit function with invalid model analysis state."""

        # Create minimal invalid state
        model_state = MagicMock()
        model_state.model_config.fit.method = "INVALID_METHOD"
        model_state.model_config.fit.init_strategy = "median"

        with pytest.raises(
            ValueError, match=r".*Unsupported inference method: INVALID_METHOD.*"
        ):
            fit(model_state)

    def test_models_to_run_config_missing_model(self):
        """Test ModelsToRunConfig with missing model in registry."""

        config_dict = {"models": ["NonExistentModel"]}

        with patch("hibayes.model.model_config.registry_get", side_effect=KeyError):
            with pytest.raises(KeyError):
                ModelsToRunConfig.from_dict(config_dict)

    def test_model_config_yaml_file_not_found(self):
        """Test ModelConfig.from_yaml with non-existent file."""

        with pytest.raises(FileNotFoundError):
            ModelConfig.from_yaml("/non/existent/file.yaml")

    def test_models_to_run_config_yaml_file_not_found(self):
        """Test ModelsToRunConfig.from_yaml with non-existent file."""

        with pytest.raises(FileNotFoundError):
            ModelsToRunConfig.from_yaml("/non/existent/file.yaml")


class TestModelAdvancedFeatures:
    """Test advanced model features and edge cases."""

    def test_model_with_complex_parameter_types(self):
        """Test models with complex parameter types."""

        @model
        def complex_param_model(
            param_dict: Dict[str, Any] = None,
            param_list: list = None,
            param_callable: Callable = None,
        ) -> Model:
            if param_dict is None:
                param_dict = {"default": "value"}
            if param_list is None:
                param_list = [1, 2, 3]
            if param_callable is None:

                def param_callable(x):
                    return x

            def model_impl(features: Features) -> None:
                check_features(features, ["obs"])
                # Use the complex parameters
                assert isinstance(param_dict, dict)
                assert isinstance(param_list, list)
                assert callable(param_callable)

            return model_impl

        # Test with default parameters
        model_instance = complex_param_model()
        features = {"obs": jnp.array([1, 0, 1])}
        model_instance(features)

        # Test with custom parameters
        custom_dict = {"custom": "value"}
        custom_list = [4, 5, 6]

        def custom_callable(x):
            return x * 2

        model_instance = complex_param_model(
            param_dict=custom_dict,
            param_list=custom_list,
            param_callable=custom_callable,
        )
        model_instance(features)

    def test_model_config_with_all_link_functions(self):
        """Test ModelConfig with all available link functions."""

        for link_name, link_func in LINK_FUNCTION_MAP.items():
            config = ModelConfig.from_dict({"link_function": link_name})
            assert config.link_function == link_func

            # Test that function works
            test_values = np.array([-2, -1, 0, 1, 2])
            result = config.link_function(test_values)
            assert isinstance(result, np.ndarray)
            assert len(result) == len(test_values)

    def test_fit_config_all_parameters(self):
        """Test FitConfig with all possible parameter combinations."""

        # Test all method types
        for method in ["NUTS", "HMC"]:
            config = FitConfig(method=method)
            assert config.method == method

        # Test boundary values
        config = FitConfig(
            samples=1,
            warmup=0,
            chains=1,
            target_accept=0.0,
            max_tree_depth=1,
        )

        assert config.samples == 1
        assert config.warmup == 0
        assert config.chains == 1
        assert config.target_accept == 0.0
        assert config.max_tree_depth == 1

    def test_model_analysis_state_with_coords_and_dims(self):
        """Test ModelAnalysisState with coords and dims."""

        # Create test data
        mock_model = MagicMock()
        features = {"obs": jnp.array([1, 0, 1, 0])}
        coords = {"group": ["A", "B"], "time": [1, 2, 3]}
        dims = {"group_effects": ["group"], "time_effects": ["time"]}

        model_state = ModelAnalysisState(
            model=mock_model,
            model_config=ModelConfig(),
            features=features,
            coords=coords,
            dims=dims,
        )

        assert model_state.coords == coords
        assert model_state.dims == dims

        # Test fit function uses coords and dims
        with (
            patch("hibayes.model.fit.MCMC"),
            patch("hibayes.model.fit.NUTS"),
            patch("hibayes.model.fit.jax.random.PRNGKey"),
            patch("hibayes.model.fit.az.from_numpyro") as mock_from_numpyro,
        ):
            mock_inference_data = MagicMock()
            mock_inference_data.extend = MagicMock()
            mock_from_numpyro.return_value = mock_inference_data

            fit(model_state)

            # Verify coords and dims were passed to arviz
            mock_from_numpyro.assert_called_once()
            call_kwargs = mock_from_numpyro.call_args[1]
            assert call_kwargs["coords"] == coords
            assert call_kwargs["dims"] == dims

    def test_model_with_numpyro_integration(self):
        """Test model that actually uses numpyro primitives."""

        @model
        def numpyro_integration_model() -> Model:
            def model_impl(features: Features) -> None:
                import numpyro
                import numpyro.distributions as dist

                # Simple beta-binomial model
                alpha = numpyro.sample("alpha", dist.Gamma(1, 1))
                beta = numpyro.sample("beta", dist.Gamma(1, 1))

                p = numpyro.sample("p", dist.Beta(alpha, beta))

                numpyro.sample(
                    "obs",
                    dist.Binomial(total_count=features["n_total"], probs=p),
                    obs=features["obs"],
                )

            return model_impl

        model_instance = numpyro_integration_model()

        features = {
            "obs": jnp.array([5, 3, 7, 2]),
            "n_total": jnp.array([10, 8, 12, 6]),
        }

        # This would work in actual numpyro context
        with patch("numpyro.sample"), patch("numpyro.distributions"):
            result = model_instance(features)
            assert result is None


class TestModelConfigurationFiles:
    """Test model configuration file handling."""

    def test_comprehensive_yaml_config(self, tmp_path: Path):
        """Test comprehensive YAML configuration loading."""

        yaml_content = {
            "path": ["/custom/models1.py", "/custom/models2.py"],
            "models": [
                "DefaultModel",
                {
                    "name": "CustomModel1",
                    "config": {
                        "fit": {
                            "method": "HMC",
                            "samples": 2000,
                            "warmup": 1000,
                            "chains": 2,
                            "seed": 42,
                            "progress_bar": False,
                            "target_accept": 0.9,
                            "max_tree_depth": 15,
                        },
                        "main_effect_params": ["alpha", "beta", "gamma"],
                        "tag": "experiment_v1",
                        "link_function": "probit",
                        "custom_param1": "value1",
                        "custom_param2": 123,
                        "custom_param3": [1, 2, 3],
                    },
                },
                {
                    "name": "CustomModel2",
                    "config": {
                        "tag": "baseline",
                        "link_function": "cloglog",
                    },
                },
            ],
        }

        yaml_file = tmp_path / "comprehensive_config.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        with (
            patch("hibayes.model.model_config._import_path") as mock_import,
            patch("hibayes.model.model_config.registry_get") as mock_registry_get,
        ):
            mock_model_builder = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_builder.return_value = mock_model_instance
            mock_registry_get.return_value = mock_model_builder

            config = ModelsToRunConfig.from_yaml(str(yaml_file))

            # Verify imports
            assert mock_import.call_count == 2
            mock_import.assert_any_call("/custom/models1.py")
            mock_import.assert_any_call("/custom/models2.py")

            # Verify models
            assert len(config.enabled_models) == 3

            # Check first model (simple)
            model1, config1 = config.enabled_models[0]
            assert isinstance(config1, ModelConfig)

            # Check second model (complex)
            model2, config2 = config.enabled_models[1]
            assert config2.fit.method == "HMC"
            assert config2.fit.samples == 2000
            assert config2.fit.chains == 2
            assert config2.main_effect_params == ["alpha", "beta", "gamma"]
            assert config2.tag == "experiment_v1"
            assert config2.link_function == probit_to_prob
            assert config2.extra_kwargs["custom_param1"] == "value1"
            assert config2.extra_kwargs["custom_param2"] == 123
            assert config2.extra_kwargs["custom_param3"] == [1, 2, 3]

            # Check third model
            model3, config3 = config.enabled_models[2]
            assert config3.tag == "baseline"
            assert config3.link_function == cloglog_to_prob

    def test_invalid_yaml_handling(self, tmp_path: Path):
        """Test handling of invalid YAML configurations."""

        # Test invalid YAML syntax
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [")

        with pytest.raises(yaml.YAMLError):
            ModelsToRunConfig.from_yaml(str(invalid_yaml))

        # Test missing file
        with pytest.raises(FileNotFoundError):
            ModelsToRunConfig.from_yaml(str(tmp_path / "nonexistent.yaml"))

    def test_model_config_json_save_load_cycle(self, tmp_path: Path):
        """Test complete save/load cycle with JSON."""

        original = ModelConfig(
            fit=FitConfig(method="HMC", samples=1500, chains=3, seed=123),
            main_effect_params=["param_a", "param_b"],
            tag="json_test",
            link_function=probit_to_prob,
            extra_kwargs={
                "string_param": "test_value",
                "int_param": 456,
                "float_param": 3.14,
                "list_param": [1, 2, 3],
                "dict_param": {"nested": "value"},
            },
        )

        # Save
        json_path = tmp_path / "model_config.json"
        original.save(json_path)

        # Load and recreate
        import json

        with open(json_path) as f:
            saved_data = json.load(f)

        recreated = ModelConfig.from_dict(saved_data)

        # Verify all fields match
        assert recreated.fit.method == "HMC"
        assert recreated.fit.samples == 1500
        assert recreated.fit.chains == 3
        assert recreated.fit.seed == 123
        assert recreated.main_effect_params == ["param_a", "param_b"]
        assert recreated.tag == "json_test"
        assert recreated.link_function == probit_to_prob
        assert recreated.extra_kwargs["string_param"] == "test_value"
        assert recreated.extra_kwargs["int_param"] == 456
        assert recreated.extra_kwargs["float_param"] == 3.14
        assert recreated.extra_kwargs["list_param"] == [1, 2, 3]
        assert recreated.extra_kwargs["dict_param"] == {"nested": "value"}


class TestModelPerformanceAndEdgeCases:
    """Test model performance characteristics and edge cases."""

    def test_model_with_large_feature_arrays(self):
        """Test model with large feature arrays."""

        @model
        def large_data_model() -> Model:
            def model_impl(features: Features) -> None:
                check_features(features, ["obs"])
                # Verify we can handle large arrays
                assert len(features["obs"]) == 10000

            return model_impl

        model_instance = large_data_model()

        # Create large feature array
        large_obs = jnp.ones(10000)
        features = {"obs": large_obs}

        model_instance(features)  # Should handle large arrays

    def test_model_with_empty_features(self):
        """Test model behavior with edge case features."""

        @model
        def edge_case_model() -> Model:
            def model_impl(features: Features) -> None:
                if len(features["obs"]) == 0:
                    raise ValueError("Cannot handle empty observations")

            return model_impl

        model_instance = edge_case_model()

        # Empty observations should be handled gracefully
        empty_features = {"obs": jnp.array([])}

        with pytest.raises(ValueError, match="Cannot handle empty observations"):
            model_instance(empty_features)

    def test_fit_config_merged_preserves_immutability(self):
        """Test that FitConfig.merged preserves immutability."""

        original = FitConfig(samples=1000, chains=2, seed=42)

        # Create multiple merged versions
        merged1 = original.merged(samples=2000)
        merged2 = original.merged(chains=4)
        merged3 = original.merged(seed=123, samples=3000)

        # Original should be unchanged
        assert original.samples == 1000
        assert original.chains == 2
        assert original.seed == 42

        # Each merged version should have correct changes
        assert merged1.samples == 2000
        assert merged1.chains == 2  # Unchanged
        assert merged1.seed == 42  # Unchanged

        assert merged2.samples == 1000  # Unchanged
        assert merged2.chains == 4
        assert merged2.seed == 42  # Unchanged

        assert merged3.samples == 3000
        assert merged3.chains == 2  # Unchanged
        assert merged3.seed == 123

    def test_models_to_run_config_duplicate_models(self):
        """Test ModelsToRunConfig with duplicate model specifications."""

        with patch("hibayes.model.model_config.registry_get") as mock_registry_get:
            mock_model_builder = MagicMock()
            mock_model_instance = MagicMock()
            mock_model_builder.return_value = mock_model_instance
            mock_registry_get.return_value = mock_model_builder

            config_dict = {
                "models": [
                    "TestModel",
                    "TestModel",  # Duplicate
                    {"name": "TestModel", "config": {"tag": "v1"}},  # Another duplicate
                ]
            }

            config = ModelsToRunConfig.from_dict(config_dict)

            # Should create separate instances for each specification
            assert len(config.enabled_models) == 3
            assert mock_model_builder.call_count == 3

    def test_concurrent_model_creation(self):
        """Test that models can be created concurrently."""
        import threading

        @model
        def thread_safe_model(thread_id: int = 0) -> Model:
            def model_impl(features: Features) -> None:
                check_features(features, ["obs"])
                # Each thread should get its own thread_id
                pass

            return model_impl

        results = []
        errors = []

        def create_model(thread_id):
            try:
                model_instance = thread_safe_model(thread_id=thread_id)
                features = {"obs": jnp.array([1, 0, 1])}
                model_instance(features)
                results.append(thread_id)
            except Exception as e:
                errors.append((thread_id, e))

        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=create_model, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(errors) == 0
        assert len(results) == 10
        assert set(results) == set(range(10))


class TestModelCustomFileIntegration:
    """Test loading custom models from files with YAML configuration."""

    def test_custom_model_file_with_yaml_args_integration(self, tmp_path: Path):
        """Test complete integration: custom model file + YAML config + argument passing."""

        # Create a temporary Python file with custom models
        custom_models_file = tmp_path / "custom_models.py"
        custom_models_file.write_text(
            '''
from typing import Dict, Any
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

from hibayes.model import model, Model
from hibayes.model.models import check_features
from hibayes.process import Features


@model
def custom_beta_binomial(
    alpha_prior: float = 1.0,
    beta_prior: float = 1.0,
    hierarchical: bool = False,
    group_effects: bool = False,
    custom_metadata: Dict[str, Any] = None
) -> Model:
    """Custom beta-binomial model with configurable priors and hierarchical structure."""

    if custom_metadata is None:
        custom_metadata = {"model_type": "beta_binomial", "version": "1.0"}

    def model_impl(features: Features) -> None:
        required_features = ["obs", "n_total"]
        if hierarchical:
            required_features.extend(["group_index", "num_groups"])

        check_features(features, required_features)

        if hierarchical and group_effects:
            # Hierarchical model with group effects
            mu_alpha = numpyro.sample("mu_alpha", dist.Gamma(alpha_prior, 1.0))
            mu_beta = numpyro.sample("mu_beta", dist.Gamma(beta_prior, 1.0))

            sigma_alpha = numpyro.sample("sigma_alpha", dist.HalfNormal(0.5))
            sigma_beta = numpyro.sample("sigma_beta", dist.HalfNormal(0.5))

            group_alpha = numpyro.sample(
                "group_alpha",
                dist.Gamma(mu_alpha, sigma_alpha).expand([features["num_groups"]])
            )
            group_beta = numpyro.sample(
                "group_beta",
                dist.Gamma(mu_beta, sigma_beta).expand([features["num_groups"]])
            )

            alpha = group_alpha[features["group_index"]]
            beta = group_beta[features["group_index"]]

        elif hierarchical:
            # Simple hierarchical model without group effects
            alpha = numpyro.sample("alpha", dist.Gamma(alpha_prior, 1.0))
            beta = numpyro.sample("beta", dist.Gamma(beta_prior, 1.0))

        else:
            # Non-hierarchical model
            alpha = numpyro.sample("alpha", dist.Gamma(alpha_prior, 1.0))
            beta = numpyro.sample("beta", dist.Gamma(beta_prior, 1.0))

        # Sample probability from Beta distribution
        p = numpyro.sample("p", dist.Beta(alpha, beta))

        # Store metadata as deterministic
        numpyro.deterministic("model_metadata", custom_metadata["version"])

        # Likelihood
        numpyro.sample(
            "obs",
            dist.Binomial(total_count=features["n_total"], probs=p),
            obs=features["obs"]
        )

    return model_impl


@model
def custom_normal_model(
    prior_mean: float = 0.0,
    prior_std: float = 1.0,
    likelihood_std: float = 1.0,
    transform_data: bool = False
) -> Model:
    """Custom normal model for testing different parameter types."""

    def model_impl(features: Features) -> None:
        check_features(features, ["obs"])

        # Transform data if requested
        obs_data = features["obs"]
        if transform_data:
            obs_data = obs_data * 2.0  # Simple transformation

        # Prior
        mu = numpyro.sample("mu", dist.Normal(prior_mean, prior_std))

        # Likelihood
        numpyro.sample(
            "obs",
            dist.Normal(mu, likelihood_std),
            obs=obs_data
        )

    return model_impl
'''
        )

        # Create YAML configuration that uses the custom models with various argument types
        yaml_config = tmp_path / "custom_models_config.yaml"
        yaml_config.write_text(
            """
path: {}
models:
  - name: custom_beta_binomial
    config:
      fit:
        method: NUTS
        samples: 1000
        warmup: 200
        chains: 2
        seed: 123
        progress_bar: false
        target_accept: 0.85
      main_effect_params: ["p", "alpha", "beta"]
      tag: "hierarchical_experiment"
      link_function: "logit"
      # Custom model arguments
      alpha_prior: 2.0
      beta_prior: 3.0
      hierarchical: true
      group_effects: true
      custom_metadata:
        model_type: "hierarchical_beta_binomial"
        version: "2.1"
        experiment_id: "exp_001"

  - name: custom_normal_model
    config:
      fit:
        method: HMC
        samples: 500
        chains: 1
        seed: 456
      tag: "normal_baseline"
      link_function: "identity"
      # Custom model arguments
      prior_mean: 5.0
      prior_std: 2.0
      likelihood_std: 0.5
      transform_data: true

  - name: custom_beta_binomial
    config:
      tag: "simple_version"
      # Different arguments for same model
      alpha_prior: 0.5
      beta_prior: 0.5
      hierarchical: false
      group_effects: false
      custom_metadata:
        model_type: "simple_beta_binomial"
        version: "1.0"
""".format(str(custom_models_file))
        )

        # Load the configuration
        config = ModelsToRunConfig.from_yaml(str(yaml_config))

        # Verify we have 3 model configurations
        assert len(config.enabled_models) == 3

        # Test first model (hierarchical beta-binomial)
        model1, config1 = config.enabled_models[0]
        assert callable(model1)
        assert config1.fit.method == "NUTS"
        assert config1.fit.samples == 1000
        assert config1.fit.warmup == 200
        assert config1.fit.chains == 2
        assert config1.fit.seed == 123
        assert config1.fit.progress_bar is False
        assert config1.fit.target_accept == 0.85
        assert config1.main_effect_params == ["p", "alpha", "beta"]
        assert config1.tag == "hierarchical_experiment"
        assert config1.link_function == logit_to_prob
        assert config1.extra_kwargs["alpha_prior"] == 2.0
        assert config1.extra_kwargs["beta_prior"] == 3.0
        assert config1.extra_kwargs["hierarchical"] is True
        assert config1.extra_kwargs["group_effects"] is True
        assert (
            config1.extra_kwargs["custom_metadata"]["model_type"]
            == "hierarchical_beta_binomial"
        )
        assert config1.extra_kwargs["custom_metadata"]["version"] == "2.1"
        assert config1.extra_kwargs["custom_metadata"]["experiment_id"] == "exp_001"

        # Test second model (normal model)
        model2, config2 = config.enabled_models[1]
        assert callable(model2)
        assert config2.fit.method == "HMC"
        assert config2.fit.samples == 500
        assert config2.fit.chains == 1
        assert config2.fit.seed == 456
        assert config2.tag == "normal_baseline"
        assert config2.link_function(np.array([1, 2, 3])).tolist() == [
            1,
            2,
            3,
        ]  # identity function
        assert config2.extra_kwargs["prior_mean"] == 5.0
        assert config2.extra_kwargs["prior_std"] == 2.0
        assert config2.extra_kwargs["likelihood_std"] == 0.5
        assert config2.extra_kwargs["transform_data"] is True

        # Test third model (simple beta-binomial)
        model3, config3 = config.enabled_models[2]
        assert callable(model3)
        assert config3.tag == "simple_version"
        assert config3.extra_kwargs["alpha_prior"] == 0.5
        assert config3.extra_kwargs["beta_prior"] == 0.5
        assert config3.extra_kwargs["hierarchical"] is False
        assert config3.extra_kwargs["group_effects"] is False
        assert (
            config3.extra_kwargs["custom_metadata"]["model_type"]
            == "simple_beta_binomial"
        )

        # Test that the models actually work with appropriate features

        # Test hierarchical model with group features
        hierarchical_features = {
            "obs": jnp.array([8, 6, 12, 4, 9, 7]),
            "n_total": jnp.array([10, 8, 15, 6, 12, 10]),
            "group_index": jnp.array([0, 0, 1, 1, 2, 2]),
            "num_groups": 3,
        }

        model_analysis_state = ModelAnalysisState(
            model=model1,
            model_config=config1,
            features=hierarchical_features,
            coords={"groups": ["A", "B", "C"]},
            dims={"group_effects": ["groups"]},
        )

        # Test that fit would work (with mocked MCMC)
        with (
            patch("hibayes.model.fit.MCMC") as mock_mcmc,
            patch("hibayes.model.fit.NUTS") as mock_nuts,
            patch("hibayes.model.fit.az.from_numpyro") as mock_from_numpyro,
        ):
            mock_kernel = MagicMock()
            mock_nuts.return_value = mock_kernel
            mock_mcmc_instance = MagicMock()
            mock_mcmc.return_value = mock_mcmc_instance
            mock_inference_data = MagicMock()
            mock_inference_data.extend = MagicMock()
            mock_from_numpyro.return_value = mock_inference_data

            fit(model_analysis_state)

            # Verify NUTS was configured with custom parameters from YAML
            mock_nuts.assert_called_once_with(
                model1,
                target_accept_prob=0.85,  # From YAML config
                max_tree_depth=10,  # Default
                init_strategy=ANY,  # init_strategy is a callable
            )

            # Verify MCMC was configured with YAML parameters
            mock_mcmc.assert_called_once_with(
                mock_kernel,
                num_samples=1000,  # From YAML
                num_warmup=200,  # From YAML
                num_chains=2,  # From YAML
                progress_bar=False,  # From YAML
                chain_method="parallel",  # Default
            )

            # Verify fit completed successfully
            assert model_analysis_state.is_fitted is True
            assert model_analysis_state.inference_data == mock_inference_data


class TestDummyCoding:
    """Test models with dummy coding (effect_coding_for_main_effects=False)."""

    def test_ordered_logistic_model_with_dummy_coding_no_error(self):
        """Test ordered_logistic_model with dummy coding doesn't raise errors."""
        from hibayes.model.models import ordered_logistic_model

        model_instance = ordered_logistic_model(
            categorical_effects=["grader"],
            effect_coding_for_main_effects=False,  # Use dummy coding
            num_classes=5,
        )

        features = {
            "obs": jnp.array([0, 1, 2, 1, 0]),
            "num_grader": 3,
            "grader_index": jnp.array([0, 0, 1, 1, 2]),
        }

        # Mock numpyro.sample to return appropriate values based on sample name
        def mock_sample_side_effect(name, *args, **kwargs):
            if name == "intercept":
                return jnp.array(0.0)  # Scalar
            elif name == "grader_effects_raw":
                return jnp.array([0.1, 0.2])  # Shape (2,) for n_levels - 1
            elif name == "first_cutpoint":
                return jnp.array(-2.0)  # Scalar
            elif name == "cutpoint_diffs":
                return jnp.array([0.5, 0.6, 0.7])  # Shape (num_classes - 2,) = (3,)
            else:
                # Default fallback
                return jnp.array(0.0)

        with (
            patch("numpyro.sample", side_effect=mock_sample_side_effect),
            patch("numpyro.deterministic"),
        ):
            result = model_instance(features)
            assert result is None

    def test_linear_group_binomial_with_dummy_coding_no_error(self):
        """Test linear_group_binomial with dummy coding doesn't raise errors."""
        from hibayes.model.models import linear_group_binomial

        model_instance = linear_group_binomial(
            main_effects=["group"],
            effect_coding_for_main_effects=False,  # Use dummy coding
        )

        features = {
            "obs": jnp.array([5, 3, 7, 2]),
            "n_total": jnp.array([10, 8, 12, 6]),
            "num_group": 3,
            "group_index": jnp.array([0, 0, 1, 1]),
        }

        # Mock numpyro.sample to return appropriate values based on sample name
        def mock_sample_side_effect(name, *args, **kwargs):
            if name == "intercept":
                return jnp.array(0.0)  # Scalar
            elif name == "group_effects_raw":
                return jnp.array([0.1, 0.2])  # Shape (2,) for n_levels - 1
            else:
                # Default fallback
                return jnp.array(0.0)

        with (
            patch("numpyro.sample", side_effect=mock_sample_side_effect),
            patch("numpyro.deterministic"),
        ):
            result = model_instance(features)
            assert result is None

    def test_dummy_coding_creates_deterministic_site(self):
        """Test that dummy coding creates deterministic site with reference category."""
        from hibayes.model.models import linear_group_binomial

        model_instance = linear_group_binomial(
            main_effects=["group"],
            effect_coding_for_main_effects=False,
        )

        features = {
            "obs": jnp.array([5, 3]),
            "n_total": jnp.array([10, 8]),
            "num_group": 3,
            "group_index": jnp.array([0, 1]),
        }

        # Mock numpyro.sample to return appropriate values based on sample name
        def mock_sample_side_effect(name, *args, **kwargs):
            if name == "intercept":
                return jnp.array(0.0)  # Scalar
            elif name == "group_effects_raw":
                return jnp.array([0.1, 0.2])  # Shape (2,) for n_levels - 1
            else:
                # Default fallback
                return jnp.array(0.0)

        with (
            patch("numpyro.sample", side_effect=mock_sample_side_effect) as mock_sample,
            patch("numpyro.deterministic") as mock_deterministic,
        ):
            model_instance(features)

            # Verify that a deterministic site was created for group_effects
            deterministic_calls = [c[0][0] for c in mock_deterministic.call_args_list]
            assert "group_effects" in deterministic_calls

            # Verify that the sample site uses a different name (not group_effects)
            sample_calls = [c[0][0] for c in mock_sample.call_args_list]
            # After fix: should use "group_effects_raw" or similar
            # Before fix: would use "group_effects" causing collision
            assert "group_effects" not in sample_calls, (
                "Sample site should not use 'group_effects' name"
            )
            assert "group_effects_raw" in sample_calls, (
                "Sample site should use 'group_effects_raw' name"
            )


class TestOrderedLogContinuousEffects:
    """Test ordered_logistic_model with continuous effects."""

    def setUp(self):
        """Set up test data for ordered_logistic with continuous effects."""
        self.features = {
                "obs": jnp.array([0, 1, 2, 1, 0]),
                "num_grader": 3,
                "grader_index": jnp.array([0, 0, 1, 1, 2]),
                "response_length": jnp.array([1.5, 2.0, 2.5, 1.0, 3.0]),
            }
    
    def _create_base_mock_sample(self, **extra_mocks):
        """Create a mock sample side effect with common mocks plus custom ones.
        
        Args:
            **extra_mocks: Additional name->value mappings to add to the mock
        """
        # Base mocks that are common across tests
        mocks = {
            "intercept": jnp.array(0.0),
            "grader_effects_constrained": jnp.array([0.1, 0.2]),
            "response_length_coef": jnp.array(0.5),
            "first_cutpoint": jnp.array(-2.0),
            "cutpoint_diffs": jnp.array([0.5, 0.6, 0.7]),
        }
        
        # Add any extra mocks provided
        mocks.update(extra_mocks)
        
        def mock_sample_side_effect(name, dist_obj, *args, **kwargs):
            return mocks.get(name, jnp.array(0.0))
        
        return mock_sample_side_effect

    def test_ordered_logistic_model_with_continuous_effects_no_error(self):
        """Test ordered_logistic_model with continuous effects doesn't raise errors."""
        from hibayes.model.models import ordered_logistic_model
        self.setUp()

        model_instance = ordered_logistic_model(
            main_effects=["grader"],
            continuous_effects=["response_length"],
            num_classes=5,
        )

        # Mock numpyro.sample to return appropriate values based on sample name
        with (
            patch("numpyro.sample", side_effect=self._create_base_mock_sample()),
            patch("numpyro.deterministic"),
        ):
            result = model_instance(self.features)
            assert result is None

    def test_ordered_logistic_model_with_cat_con_interactions(self):
        """Test that continuous interactions are handled without error."""
        from hibayes.model.models import ordered_logistic_model
        self.setUp()

        model_instance = ordered_logistic_model(
            main_effects=["grader"],
            continuous_effects=["response_length"],
            interactions=[("grader", "response_length")],
            num_classes=5,
        )

        # Add the interaction-specific mock
        mock_sample = self._create_base_mock_sample(
            grader_response_length_effects=jnp.array([0.05, 0.1, 0.15])
        )

        with (
            patch("numpyro.sample", side_effect=mock_sample),
            patch("numpyro.deterministic"),
        ):
            result = model_instance(self.features)
            assert result is None

    def test_ordered_logistic_model_with_con_con_interactions(self):
        """Test that continuous interactions are handled without error."""
        from hibayes.model.models import ordered_logistic_model
        self.setUp()

        model_instance = ordered_logistic_model(
            main_effects=["grader"],
            continuous_effects=["response_length", "time_taken"],
            interactions=[("response_length", "time_taken")],
            num_classes=5,
        )

        # add time_taken as a feature
        self.features["time_taken"] = jnp.array([0.5, 1.0, 1.5, 0.2, 0.8])

        # Add the interaction-specific mock
        mock_sample = self._create_base_mock_sample(
            time_taken_coef=jnp.array(0.3),
            response_length_time_taken_effects=jnp.array(0.02) 
        )

        with (
            patch("numpyro.sample", side_effect=mock_sample),
            patch("numpyro.deterministic"),
        ):
            result = model_instance(self.features)
            assert result is None