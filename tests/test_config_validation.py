"""Tests for configuration key validation across all config classes."""

from __future__ import annotations

import pytest

from hibayes.check.checker_config import CheckerConfig
from hibayes.communicate.communicate_config import CommunicateConfig
from hibayes.model.model_config import ModelsToRunConfig
from hibayes.platform.config import PlatformConfig
from hibayes.process.process_config import ProcessConfig


class TestModelsToRunConfigValidation:
    """Test key validation for ModelsToRunConfig."""

    def test_valid_keys_accepted(self):
        """Test that valid keys are accepted without error."""
        config = ModelsToRunConfig.from_dict({"models": []})
        assert config is not None

    def test_unknown_key_raises_error(self):
        """Test that unknown keys raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ModelsToRunConfig.from_dict({"unknown_key": "value"})

        assert "Unknown keys in ModelsToRunConfig" in str(exc_info.value)
        assert "unknown_key" in str(exc_info.value)

    def test_typo_suggests_correction(self):
        """Test that typos in keys suggest corrections."""
        with pytest.raises(ValueError) as exc_info:
            ModelsToRunConfig.from_dict({"paths": "some/path"})

        error_msg = str(exc_info.value)
        assert "paths" in error_msg
        assert "did you mean 'path'" in error_msg

    def test_multiple_unknown_keys(self):
        """Test that multiple unknown keys are all reported."""
        with pytest.raises(ValueError) as exc_info:
            ModelsToRunConfig.from_dict({"paths": "x", "model": []})

        error_msg = str(exc_info.value)
        assert "paths" in error_msg
        assert "model" in error_msg

    def test_none_config_accepted(self):
        """Test that None config is accepted (uses defaults)."""
        config = ModelsToRunConfig.from_dict(None)
        assert config is not None

    def test_empty_config_accepted(self):
        """Test that empty config is accepted (uses defaults)."""
        config = ModelsToRunConfig.from_dict({})
        assert config is not None


class TestProcessConfigValidation:
    """Test key validation for ProcessConfig."""

    def test_valid_keys_accepted(self):
        """Test that valid keys are accepted without error."""
        config = ProcessConfig.from_dict({"processors": []})
        assert config is not None

    def test_unknown_key_raises_error(self):
        """Test that unknown keys raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ProcessConfig.from_dict({"unknown_key": "value"})

        assert "Unknown keys in ProcessConfig" in str(exc_info.value)
        assert "unknown_key" in str(exc_info.value)

    def test_typo_suggests_correction(self):
        """Test that typos in keys suggest corrections."""
        with pytest.raises(ValueError) as exc_info:
            ProcessConfig.from_dict({"processor": []})

        error_msg = str(exc_info.value)
        assert "processor" in error_msg
        assert "did you mean 'processors'" in error_msg

    def test_none_config_accepted(self):
        """Test that None config is accepted (uses defaults)."""
        config = ProcessConfig.from_dict(None)
        assert config is not None

    def test_empty_config_accepted(self):
        """Test that empty config is accepted (uses defaults)."""
        config = ProcessConfig.from_dict({})
        assert config is not None


class TestPlatformConfigValidation:
    """Test key validation for PlatformConfig."""

    def test_valid_keys_accepted(self):
        """Test that valid keys are accepted without error."""
        config = PlatformConfig.from_dict({
            "device_type": "cpu",
            "num_devices": 4,
            "gpu_memory_fraction": 0.8,
            "chain_method": "sequential",
        })
        assert config is not None
        assert config.device_type == "cpu"
        assert config.num_devices == 4

    def test_unknown_key_raises_error(self):
        """Test that unknown keys raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            PlatformConfig.from_dict({"unknown_key": "value"})

        assert "Unknown keys in PlatformConfig" in str(exc_info.value)
        assert "unknown_key" in str(exc_info.value)

    def test_typo_suggests_correction(self):
        """Test that typos in keys suggest corrections."""
        with pytest.raises(ValueError) as exc_info:
            PlatformConfig.from_dict({"device_types": "cpu"})

        error_msg = str(exc_info.value)
        assert "device_types" in error_msg
        assert "did you mean 'device_type'" in error_msg

    def test_none_config_accepted(self):
        """Test that None config is accepted (uses defaults)."""
        config = PlatformConfig.from_dict(None)
        assert config is not None

    def test_empty_config_accepted(self):
        """Test that empty config is accepted (uses defaults)."""
        config = PlatformConfig.from_dict({})
        assert config is not None


class TestCommunicateConfigValidation:
    """Test key validation for CommunicateConfig."""

    def test_valid_keys_accepted(self):
        """Test that valid keys are accepted without error."""
        config = CommunicateConfig.from_dict({"communicators": []})
        assert config is not None

    def test_unknown_key_raises_error(self):
        """Test that unknown keys raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CommunicateConfig.from_dict({"unknown_key": "value"})

        assert "Unknown keys in CommunicateConfig" in str(exc_info.value)
        assert "unknown_key" in str(exc_info.value)

    def test_typo_suggests_correction(self):
        """Test that typos in keys suggest corrections."""
        with pytest.raises(ValueError) as exc_info:
            CommunicateConfig.from_dict({"communicator": []})

        error_msg = str(exc_info.value)
        assert "communicator" in error_msg
        assert "did you mean 'communicators'" in error_msg

    def test_none_config_accepted(self):
        """Test that None config is accepted (uses defaults)."""
        config = CommunicateConfig.from_dict(None)
        assert config is not None

    def test_empty_config_accepted(self):
        """Test that empty config is accepted (uses defaults)."""
        config = CommunicateConfig.from_dict({})
        assert config is not None


class TestCheckerConfigValidation:
    """Test key validation for CheckerConfig."""

    def test_valid_keys_accepted(self):
        """Test that valid keys are accepted without error."""
        config = CheckerConfig.from_dict({"checkers": []})
        assert config is not None

    def test_unknown_key_raises_error(self):
        """Test that unknown keys raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CheckerConfig.from_dict({"unknown_key": "value"})

        assert "Unknown keys in CheckerConfig" in str(exc_info.value)
        assert "unknown_key" in str(exc_info.value)

    def test_typo_suggests_correction(self):
        """Test that typos in keys suggest corrections."""
        with pytest.raises(ValueError) as exc_info:
            CheckerConfig.from_dict({"checker": []})

        error_msg = str(exc_info.value)
        assert "checker" in error_msg
        assert "did you mean 'checkers'" in error_msg

    def test_paths_typo_suggests_path(self):
        """Test that 'paths' typo suggests 'path'."""
        with pytest.raises(ValueError) as exc_info:
            CheckerConfig.from_dict({"paths": "some/path"})

        error_msg = str(exc_info.value)
        assert "paths" in error_msg
        assert "did you mean 'path'" in error_msg

    def test_none_config_accepted(self):
        """Test that None config is accepted (uses defaults)."""
        config = CheckerConfig.from_dict(None)
        assert config is not None

    def test_empty_config_accepted(self):
        """Test that empty config is accepted (uses defaults)."""
        config = CheckerConfig.from_dict({})
        assert config is not None


class TestValidationHelperFunction:
    """Test the _validate_config_keys helper function behavior."""

    def test_error_message_includes_allowed_keys(self):
        """Test that error messages include the list of allowed keys."""
        with pytest.raises(ValueError) as exc_info:
            ModelsToRunConfig.from_dict({"bad_key": "value"})

        error_msg = str(exc_info.value)
        assert "Allowed keys are:" in error_msg
        assert "path" in error_msg
        assert "models" in error_msg

    def test_no_suggestion_for_unrelated_key(self):
        """Test that completely unrelated keys don't get suggestions."""
        with pytest.raises(ValueError) as exc_info:
            ModelsToRunConfig.from_dict({"xyzabc123": "value"})

        error_msg = str(exc_info.value)
        assert "xyzabc123" in error_msg
        # Should not have "did you mean" for unrelated keys
        assert "did you mean" not in error_msg
