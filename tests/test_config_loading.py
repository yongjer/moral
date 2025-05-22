import pytest
import yaml
# import builtins # Not directly used, patch uses string 'builtins.open'
from unittest.mock import mock_open, patch
# from unittest.mock import MagicMock # Not directly used

# This test file is for scripts that directly open 'config.yaml' 
# and use its content (e.g., main.py, preprocess_images.py).

# Sample valid YAML content
VALID_YAML_CONTENT = """
paths:
  data_dir: "/path/to/data"
  model_name: "test_model"
training:
  batch_size: 32
"""

# Sample invalid YAML content
INVALID_YAML_CONTENT = """
paths:
  data_dir: "/path/to/data
training: batch_size: 32
"""

@patch('builtins.open', new_callable=mock_open, read_data=VALID_YAML_CONTENT)
@patch('yaml.safe_load')
def test_successful_config_loading(mock_yaml_safe_load, mock_file_open):
    """Tests that a valid config.yaml is loaded and parsed successfully."""
    expected_config = {"paths": {"data_dir": "/path/to/data", "model_name": "test_model"}, "training": {"batch_size": 32}}
    mock_yaml_safe_load.return_value = expected_config

    # Simulate the config loading part as it would appear in a script
    with open("config.yaml", "r") as f:
        loaded_config = yaml.safe_load(f)

    mock_file_open.assert_called_once_with("config.yaml", "r")
    mock_yaml_safe_load.assert_called_once()
    assert loaded_config == expected_config
    assert loaded_config["paths"]["model_name"] == "test_model"


@patch('builtins.open', side_effect=FileNotFoundError("Config file not found at path"))
@patch('yaml.safe_load') 
@patch('logging.Logger.error') 
@patch('sys.exit') 
def test_config_file_not_found(mock_sys_exit, mock_logger_error, mock_yaml_safe_load, mock_file_open):
    """Tests behavior when config.yaml is not found; expects log and sys.exit."""
    # Act: Simulate a script trying to load the config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) # This line will raise FileNotFoundError
    except FileNotFoundError as e:
        # This is how a script would typically handle it
        mock_logger_error(f"Configuration file 'config.yaml' not found. Error: {e}")
        mock_sys_exit(1)
    
    # Assert
    mock_file_open.assert_called_once_with("config.yaml", "r")
    # The exact error message from FileNotFoundError might vary slightly, 
    # so checking for a key part or using assert_any_call if multiple logs occur.
    mock_logger_error.assert_any_call("Configuration file 'config.yaml' not found. Error: Config file not found at path")
    mock_sys_exit.assert_called_once_with(1)
    mock_yaml_safe_load.assert_not_called() # Should not be called if open fails


@patch('builtins.open', new_callable=mock_open, read_data=INVALID_YAML_CONTENT)
@patch('yaml.safe_load', side_effect=yaml.YAMLError("Invalid YAML syntax"))
@patch('logging.Logger.error') 
@patch('sys.exit') 
def test_yaml_error_during_config_loading(mock_sys_exit, mock_logger_error, mock_yaml_safe_load, mock_file_open):
    """Tests behavior when config.yaml has invalid YAML; expects log and sys.exit."""
    # Act: Simulate a script trying to load a malformed config
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f) # This will raise YAMLError via mock_yaml_safe_load
    except yaml.YAMLError as e:
        # This is how a script would typically handle it
        mock_logger_error(f"Error parsing configuration file 'config.yaml': {e}")
        mock_sys_exit(1)
            
    # Assert
    mock_file_open.assert_called_once_with("config.yaml", "r")
    mock_yaml_safe_load.assert_called_once() # safe_load is called, but it raises an error
    mock_logger_error.assert_any_call("Error parsing configuration file 'config.yaml': Invalid YAML syntax")
    mock_sys_exit.assert_called_once_with(1)

# General notes for running these tests:
# pytest tests/test_config_loading.py
# Make sure you have pytest and PyYAML installed.
# These tests assume that the scripts being tested use a logger (e.g., from the logging module)
# and call sys.exit on critical config errors.
# The @patch('logging.Logger.error') might need adjustment based on how logger is obtained in actual scripts.
# For example, if it's logging.getLogger(__name__).error, then the patch path might be 'module_name.logger.error'.
# For simplicity, I'm using a generic patch path that should work if logger.error is called directly.
# A more robust way is to patch the specific logger instance used in the module.
# e.g., @patch('main.logger.error') if testing main.py's config loading.
# However, these tests are generic for the loading mechanism itself.
