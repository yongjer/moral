import pytest
from unittest.mock import patch, MagicMock
from PIL import Image, UnidentifiedImageError
import os

# Import functions from inference.py
# Add parent directory to sys.path to allow importing inference
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import yaml # For mocking config load
from unittest.mock import mock_open # For mocking config file open

# Mock config data for inference.py
# Needs to provide 'paths.final_model_path'
mock_inference_config_data = {
    "paths": {"final_model_path": "./mocked_model_path_for_inference"}
}

# Fixture to provide the inference module
@pytest.fixture
def inference_module_fixture():
    """Fixture that provides the 'inference' module, re-imported with mocks for config and logger."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_inference_config_data))) as mock_config_file, \
         patch('yaml.safe_load', return_value=mock_inference_config_data) as mock_yaml_load, \
         patch('transformers.pipeline', MagicMock(name="mock_pipeline_inference_fixture")) as mock_pipeline, \
         patch('inference.logger', MagicMock(name="mock_logger_inference_fixture")) as mock_logger:
        
        if 'inference' in sys.modules: # Ensure module re-import to apply mocks
            del sys.modules['inference']
        import inference
        
        # Attach mocks and mocked config to the module for easy access in tests
        inference.mock_pipeline_fixture = mock_pipeline
        inference.mock_logger_fixture = mock_logger
        inference.mocked_config = mock_inference_config_data # Make config accessible
        
        yield inference
        
        if 'inference' in sys.modules:
            del sys.modules['inference']

# Tests for load_model() in inference.py
# ======================================

def test_inference_load_model_successful(inference_module_fixture):
    """Tests successful model loading in inference.py."""
    inference = inference_module_fixture
    mock_pipeline = inference.mock_pipeline_fixture
    
    mock_classifier_instance = MagicMock(name="inference_model_instance")
    mock_pipeline.reset_mock() 
    mock_pipeline.return_value = mock_classifier_instance
    
    model_path_to_test = "./some_model_path" # Test with an explicit path
    classifier = inference.load_model(model_path=model_path_to_test)

    mock_pipeline.assert_called_once_with(
        task="image-classification",
        model=model_path_to_test,
    )
    assert classifier == mock_classifier_instance
    inference.mock_logger_fixture.info.assert_any_call(f"Model from '{model_path_to_test}' loaded successfully.")

def test_inference_load_model_failure(inference_module_fixture):
    """Tests model loading failure in inference.py."""
    inference = inference_module_fixture
    mock_pipeline = inference.mock_pipeline_fixture
    
    mock_pipeline.reset_mock()
    mock_pipeline.side_effect = Exception("Model loading failed for inference")
    
    model_path_to_test = "./a_bad_model_path"
    classifier = inference.load_model(model_path=model_path_to_test)

    mock_pipeline.assert_called_once_with(
        task="image-classification",
        model=model_path_to_test,
    )
    assert classifier is None
    inference.mock_logger_fixture.exception.assert_any_call(f"Error loading the model from '{model_path_to_test}': Model loading failed for inference")


# Tests for predict_image() in inference.py
# =========================================

def test_predict_image_successful(inference_module_fixture):
    """Tests successful image prediction."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture

    mock_classifier = MagicMock(name="mock_classifier_for_predict_image")
    mock_prediction_result = [{'label': 'CLASS_A', 'score': 0.95}]
    mock_classifier.return_value = mock_prediction_result
    
    mock_image_object = MagicMock(spec=Image.Image) # PIL.Image spec

    # Use the default image path from inference.py for this test
    default_image_path = inference._DEFAULT_EXAMPLE_IMAGE_PATH

    with patch('inference.Image.open', return_value=mock_image_object) as mock_pil_open:
        predictions = inference.predict_image(mock_classifier, image_path=default_image_path)

    mock_pil_open.assert_called_once_with(default_image_path)
    mock_classifier.assert_called_once_with(mock_image_object)
    assert predictions == mock_prediction_result
    mock_logger.info.assert_any_call(f"Image '{default_image_path}' loaded successfully.")
    mock_logger.info.assert_any_call(f"Predictions: {mock_prediction_result}")


def test_predict_image_classifier_none(inference_module_fixture):
    """Tests predict_image when the classifier is None."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    
    predictions = inference.predict_image(None, image_path="any_image.jpg")
    
    assert predictions is None
    mock_logger.error.assert_called_once_with("Classifier model is not loaded. Cannot perform prediction.")


@patch('inference.Image.open', side_effect=FileNotFoundError("File does not exist"))
def test_predict_image_file_not_found(mock_pil_open, inference_module_fixture):
    """Tests predict_image when the image file is not found."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    mock_classifier = MagicMock()
    image_path = "nonexistent.jpg"

    predictions = inference.predict_image(mock_classifier, image_path=image_path)

    assert predictions is None
    mock_pil_open.assert_called_once_with(image_path)
    mock_logger.error.assert_called_once_with(f"Image file not found at path: {image_path}")


@patch('inference.Image.open', side_effect=UnidentifiedImageError("Corrupted image"))
def test_predict_image_unidentified_image(mock_pil_open, inference_module_fixture):
    """Tests predict_image with a corrupted or unidentified image file."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    mock_classifier = MagicMock()
    image_path = "corrupt.jpg"

    predictions = inference.predict_image(mock_classifier, image_path=image_path)

    assert predictions is None
    mock_pil_open.assert_called_once_with(image_path)
    mock_logger.error.assert_called_once_with(f"Cannot identify image file, it might be corrupted or an unsupported format: {image_path}")


@patch('inference.Image.open')
def test_predict_image_prediction_exception(mock_pil_open, inference_module_fixture):
    """Tests predict_image when the classifier raises an exception during prediction."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    
    mock_classifier = MagicMock(name="failing_classifier_for_predict")
    mock_classifier.side_effect = Exception("Prediction engine failure")
    
    mock_image_object = MagicMock(spec=Image.Image) # PIL.Image spec
    mock_pil_open.return_value = mock_image_object
    image_path = "image_that_fails.jpg"

    predictions = inference.predict_image(mock_classifier, image_path=image_path)

    assert predictions is None
    mock_pil_open.assert_called_once_with(image_path)
    mock_classifier.assert_called_once_with(mock_image_object)
    mock_logger.exception.assert_called_once_with(f"An error occurred during model prediction for image '{image_path}': Prediction engine failure")


# Tests for main() in inference.py (High-level)
# ============================================

@patch('inference.load_model')
@patch('inference.predict_image')
@patch('inference.os.path.exists')
@patch('inference.Image.new') # For dummy image creation
@patch('inference.sys.exit') 
def test_inference_main_successful_flow(mock_sys_exit, mock_image_new, mock_os_path_exists, mock_predict_image, mock_load_model, inference_module_fixture):
    """Tests the main function of inference.py for a successful flow."""
    inference = inference_module_fixture 
    mock_logger = inference.mock_logger_fixture
    expected_model_path = inference.mocked_config["paths"]["final_model_path"]
    default_example_image_path = inference._DEFAULT_EXAMPLE_IMAGE_PATH

    mock_classifier_instance = MagicMock(name="main_flow_classifier")
    mock_load_model.return_value = mock_classifier_instance
    mock_predict_image.return_value = [{'label': 'OK', 'score': 1.0}]
    mock_os_path_exists.return_value = True 

    inference.main()

    mock_load_model.assert_called_once_with(model_path=expected_model_path)
    mock_predict_image.assert_called_once_with(mock_classifier_instance, image_path=default_example_image_path)
    mock_logger.info.assert_any_call(f"Successfully processed '{default_example_image_path}'.")
    mock_sys_exit.assert_not_called()


@patch('inference.load_model', return_value=None) 
@patch('inference.predict_image')
@patch('inference.sys.exit')
def test_inference_main_model_load_fails(mock_sys_exit, mock_predict_image, mock_load_model, inference_module_fixture):
    """Tests main function behavior when model loading fails."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    expected_model_path = inference.mocked_config["paths"]["final_model_path"]
    
    inference.main()

    mock_load_model.assert_called_once_with(model_path=expected_model_path)
    mock_predict_image.assert_not_called() 
    mock_logger.error.assert_any_call("Model could not be loaded. Exiting inference script.")
    mock_sys_exit.assert_called_once_with(1)


@patch('inference.load_model')
@patch('inference.predict_image')
@patch('inference.os.path.exists', return_value=False) 
@patch('inference.Image.new') 
@patch('PIL.Image.Image.save') 
@patch('inference.sys.exit')
def test_inference_main_dummy_image_creation(mock_image_save, mock_sys_exit, mock_image_new, mock_os_path_exists, mock_predict_image, mock_load_model, inference_module_fixture):
    """Tests main function's dummy image creation logic."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    default_example_image_path = inference._DEFAULT_EXAMPLE_IMAGE_PATH

    mock_classifier_instance = MagicMock()
    mock_load_model.return_value = mock_classifier_instance
    mock_predict_image.return_value = True 

    mock_dummy_image_instance = MagicMock(spec=Image.Image) # PIL.Image spec
    mock_image_new.return_value = mock_dummy_image_instance

    inference.main()
    
    mock_os_path_exists.assert_called_once_with(default_example_image_path)
    mock_image_new.assert_called_once_with('RGB', (100, 100), color='red')
    mock_dummy_image_instance.save.assert_called_once_with(default_example_image_path)
    mock_logger.info.assert_any_call(f"Created a dummy image at '{default_example_image_path}' for demonstration.")
    mock_predict_image.assert_called_once() 
    mock_sys_exit.assert_not_called()


@patch('inference.load_model')
@patch('inference.predict_image', return_value=None) 
@patch('inference.os.path.exists', return_value=True)
@patch('inference.sys.exit')
def test_inference_main_prediction_fails(mock_sys_exit, mock_os_path_exists, mock_predict_image, mock_load_model, inference_module_fixture):
    """Tests main function behavior when prediction fails (returns None)."""
    inference = inference_module_fixture
    mock_logger = inference.mock_logger_fixture
    default_example_image_path = inference._DEFAULT_EXAMPLE_IMAGE_PATH
    
    mock_classifier_instance = MagicMock()
    mock_load_model.return_value = mock_classifier_instance

    inference.main()

    mock_load_model.assert_called_once()
    mock_predict_image.assert_called_once()
    mock_logger.error.assert_any_call(f"Failed to process '{default_example_image_path}'.")
    mock_sys_exit.assert_not_called() 
    
# Note: The tests for `inference.main()` now correctly use the mocked config
# for `model_path_from_config` and the constant `_DEFAULT_EXAMPLE_IMAGE_PATH`.
# The fixture `inference_module_fixture` ensures that `inference.py` loads the mocked config.
