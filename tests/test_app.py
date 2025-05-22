import pytest
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image, UnidentifiedImageError
import yaml # For mocking config load

# Import the functions/classes to be tested
# Add parent directory to sys.path to allow importing app
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Mock config data for app.py
mock_app_config_data = {
    "paths": {"final_model_path": "./mocked_custom_model_path_for_app"}
    # Add other keys if app.py starts using them directly
}

# Using a fixture to provide the app module with necessary mocks active.
@pytest.fixture
def app_module_fixture():
    """Fixture to provide the 'app' module, re-imported with mocks for each test."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_app_config_data))) as mock_file, \
         patch('yaml.safe_load', return_value=mock_app_config_data) as mock_yaml_load, \
         patch('transformers.pipeline', MagicMock(name="mock_pipeline_fixture")) as mock_pipeline, \
         patch('torch.compile', lambda x, **kwargs: x) as mock_compile, \
         patch('app.logger', MagicMock(name="mock_logger_fixture")) as mock_logger:

        if 'app' in sys.modules:
            del sys.modules['app'] # Force reload to apply mocks to module-level code
        
        import app

        # Attach mocks to the app module for easy access in tests
        app.mock_pipeline_fixture = mock_pipeline
        app.mock_logger_fixture = mock_logger
        app.mock_app_config = mock_app_config_data # Make mocked config accessible if needed
        
        yield app 

        if 'app' in sys.modules:
            del sys.modules['app']


# Tests for load_model()
# ======================

def test_load_model_successful_custom(app_module_fixture):
    """Tests successful loading of the custom model specified in (mocked) config."""
    app = app_module_fixture
    mock_pipeline = app.mock_pipeline_fixture
    expected_model_path = app.mock_app_config["paths"]["final_model_path"]
    
    mock_custom_classifier_instance = MagicMock(name="custom_model_instance")
    mock_custom_classifier_instance.model = MagicMock(name="actual_torch_model")

    mock_pipeline.reset_mock() # Reset from the initial load_model() call during app import
    mock_pipeline.return_value = mock_custom_classifier_instance
    
    app.load_model() 

    mock_pipeline.assert_called_once_with(
        task="image-classification",
        model=expected_model_path, # Uses path from mocked config
        torch_dtype=torch.bfloat16, 
        device="cuda",
    )
    assert app.classifier == mock_custom_classifier_instance
    assert mock_custom_classifier_instance.model.attn_implementation == "flash_attention_2"
    app.mock_logger_fixture.info.assert_any_call("Custom model loaded and configured successfully.")


def test_load_model_failure_custom_success_fallback(app_module_fixture):
    """Tests fallback to default model if custom model loading fails."""
    app = app_module_fixture
    mock_pipeline = app.mock_pipeline_fixture
    mock_logger = app.mock_logger_fixture
    expected_custom_model_path = app.mock_app_config["paths"]["final_model_path"]
    
    mock_fallback_classifier_instance = MagicMock(name="fallback_model_instance")
    
    mock_pipeline.reset_mock()
    mock_pipeline.side_effect = [
        Exception("Custom model load error"), 
        mock_fallback_classifier_instance    
    ]
    
    app.load_model()

    assert mock_pipeline.call_count == 2
    mock_pipeline.assert_any_call(
        task="image-classification",
        model=expected_custom_model_path, 
        torch_dtype=torch.bfloat16,
        device="cuda",
    )
    mock_pipeline.assert_any_call(
        task="image-classification",
        model="google/vit-base-patch16-224", 
        torch_dtype="auto",
    )
    assert app.classifier == mock_fallback_classifier_instance
    mock_logger.exception.assert_any_call(f"Error loading the custom model: Custom model load error")
    mock_logger.info.assert_any_call("Default Hugging Face model 'google/vit-base-patch16-224' loaded successfully.")

def test_load_model_failure_both(app_module_fixture):
    """Tests behavior when both custom and fallback model loading fail."""
    app = app_module_fixture
    mock_pipeline = app.mock_pipeline_fixture
    mock_logger = app.mock_logger_fixture

    mock_pipeline.reset_mock()
    mock_pipeline.side_effect = [
        Exception("Custom model load error"),   
        Exception("Fallback model load error") 
    ]
    
    app.load_model()

    assert mock_pipeline.call_count == 2
    assert app.classifier is None 
    mock_logger.exception.assert_any_call(f"Error loading the custom model: Custom model load error")
    mock_logger.exception.assert_any_call("Error loading the fallback Hugging Face model: Fallback model load error")
    mock_logger.critical.assert_any_call("Both custom and fallback model loading failed. Application cannot start.")


# Tests for predict()
# ===================

# We need a classifier mock for these tests.
# The app_module_fixture will run app.load_model() on import.
# We can either rely on that by ensuring the fixture sets up a working classifier,
# or re-assign app.classifier in each test for more fine-grained control.

@patch('app.Image.open') 
def test_predict_successful(mock_pil_open, app_module_fixture):
    """Tests successful prediction flow."""
    app = app_module_fixture
    
    # Ensure a mock classifier is set up by the fixture's load_model or set it manually.
    # If load_model in fixture resulted in a classifier:
    if app.classifier is None: # If fixture's default load_model failed, set a mock one
        app.classifier = MagicMock(name="test_classifier_for_predict")
        app.classifier.return_value = [{'label': 'AI', 'score': 0.9}]

    mock_image_object = MagicMock(spec=Image.Image) # PIL.Image spec
    mock_pil_open.return_value = mock_image_object
    
    image_path = "dummy_path.jpg"
    result = app.predict(image_path)

    mock_pil_open.assert_called_once_with(image_path)
    app.classifier.assert_called_once_with(mock_image_object)
    assert result == {'AI': 0.9}
    app.mock_logger_fixture.info.assert_any_call(f"Image '{os.path.basename(image_path)}' opened successfully.")

def test_predict_image_path_none(app_module_fixture):
    """Tests predict function when image_path is None."""
    app = app_module_fixture
    app.classifier = MagicMock() 

    result = app.predict(None)
    assert result == {"Error": "No image uploaded."}
    app.mock_logger_fixture.warning.assert_any_call("No image path provided to predict function.")

def test_predict_classifier_none(app_module_fixture):
    """Tests predict function when classifier is not loaded."""
    app = app_module_fixture
    app.classifier = None 

    result = app.predict("dummy_path.jpg")
    assert result == {"Error": "Model not loaded. Please check server logs."}
    app.mock_logger_fixture.error.assert_any_call("Classifier model is not loaded. Cannot perform prediction.")

@patch('app.Image.open', side_effect=FileNotFoundError("File missing"))
def test_predict_file_not_found(mock_pil_open, app_module_fixture):
    """Tests predict function when image file is not found."""
    app = app_module_fixture
    app.classifier = MagicMock() 

    image_path = "nonexistent.jpg"
    result = app.predict(image_path)
    
    mock_pil_open.assert_called_once_with(image_path)
    assert result == {"Error": f"Image file not found: {os.path.basename(image_path)}"}
    app.mock_logger_fixture.error.assert_any_call(f"Image file not found at path: {image_path}")

@patch('app.Image.open', side_effect=UnidentifiedImageError("Bad image"))
def test_predict_unidentified_image_error(mock_pil_open, app_module_fixture):
    """Tests predict function when image file is corrupted or unsupported."""
    app = app_module_fixture
    app.classifier = MagicMock()

    image_path = "corrupt.jpg"
    result = app.predict(image_path)

    mock_pil_open.assert_called_once_with(image_path)
    assert "Cannot open or read image file" in result["Error"]
    app.mock_logger_fixture.error.assert_any_call(f"Cannot identify image file, it might be corrupted or an unsupported format: {image_path}")

@patch('app.Image.open')
def test_predict_exception_during_classification(mock_pil_open, app_module_fixture):
    """Tests predict function when an exception occurs during classification."""
    app = app_module_fixture
    
    mock_classifier_instance = MagicMock(name="failing_classifier")
    mock_classifier_instance.side_effect = Exception("Classification failed") 
    app.classifier = mock_classifier_instance

    mock_image_object = MagicMock(spec=Image.Image) # PIL.Image spec
    mock_pil_open.return_value = mock_image_object
    
    image_path = "dummy_path.jpg"
    result = app.predict(image_path)

    mock_pil_open.assert_called_once_with(image_path)
    mock_classifier_instance.assert_called_once_with(mock_image_object)
    assert result == {"Error": "An unexpected error occurred during prediction."}
    app.mock_logger_fixture.exception.assert_any_call(f"An unexpected error occurred during prediction for image '{image_path}': Classification failed")

# Regarding Gradio interface (iface):
# Testing Gradio's gr.Interface() itself is usually out of scope for unit tests,
# as it involves UI components and their interactions.
# However, we can test that it's created, or that the fallback interface is created
# if app.classifier is None *after* app.load_model() has run.

# The fixture `app_module_fixture` reloads `app` module, so `app.load_model()` runs.
# We can inspect `app.iface` based on the mocked outcomes of `app.load_model()`
# when the app module is imported by the fixture.

def test_gradio_interface_creation_model_loaded(app_module_fixture):
    """Tests that the main Gradio interface is created when model loads successfully."""
    app = app_module_fixture
    # The fixture 'app_module_fixture' mocks config loading.
    # We need to ensure that the default pipeline mock in this fixture leads to a successful model load.
    mock_pipeline = app.mock_pipeline_fixture
    mock_pipeline.reset_mock() # Clear any calls from the initial import within the fixture
    
    # Simulate a successful pipeline creation for the load_model call made during app import or direct call
    mock_successful_classifier = MagicMock(name="successful_classifier_for_iface")
    mock_successful_classifier.model = MagicMock() # Mimic structure expected by torch.compile
    mock_pipeline.return_value = mock_successful_classifier
    
    app.load_model() # Re-run load_model to ensure classifier is set based on this test's mocks
    
    assert app.classifier is not None, "Classifier should be loaded for this test."
    
    # Since app.iface is defined at module level based on app.classifier's state *after the initial load_model()*:
    # To properly test this, the fixture itself needs to ensure 'app.classifier' is not None.
    # This test becomes more of an assertion on the state set by the fixture if it's designed for success.
    # Let's assume the fixture `app_module_fixture` is set up for a successful load by default.
    # We might need to adjust the main fixture to ensure `app.load_model` is successful by default.
    
    # For the current `app_module_fixture`, it re-imports `app`.
    # If `app.mock_pipeline_fixture` (the one from the fixture) doesn't have a side_effect
    # that makes it fail, `load_model` will likely succeed.
    
    # This check is a bit indirect for the module-level `iface` definition.
    # A direct test of `iface` creation logic would require re-executing that part of app.py
    # with `app.classifier` in a controlled state.
    import gradio as gr # Import gr here for isinstance check
    if app.classifier is not None: # If model loading was successful
        assert isinstance(app.iface, gr.Interface)
        assert app.iface.title == "Custom Image Classifier"
    else:
        pytest.fail("Model did not load successfully in app_module_fixture for UI test.")


@pytest.fixture
def app_module_model_load_fails_fixture():
    """Fixture to provide 'app' module where model loading is guaranteed to fail."""
    with patch('builtins.open', mock_open(read_data=yaml.dump(mock_app_config_data))), \
         patch('yaml.safe_load', return_value=mock_app_config_data), \
         patch('transformers.pipeline', MagicMock(name="mock_pipeline_fixture_fails", side_effect=Exception("Force model load fail"))) as mock_pipeline, \
         patch('torch.compile', lambda x, **kwargs: x), \
         patch('app.logger', MagicMock(name="mock_logger_fixture_fails")) as mock_logger:
        
        if 'app' in sys.modules:
            del sys.modules['app']
        import app 

        app.mock_pipeline_fixture = mock_pipeline
        app.mock_logger_fixture = mock_logger
        yield app
        
        if 'app' in sys.modules:
            del sys.modules['app']

def test_gradio_interface_creation_model_failed(app_module_model_load_fails_fixture):
    """Tests that the fallback Gradio interface is created when model loading fails."""
    app = app_module_model_load_fails_fixture
    import gradio as gr # Import gr here for isinstance check
    
    assert app.classifier is None, "Classifier should be None due to load_model failure in fixture."
    assert isinstance(app.iface, gr.Blocks) # Check if the fallback UI (gr.Blocks) is created
    app.mock_logger_fixture.critical.assert_any_call("Gradio interface created with an error message because classifier is None.")

# Note: `torch.bfloat16` is used in app.py. Tests assume `torch` is importable.
# The mock for `transformers.pipeline` handles the actual model loading part.
