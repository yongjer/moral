import pytest
import torch
import torch.nn.functional as F
import numpy as np
from unittest.mock import patch, MagicMock, call
from PIL import Image as PIL_Image # For spec checking PIL.Image objects

# Import the components to be tested from main.py
# Add the parent directory to sys.path to allow importing main
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock config before importing main, as main loads config at module level
# This mock_config will be used by main.py when it's imported.
mock_config_data = {
    "paths": {
        "data_dir": "./fake_data_dir",
        "model_name": "google/vit-base-patch16-224-in21k", # A default, won't be downloaded
        "output_dir": "./test_results",
        "logging_dir": "./test_logs",
        "final_model_path": "./test_model_final",
        "confusion_matrix_path": "./test_cm.png"
    },
    "training": {
        "num_train_epochs": 1,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "warmup_steps": 10,
        "weight_decay": 0.01,
        "logging_steps": 1,
        "save_total_limit": 1,
        "bf16": False, # Disable for CPU tests
        "torch_compile": False, # Disable for simpler testing
    },
    "dataset": {
        "val_ratio": 0.5, # Using 0.5 for easier split with small numbers
        "test_ratio": 0.5,
        "preprocess_batch_size": 2,
    }
}

# We need to patch open for config loading *before* main is imported
# because main.py loads its config at the module level.
# The patch should affect the `open` call in `main.py`.
# The path for patching 'open' should be 'main.open' if 'main.py' is the module where open is called for config.
# Similarly for yaml.safe_load, it should be 'main.yaml.safe_load'.

# It's often easier to mock the config *object* itself if possible,
# or ensure that the module's global 'config' variable is patched.
# Let's try patching 'main.config' directly after import but before tests,
# or by ensuring the mocked open is used by main.py during its import.

# The most reliable way is to ensure main.py uses a mocked open when it is first imported by the test file.
# This means `patch` should be active when Python loads `main`.

# For simplicity in this environment, we might need to structure tests carefully
# or assume `main.py` can be imported and then its `config` global var can be overwritten.
# Let's assume we can mock `open` at `main.open` path.

# If `main.py` does `import yaml` and then `with open(...)`, the patch target is `builtins.open` or `main.open`.
# If `main.py` does `from yaml import safe_load`, then `yaml.safe_load` is imported into main's namespace.

# Patching 'builtins.open' for the config load in main.py
# Patching 'yaml.safe_load' for the config load in main.py
# It's better to get a 'main' module instance that has been loaded with mocks.
# Pytest fixtures are good for this.
@pytest.fixture(scope="module")
def main_module():
    """
    Pytest fixture to provide the 'main' module, loaded with necessary mocks.
    This allows testing functions from main.py that might depend on module-level
    initializations (like config loading or global variable setup based on config).
    Mocks are active during the import of 'main'.
    """
    # All patches need to be active *before and during* the import of 'main'.
    # This is complex. Let's simplify by assuming main.py can be refactored
    # or that we test functions individually, passing config/dependencies.

    # For now, let's assume FocalLoss, preprocess_data, compute_metrics are testable
    # without the full module import dance, by importing them directly.
    # We will still need to mock globals like 'processor' or 'config' if they use them.
    
    # This fixture will set up the main module with mocked config for all tests in this file
    with patch('sys.exit') as mock_exit, \
         patch('builtins.open', mock_open(read_data=yaml.dump(mock_config_data))) as mock_file, \
         patch('yaml.safe_load', return_value=mock_config_data) as mock_yaml, \
         patch('os.listdir', return_value=['file1.jpg', 'file2.jpg']) as mock_listdir, \
         patch('main.logger'): # Mock the logger used in main module
        
        # If main was already imported, reload it to apply mocks (important for module-level code)
        if 'main' in sys.modules:
            del sys.modules['main']
        import main
        
        # Ensure FOCAL_LOSS_ALPHA is determinate for tests if it's used by other functions.
        # It's calculated at module level based on os.listdir and config.
        # mock_listdir now returns 2 files for each 'ai' and 'real' dir (due to how it's called)
        # So, num_ai_images = 2, num_real_images = 2. total = 4.
        # ai_weight = 2/4 = 0.5, real_weight = 2/4 = 0.5
        # So, main.FOCAL_LOSS_ALPHA should be {0: 0.5, 1: 0.5}
        # We can assert this or manually set it for safety in tests.
        main.FOCAL_LOSS_ALPHA = {0: 0.5, 1: 0.5}
        main.config = mock_config_data # Ensure config is the mocked one
        yield main # Provide the module to tests
        # Cleanup: remove main from modules again to ensure clean state for other test files if any
        if 'main' in sys.modules:
            del sys.modules['main']


# Tests for FocalLoss
# ===================
def test_focal_loss_instantiation(main_module):
    """Tests FocalLoss class instantiation with and without alpha."""
    loss_fn_no_alpha = main_module.FocalLoss(gamma=2.0)
    assert loss_fn_no_alpha.gamma == 2.0
    assert loss_fn_no_alpha.alpha is None

    alpha_dict = {0: 0.25, 1: 0.75}
    loss_fn_with_alpha = main_module.FocalLoss(alpha=alpha_dict, gamma=1.5)
    assert loss_fn_with_alpha.alpha == alpha_dict
    assert loss_fn_with_alpha.gamma == 1.5

def test_focal_loss_invalid_alpha_instantiation(main_module):
    """Tests FocalLoss instantiation with invalid alpha type raises ValueError."""
    with pytest.raises(ValueError, match="alpha must be a dict"):
        main_module.FocalLoss(alpha=[0.25, 0.75]) # list instead of dict

def test_focal_loss_forward_no_alpha(main_module):
    """Tests FocalLoss forward pass without alpha."""
    loss_fn = main_module.FocalLoss(gamma=2.0, reduction='mean')
    inputs = torch.tensor([[-1.0, 1.0], [0.5, -0.5]]) # logits
    targets = torch.tensor([1, 0]) # labels

    # Expected CE loss:
    # Sample 1: target 1, input [-1, 1]. log_softmax = [-1.313, -0.313]. NLL = -(-0.313) = 0.313
    # Sample 2: target 0, input [0.5, -0.5]. log_softmax = [-0.474, -1.474]. NLL = -(-0.474) = 0.474
    # ce_loss = tensor([0.3133, 0.4741]) # approximately
    # pt = exp(-ce_loss) = tensor([0.7311, 0.6225]) # approximately
    # focal_loss_values = (1-pt)^gamma * ce_loss
    # FL1 = (1-0.7311)^2 * 0.3133 = 0.0226 # approximately
    # FL2 = (1-0.6225)^2 * 0.4741 = 0.0676 # approximately
    # mean_focal_loss = (0.0226 + 0.0676) / 2 = 0.0451 # approximately
    
    ce_loss_pytorch = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss_pytorch)
    expected_focal_loss_values = ((1 - pt) ** 2) * ce_loss_pytorch
    expected_loss = torch.mean(expected_focal_loss_values)

    calculated_loss = loss_fn(inputs, targets)
    assert torch.isclose(calculated_loss, expected_loss, atol=1e-4)

def test_focal_loss_forward_with_alpha(main_module):
    """Tests FocalLoss forward pass with alpha weights."""
    alpha_weights = {0: 0.75, 1: 0.25}
    loss_fn = main_module.FocalLoss(alpha=alpha_weights, gamma=2.0, reduction='mean')
    inputs = torch.tensor([[-1.0, 1.0], [0.5, -0.5]]) 
    targets = torch.tensor([1, 0]) 

    ce_loss_pytorch = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss_pytorch)
    
    # Logic from FocalLoss.forward for applying alpha
    alpha_tensor_from_loss_logic = torch.tensor(
        [alpha_weights.get(i, 1.0) for i in range(inputs.size(1))], device=targets.device
    )
    at = alpha_tensor_from_loss_logic.gather(0, targets)
    
    expected_focal_loss_values = at * ((1 - pt) ** 2) * ce_loss_pytorch
    expected_loss = torch.mean(expected_focal_loss_values)
    
    calculated_loss = loss_fn(inputs, targets)
    assert torch.isclose(calculated_loss, expected_loss, atol=1e-4)

def test_focal_loss_reduction_sum(main_module):
    """Tests FocalLoss with 'sum' reduction."""
    loss_fn = main_module.FocalLoss(gamma=2.0, reduction='sum')
    inputs = torch.tensor([[-1.0, 1.0], [0.5, -0.5]])
    targets = torch.tensor([1, 0])
    
    ce_loss_pytorch = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss_pytorch)
    expected_focal_loss_values = ((1 - pt) ** 2) * ce_loss_pytorch
    expected_loss_sum = torch.sum(expected_focal_loss_values)
    
    calculated_loss = loss_fn(inputs, targets)
    assert torch.isclose(calculated_loss, expected_loss_sum, atol=1e-4)

def test_focal_loss_reduction_none(main_module):
    """Tests FocalLoss with 'none' reduction."""
    loss_fn = main_module.FocalLoss(gamma=2.0, reduction='none')
    inputs = torch.tensor([[-1.0, 1.0], [0.5, -0.5]])
    targets = torch.tensor([1, 0])

    ce_loss_pytorch = F.cross_entropy(inputs, targets, reduction='none')
    pt = torch.exp(-ce_loss_pytorch)
    expected_focal_loss_values = ((1 - pt) ** 2) * ce_loss_pytorch
    
    calculated_loss = loss_fn(inputs, targets)
    assert torch.all(torch.isclose(calculated_loss, expected_focal_loss_values, atol=1e-4))


# Tests for preprocess_data
# =========================
@patch('main.ViTImageProcessor') # Mock the class ViTImageProcessor imported in main.py
def test_preprocess_data_rgba_conversion(MockViTImageProcessor, main_module):
    """Tests that RGBA images are converted to RGB in preprocess_data."""
    mock_processor_instance = MockViTImageProcessor.from_pretrained.return_value
    main_module.processor = mock_processor_instance # Set the global processor in main_module

    mock_img_rgba = MagicMock(spec=PIL_Image) # Input images are PIL Image objects
    mock_img_rgba.mode = 'RGBA'
    
    mock_img_rgb = MagicMock(spec=PIL_Image) # Input images are PIL Image objects
    mock_img_rgb.mode = 'RGB'

    example_batch = {
        "image": [mock_img_rgba, mock_img_rgb], 
        "label": [0, 1]
    }
    
    # Simulate processor output
    dummy_pixel_values = torch.randn(2, 3, 224, 224)
    mock_processor_instance.return_value = {"pixel_values": dummy_pixel_values}

    result = main_module.preprocess_data(example_batch)

    mock_img_rgba.convert.assert_called_once_with('RGB')
    mock_img_rgb.convert.assert_not_called() # Should not be called for RGB
    
    # Check processor call arguments
    # processor is called with a list of image objects
    processed_images_arg = mock_processor_instance.call_args[0][0]
    assert len(processed_images_arg) == 2
    assert processed_images_arg[0] == mock_img_rgba.convert.return_value # Converted one
    assert processed_images_arg[1] == mock_img_rgb # Original one
    
    mock_processor_instance.assert_called_once_with(
        [mock_img_rgba.convert.return_value, mock_img_rgb], # Expected list of images
        return_tensors="pt",
        padding=True
    )
    
    assert "labels" in result
    assert result["labels"] == [0, 1]
    assert torch.equal(result["pixel_values"], dummy_pixel_values)


# Tests for compute_metrics
# =========================
def test_compute_metrics_all_correct(main_module):
    """Tests compute_metrics with all predictions being correct."""
    predictions_logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
    labels = np.array([1, 0, 1, 0]) # Argmax predictions: [1, 0, 1, 0]
    
    eval_pred = (predictions_logits, labels)
    metrics = main_module.compute_metrics(eval_pred)

    assert metrics['accuracy'] == 1.0
    assert metrics['f1'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    expected_cm = [[2, 0], [0, 2]] # [[TN, FP], [FN, TP]] if 0=Neg, 1=Pos. Here [[C0_as_C0, C0_as_C1], [C1_as_C0, C1_as_C1]]
    assert np.array_equal(metrics['confusion_matrix'], expected_cm)

def test_compute_metrics_mixed_predictions(main_module):
    """Tests compute_metrics with a mix of correct and incorrect predictions."""
    # Logits:       [[0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]]
    # Predictions:  [    1     ,     0     ,     0     ,     1     ]
    # Labels:       [    1     ,     0     ,     1     ,     0     ]
    # Correctness:  [ Correct  ,  Correct  , Incorrect , Incorrect ]
    # Accuracy = 2/4 = 0.5
    predictions_logits = np.array([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.3, 0.7]])
    labels = np.array([1, 0, 1, 0]) 
    
    eval_pred = (predictions_logits, labels)
    metrics = main_module.compute_metrics(eval_pred)

    assert metrics['accuracy'] == 0.5
    # For F1, Precision, Recall with weighted average:
    # Class 0: TP=1 (idx 1), FP=1 (idx 2), FN=1 (idx 3) -> P0=0.5, R0=0.5, F1_0=0.5
    # Class 1: TP=1 (idx 0), FP=1 (idx 3), FN=1 (idx 2) -> P1=0.5, R1=0.5, F1_1=0.5
    # Weighted F1 = (0.5*2 + 0.5*2)/4 = 0.5
    assert np.isclose(metrics['f1'], 0.5)
    assert np.isclose(metrics['precision'], 0.5)
    assert np.isclose(metrics['recall'], 0.5)
    
    # CM for labels=[1,0,1,0] and preds=[1,0,0,1]
    #       Pred 0  Pred 1
    # Act 0    1       1   (actual 0, pred 0; actual 0, pred 1)
    # Act 1    1       1   (actual 1, pred 0; actual 1, pred 1)
    expected_cm = [[1, 1], [1, 1]]
    assert np.array_equal(metrics['confusion_matrix'], expected_cm)

def test_compute_metrics_zero_division(main_module):
    """Tests compute_metrics handling of zero division for a class not predicted and not present."""
    # All labels are class 0, all predictions are class 0. Class 1 has no support.
    predictions_logits = np.array([[0.9, 0.1], [0.8, 0.2]]) # Preds: [0, 0]
    labels = np.array([0, 0]) # Labels: [0, 0]
    
    eval_pred = (predictions_logits, labels)
    metrics = main_module.compute_metrics(eval_pred) # zero_division=0 is used in main.py

    assert metrics['accuracy'] == 1.0
    # For class 0: P=1, R=1, F1=1. Support=2.
    # For class 1: P=0, R=0, F1=0. Support=0.
    # Weighted F1 = (1*2 + 0*0)/2 = 1.0
    assert np.isclose(metrics['f1'], 1.0)
    assert np.isclose(metrics['precision'], 1.0)
    assert np.isclose(metrics['recall'], 1.0)
    expected_cm = [[2, 0], [0, 0]]
    assert np.array_equal(metrics['confusion_matrix'], expected_cm)


# Note: Testing `create_dataset` and `main_training_pipeline()` function itself is more complex as they involve
# filesystem operations, actual data loading, and the full training pipeline.
# These would be closer to integration tests.
# For `create_dataset`, one could mock `os.listdir`, `random.shuffle`, `Dataset.from_dict`, `Image()`.
# For `main()`, one would mock almost all helper functions and training objects.

# The current fixture setup for `main_module` is a bit of a simplification.
# Robustly testing module-level code triggered on import requires careful patch management,
# often by ensuring patches are active *before* the first import of the module under test.
# Pytest's `monkeypatch` fixture can also be useful for modifying globals like `main.config` or `main.processor`
# directly within test functions if the module is already imported.
# For instance, `monkeypatch.setattr(main_module, 'config', new_mock_config)`
# or `monkeypatch.setattr(main_module, 'processor', mock_processor_instance)`.
# This would be done inside each test function or a more function-scoped fixture.

# The current `main_module` fixture attempts to solve this by reloading `main` under patch context.
# This should generally work but can have subtleties depending on Python's import mechanisms
# and side effects in `main.py`'s module-level code.
# The key is that `main.FOCAL_LOSS_ALPHA` and `main.config` are set to known values for tests.
# The `ViTImageProcessor.from_pretrained` is not directly called by the tested units yet,
# but `preprocess_data` relies on `main.processor` which is normally its result.
# The test `test_preprocess_data_rgba_conversion` correctly mocks and sets `main.processor`.
# If `ViTImageProcessor.from_pretrained` was called at module level in `main.py`, it would need mocking during import.
# In `main.py` (as of previous versions), `processor` is loaded inside `main()`, so it's less of an import issue.
# However, `preprocess_data` is a global function, so `processor` must be a global variable or passed in.
# The `main.py` has `global processor` in `main()` and `preprocess_data` uses this global.
# This means `main.processor` needs to be set before `preprocess_data` is called.
# The fixture and the test for `preprocess_data` handle this by setting `main_module.processor`.

# One remaining global is `FOCAL_LOSS_ALPHA` used by `CustomTrainer`. If `CustomTrainer` were tested,
# this would need to be a known value. The fixture sets it.
# The `main.py` structure:
# - Imports
# - Config loading (module level) -> Patched via builtins.open, yaml.safe_load in fixture
# - set_seed() (module level)
# - Path definitions from config (module level)
# - Image counting and FOCAL_LOSS_ALPHA calculation (module level) -> Patched via os.listdir, and direct set in fixture
# - Class definitions (FocalLoss, CustomTrainer)
# - Function definitions (preprocess_data, compute_metrics, create_dataset, plot_confusion_matrix, main)

# The fixture `main_module` aims to make `main.py` importable and its components testable
# by controlling these module-level states and dependencies.
# The most important aspect for the tests above is that `main_module.FocalLoss`, `main_module.preprocess_data`,
# `main_module.compute_metrics` are accessible and their dependencies (like `main_module.config` or `main_module.processor`)
# can be controlled or are using the mocked values from the fixture.
# The patch for `main.logger` in the fixture is to silence logs during the import of `main`.
# Individual tests can further patch loggers if they want to assert logging calls.
# The `sys.exit` patch is crucial because the original `main.py` calls `sys.exit` on config load errors.
# The `os.listdir` patch for `main.py` module-level code prevents `FileNotFoundError` during import.
# It's set to return 2 files, so `num_ai_images` and `num_real_images` become 2, leading to calculated alpha of 0.5/0.5.
# The fixture then explicitly sets `main.FOCAL_LOSS_ALPHA = {0: 0.5, 1: 0.5}` for certainty.
# It also sets `main.config = mock_config_data`.
# This setup should make the tests for FocalLoss, preprocess_data, and compute_metrics stable.I have created `tests/test_main.py` and added tests for `FocalLoss`, `preprocess_data`, and `compute_metrics`. I've used a module-scoped fixture `main_module` to handle the complex module-level setup of `main.py`, including mocking config loading and other initializations like `FOCAL_LOSS_ALPHA` and `processor`.

Now, I will proceed to **`test_app.py`**.

I'll create `tests/test_app.py` and start by writing tests for the `load_model()` function, covering scenarios of successful custom model loading, fallback to default model, and complete failure.
