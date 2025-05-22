import pytest
import os
from unittest.mock import patch, MagicMock, mock_open
from PIL import Image, UnidentifiedImageError

# Import the functions to be tested
# This assumes preprocess_images.py is in the parent directory or Python path
# Adjust the import path if your structure is different, e.g., from project_name import preprocess_images
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocess_images import convert_image_to_jpeg, process_image_folder, HEIF_SUPPORT_ENABLED


@pytest.fixture
def mock_image():
    """Fixture to create a mock PIL Image object."""
    img = MagicMock(spec=Image.Image)
    img.mode = 'RGB' # Default mode
    img.format = 'JPEG' # Default format
    return img

# Tests for convert_image_to_jpeg
# ================================

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove')
def test_convert_image_to_jpeg_rgb_mode(mock_os_remove, mock_image_open, mock_image):
    """Test conversion of an image already in RGB mode."""
    mock_image.mode = 'RGB'
    mock_image_open.return_value.__enter__.return_value = mock_image # Simulate 'with Image.open(...) as img:'

    result = convert_image_to_jpeg("source.jpg", "target.jpg", delete_original=False)

    mock_image_open.assert_called_once_with("source.jpg")
    mock_image.convert.assert_not_called() # Should not be called for RGB
    mock_image.save.assert_called_once_with("target.jpg", "JPEG", quality=90)
    mock_os_remove.assert_not_called()
    assert result is True

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove')
def test_convert_image_to_jpeg_rgba_mode(mock_os_remove, mock_image_open, mock_image):
    """Test conversion of an RGBA image, should be converted to RGB."""
    mock_image.mode = 'RGBA'
    mock_image_open.return_value.__enter__.return_value = mock_image

    result = convert_image_to_jpeg("source.png", "target.jpg", delete_original=True)

    mock_image_open.assert_called_once_with("source.png")
    mock_image.convert.assert_called_once_with('RGB')
    mock_image.save.assert_called_once_with("target.jpg", "JPEG", quality=90)
    mock_os_remove.assert_called_once_with("source.png") # delete_original is True
    assert result is True

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove')
def test_convert_image_to_jpeg_p_mode(mock_os_remove, mock_image_open, mock_image):
    """Test conversion of a P (paletted) mode image."""
    mock_image.mode = 'P'
    mock_image_open.return_value.__enter__.return_value = mock_image

    result = convert_image_to_jpeg("source.gif", "target.jpg", delete_original=False)

    mock_image_open.assert_called_once_with("source.gif")
    mock_image.convert.assert_called_once_with('RGB')
    mock_image.save.assert_called_once_with("target.jpg", "JPEG", quality=90)
    assert result is True

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove')
def test_convert_image_to_jpeg_delete_original_false(mock_os_remove, mock_image_open, mock_image):
    """Test delete_original=False."""
    mock_image.mode = 'RGB'
    mock_image_open.return_value.__enter__.return_value = mock_image

    convert_image_to_jpeg("source.jpg", "target.jpg", delete_original=False)
    mock_os_remove.assert_not_called()

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove')
def test_convert_image_to_jpeg_delete_original_true_different_files(mock_os_remove, mock_image_open, mock_image):
    """Test delete_original=True when source and target are different."""
    mock_image.mode = 'RGB'
    mock_image_open.return_value.__enter__.return_value = mock_image

    convert_image_to_jpeg("source.jpg", "target.jpg", delete_original=True)
    mock_os_remove.assert_called_once_with("source.jpg")

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove')
def test_convert_image_to_jpeg_delete_original_true_same_files(mock_os_remove, mock_image_open, mock_image):
    """Test delete_original=True but source and target are the same (should not delete)."""
    mock_image.mode = 'RGB'
    mock_image_open.return_value.__enter__.return_value = mock_image

    convert_image_to_jpeg("source.jpg", "source.jpg", delete_original=True)
    mock_os_remove.assert_not_called() # Critical: do not delete if source is same as target

@patch('preprocess_images.Image.open', side_effect=UnidentifiedImageError("Cannot identify image"))
@patch('preprocess_images.logger.error') # Assuming logger is used as in the refactored script
def test_convert_image_to_jpeg_unidentified_image_error(mock_logger_error, mock_image_open):
    """Test UnidentifiedImageError handling."""
    result = convert_image_to_jpeg("corrupt.jpg", "target.jpg")
    mock_logger_error.assert_called_with(
        "Pillow (even with HEIF support if enabled) cannot identify image file: 'corrupt.jpg'. It might be corrupted or an unsupported format."
    )
    assert result is False

@patch('preprocess_images.Image.open', side_effect=FileNotFoundError("File not found"))
@patch('preprocess_images.logger.error')
def test_convert_image_to_jpeg_file_not_found(mock_logger_error, mock_image_open):
    """Test FileNotFoundError handling."""
    result = convert_image_to_jpeg("nonexistent.jpg", "target.jpg")
    mock_logger_error.assert_called_with("File not found during conversion: 'nonexistent.jpg'. Skipping.")
    assert result is False

@patch('preprocess_images.Image.open')
@patch('preprocess_images.os.remove', side_effect=OSError("Cannot delete"))
@patch('preprocess_images.logger.error')
def test_convert_image_to_jpeg_os_error_on_remove(mock_logger_error, mock_os_remove, mock_image_open, mock_image):
    """Test OSError during os.remove."""
    mock_image.mode = 'RGBA' # Needs conversion, so delete will be attempted
    mock_image_open.return_value.__enter__.return_value = mock_image

    result = convert_image_to_jpeg("source.png", "target.jpg", delete_original=True)
    
    mock_image.save.assert_called_once() # Ensure save was attempted
    mock_os_remove.assert_called_once_with("source.png")
    mock_logger_error.assert_called_with("Could not delete original file 'source.png': Cannot delete")
    assert result is True # Conversion itself was successful

@patch('preprocess_images.Image.open')
@patch('preprocess_images.logger.exception') # Use logger.exception for unexpected errors
def test_convert_image_to_jpeg_exception_on_save(mock_logger_exception, mock_image_open, mock_image):
    """Test a generic Exception during image.save."""
    mock_image.mode = 'RGB'
    mock_image_open.return_value.__enter__.return_value = mock_image
    mock_image.save.side_effect = Exception("Disk full")

    result = convert_image_to_jpeg("source.jpg", "target.jpg")

    mock_logger_exception.assert_called_with("An unexpected error occurred while converting 'source.jpg': Disk full")
    assert result is False


# Tests for process_image_folder
# ==============================
# These are more complex to unit test thoroughly. We'll focus on key logic.

@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open') # To inspect image format
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True) # Test with HEIF support
def test_process_image_folder_heif_conversion(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test that HEIF images trigger conversion when HEIF support is enabled."""
    # Setup os.walk mock
    mock_os_walk.return_value = [
        ("/data", [], ["test.heic", "image.jpg"]),
    ]

    # Setup Image.open mock for the HEIF file
    mock_heif_image = MagicMock(spec=Image.Image)
    mock_heif_image.format = "HEIF"
    mock_heif_image.mode = "RGB"

    # Setup Image.open mock for the JPG file (should not be HEIF)
    mock_jpg_image = MagicMock(spec=Image.Image)
    mock_jpg_image.format = "JPEG"
    mock_jpg_image.mode = "RGB"

    # Make Image.open return the correct mock image based on filename
    def side_effect_image_open(filepath):
        if filepath.endswith(".heic"):
            return mock_heif_image
        elif filepath.endswith(".jpg"):
            return mock_jpg_image
        raise FileNotFoundError # Should not happen with mock_os_walk
    
    # The __enter__ is needed because Image.open is used in a 'with' statement
    mock_image_open_global.return_value.__enter__.side_effect = side_effect_image_open


    mock_convert_jpeg.return_value = True # Assume conversion is successful

    process_image_folder("/data", target_extension="jpg") # Updated parameter name

    # Assert for HEIF file
    # Expected call: convert_image_to_jpeg('/data/test.heic', '/data/test.jpg', delete_original=True)
    # The delete_original depends on logic: (original_filepath != new_filepath or is_heif_misnamed)
    # Since it's HEIF, is_heif_misnamed = True, so delete_original = True
    mock_convert_jpeg.assert_any_call(os.path.join("/data", "test.heic"), os.path.join("/data", "test.jpg"), delete_original=True)
    
    # Assert that the JPG was not converted again if it's already JPEG (and not HEIF misnamed)
    # This part is tricky because convert_image_to_jpeg might be called for problematic JPEGs (e.g. RGBA mode)
    # For this specific test, mock_jpg_image is simple RGB JPEG, so it shouldn't be called.
    # We need to count calls carefully or be more specific with assertions.
    
    # Check that convert_image_to_jpeg was called for the .heic file.
    # If other files could also lead to calls, assert_any_call is safer.
    mock_convert_jpeg.assert_any_call(
        os.path.join("/data", "test.heic"), 
        os.path.join("/data", "test.jpg"), 
        delete_original=True
    )
    
    # To be more precise, if only the HEIC file should be converted:
    # Check that convert_image_to_jpeg was called for test.heic and not for image.jpg
    calls = mock_convert_jpeg.call_args_list
    heic_converted = any(call[0][0] == os.path.join("/data", "test.heic") for call in calls)
    jpg_converted = any(call[0][0] == os.path.join("/data", "image.jpg") for call in calls)
    assert heic_converted, "HEIC file was not converted"
    assert not jpg_converted, "JPEG file should not have been converted in this specific test setup"


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', False) # Test without HEIF support
@patch('preprocess_images.logger.error')
def test_process_image_folder_heif_no_support(mock_logger_error, mock_heif_disabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test HEIF images are skipped and logged if HEIF support is disabled."""
    mock_os_walk.return_value = [
        ("/data", [], ["test.heic"]),
    ]
    mock_heif_image = MagicMock(spec=Image.Image)
    mock_heif_image.format = "HEIF"
    mock_image_open_global.return_value.__enter__.return_value = mock_heif_image

    process_image_folder("/data", target_extension="jpg") # Updated parameter name

    mock_convert_jpeg.assert_not_called()
    # The error is logged by _determine_image_action in the refactored code.
    # process_image_folder then increments error_files and continues.
    mock_logger_error.assert_any_call("'test.heic' is HEIF, but pillow-heif is not functional. Cannot process.")

@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True)
def test_process_image_folder_problematic_jpeg(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test that a JPEG with a problematic mode (e.g., RGBA) gets re-saved."""
    mock_os_walk.return_value = [
        ("/data", [], ["problem.jpg"]),
    ]
    
    mock_problematic_jpeg = MagicMock(spec=Image.Image)
    mock_problematic_jpeg.format = "JPEG"
    mock_problematic_jpeg.mode = "RGBA" # Problematic mode for a JPG
    
    mock_image_open_global.return_value.__enter__.return_value = mock_problematic_jpeg
    mock_convert_jpeg.return_value = True

    process_image_folder("/data", target_extension="jpg") # Updated parameter name

    # It should be called to convert the problematic JPEG
    mock_convert_jpeg.assert_called_once_with(os.path.join("/data", "problem.jpg"), os.path.join("/data", "problem.jpg"), delete_original=False)


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True)
def test_process_image_folder_skips_unidentified(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test that UnidentifiedImageError during Image.open in process_image_folder is handled."""
    mock_os_walk.return_value = [
        ("/data", [], ["corrupt.jpg"]),
    ]
    mock_image_open_global.return_value.__enter__.side_effect = UnidentifiedImageError("cannot open")

    process_image_folder("/data", target_extension="jpg") # Updated parameter name
    mock_convert_jpeg.assert_not_called()
    # Error is logged by process_image_folder's main exception handler
    mock_logger_error.assert_any_call("Pillow cannot identify image file (even after HEIF check): '/data/corrupt.jpg'. Skipping.")


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True)
def test_process_image_folder_standard_jpeg_no_action(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test that a standard, healthy JPEG does not trigger conversion."""
    mock_os_walk.return_value = [
        ("/data", [], ["good.jpg"]),
    ]
    
    mock_good_jpeg = MagicMock(spec=Image.Image)
    mock_good_jpeg.format = "JPEG"
    mock_good_jpeg.mode = "RGB" 
    
    mock_image_open_global.return_value.__enter__.return_value = mock_good_jpeg

    process_image_folder("/data", target_extension="jpg") # Updated parameter name

    # convert_image_to_jpeg should not be called for a good JPEG when target is jpg
    mock_convert_jpeg.assert_not_called()


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True)
def test_process_image_folder_png_is_not_converted_to_jpg(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test PNG is NOT converted to JPG if target is 'jpg' (current script logic)."""
    mock_os_walk.return_value = [
        ("/data", [], ["image.png"]),
    ]
    
    mock_png_image = MagicMock(spec=Image.Image)
    mock_png_image.format = "PNG"
    mock_png_image.mode = "RGBA" # PNGs often have RGBA
    
    mock_image_open_global.return_value.__enter__.return_value = mock_png_image
    mock_convert_jpeg.return_value = True

    # The script's current logic for non-HEIF, non-problematic JPEG is to *not* convert if extension matches target.
    # Let's adjust the script's logic or this test.
    # The script's Scenario 2 is commented out:
    # # elif actual_format.upper() != target_format_ext.upper():
    # #     print(f"INFO: '{filename}' is {actual_format} but target is {target_format_ext.upper()}. Marking for conversion.")
    # #     needs_action = True
    # If this logic were active, this test would be different.
    # As of now, a .png will only be converted if it's misnamed as .jpg and has problematic mode, or is HEIF.
    # This test should reflect that a .png is *not* automatically converted to .jpg by current script logic
    # unless the target_format_ext forces it (which it doesn't directly, only via HEIF or problematic JPEG logic)

    # To test conversion of PNG to JPG, we'd need to make `target_format_ext` 'jpg'
    # and the script logic would need to actively convert non-JPGs to JPG.
    # The current `process_image_folder` logic primarily targets HEIFs and problematic JPEGs.
    # It doesn't convert all non-target-extension files.
    
    # Let's assume for this test that the target is 'jpg' and we want to see if a PNG is identified.
    # The current script logic (via _determine_image_action) does not mark a standard PNG for action
    # if the target is JPG, unless it's misnamed HEIF or problematic JPEG.
    process_image_folder("/data", target_extension="jpg") # Updated parameter name
    mock_convert_jpeg.assert_not_called() # Based on current script logic, PNGs are not converted to JPG


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True)
def test_process_image_folder_misnamed_heif_as_jpg(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test that a HEIF file misnamed as .jpg is correctly identified and converted."""
    mock_os_walk.return_value = [
        ("/data", [], ["fake.jpg"]), # File has .jpg extension
    ]
    
    mock_misnamed_heif = MagicMock(spec=Image.Image)
    mock_misnamed_heif.format = "HEIF" # Actual format is HEIF
    mock_misnamed_heif.mode = "RGB"
    
    mock_image_open_global.return_value.__enter__.return_value = mock_misnamed_heif
    mock_convert_jpeg.return_value = True

    process_image_folder("/data", target_extension="jpg") # Updated parameter name

    # Should be converted to a new .jpg file (or overwrite if names are same after extension change)
    # and original (fake.jpg) should be deleted because is_heif_misnamed = True
    # In the refactored code, is_heif_misnamed is true, so delete_original becomes true.
    mock_convert_jpeg.assert_called_once_with(os.path.join("/data", "fake.jpg"), os.path.join("/data", "fake.jpg"), delete_original=True)


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.HEIF_SUPPORT_ENABLED', True)
def test_process_image_folder_target_ext_png(mock_heif_enabled, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test processing when target_format_ext is 'png'."""
    # This test assumes convert_image_to_jpeg is adapted or mocked to save as PNG
    # For simplicity, we'll just check if it's called with the correct new extension.
    mock_os_walk.return_value = [
        ("/data", [], ["image.heic"]),
    ]
    
    mock_heic_image = MagicMock(spec=Image.Image)
    mock_heic_image.format = "HEIF"
    mock_heic_image.mode = "RGB"
    
    mock_image_open_global.return_value.__enter__.return_value = mock_heic_image
    mock_convert_jpeg.return_value = True # Assume convert_image_to_jpeg handles PNG saving

    process_image_folder("/data", target_extension="png") # Updated parameter name

    mock_convert_jpeg.assert_called_once_with(os.path.join("/data", "image.heic"), os.path.join("/data", "image.png"), delete_original=True)


@patch('preprocess_images.os.walk')
@patch('preprocess_images.convert_image_to_jpeg')
@patch('preprocess_images.Image.open')
@patch('preprocess_images.logger.info') # To check log messages
def test_process_image_folder_skip_conversion_if_already_correct_and_not_heif(mock_logger_info, mock_image_open_global, mock_convert_jpeg, mock_os_walk):
    """Test the skip logic: original_filepath.lower() == new_filepath.lower() and not is_heif_misnamed ..."""
    mock_os_walk.return_value = [
        ("/data", [], ["good.jpg"]),
    ]
    
    mock_good_jpeg = MagicMock(spec=Image.Image)
    mock_good_jpeg.format = "JPEG"
    mock_good_jpeg.mode = "RGB" 
    
    # __enter__ because Image.open is used in a 'with' statement
    mock_image_open_global.return_value.__enter__.return_value = mock_good_jpeg

    # Patch HEIF_SUPPORT_ENABLED for this specific test if it's relevant to the condition
    with patch('preprocess_images.HEIF_SUPPORT_ENABLED', True): # Ensure HEIF_SUPPORT_ENABLED is True for this test context
      process_image_folder("/data", target_extension="jpg") # Updated parameter name

    mock_convert_jpeg.assert_not_called()
    # Check for the specific log message indicating skipping
    # The refactored code logs: logger.info(f"'{filename}' requires no action (already correct format/mode or not targeted for conversion).")
    mock_logger_info.assert_any_call("'good.jpg' requires no action (already correct format/mode or not targeted for conversion).")


# Final notes from previous version remain relevant regarding logger and HEIF_SUPPORT_ENABLED patching.
# Ensure these tests align with the actual logging statements and variable names in `preprocess_images.py`.
