import os
from PIL import Image, UnidentifiedImageError
import yaml
import logging
import sys

# Basic Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration file 'config.yaml' loaded successfully.")
except FileNotFoundError:
    logger.error("Configuration file 'config.yaml' not found. Please ensure it exists in the current directory.")
    sys.exit(1)
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file 'config.yaml': {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"An unexpected error occurred while loading 'config.yaml': {e}")
    sys.exit(1)


# Attempt to import and register HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT_ENABLED = True
    logger.info("HEIF support enabled via pillow-heif.")
except ImportError:
    HEIF_SUPPORT_ENABLED = False
    logger.warning("pillow-heif not found. HEIF/HEIC files cannot be processed directly.")
    logger.warning("Please install it: pip install pillow-heif and its dependencies (e.g., sudo apt install libheif1 libde265-0 for Debian/Ubuntu)")


def convert_image_to_jpeg(source_path, target_path, delete_original=True):
    """
    Converts an image to JPEG format.
    If the source is HEIF, it will be handled if pillow-heif is installed.
    """
    try:
        with Image.open(source_path) as img:
            # Ensure the image is in RGB mode for JPEG compatibility
            if img.mode == 'RGBA' or img.mode == 'P':  # P is paletted
                logger.info(f"Converting '{source_path}' from mode {img.mode} to RGB for JPEG.")
                img = img.convert('RGB')
            elif img.mode != 'RGB':  # For other modes like L (grayscale), CMYK etc.
                logger.info(f"Image '{source_path}' is mode {img.mode}. Converting to RGB for JPEG if not already compatible.")
                img = img.convert('RGB')

            img.save(target_path, "JPEG", quality=90)  # Adjust quality as needed
            logger.info(f"CONVERTED: '{source_path}' to '{target_path}'")

            if delete_original and source_path != target_path:
                try:
                    os.remove(source_path)
                    logger.info(f"DELETED original: '{source_path}'")
                except OSError as e:
                    logger.error(f"Could not delete original file '{source_path}': {e}")
            return True
    except UnidentifiedImageError:
        logger.error(f"Pillow (even with HEIF support if enabled) cannot identify image file: '{source_path}'. It might be corrupted or an unsupported format.")
        return False
    except FileNotFoundError:
        logger.error(f"File not found during conversion: '{source_path}'. Skipping.")
        return False
    except PermissionError as e:
        logger.error(f"Permission error during conversion of '{source_path}': {e}")
        return False
    except Exception as e:
        logger.exception(f"An unexpected error occurred while converting '{source_path}': {e}")
        return False

def _determine_image_action(img, filename, target_extension, is_heif_supported):
    """
    Determines if an image file needs conversion or re-saving.

    Args:
        img (PIL.Image.Image): The opened image object.
        filename (str): The name of the image file.
        target_extension (str): The desired target extension (e.g., "jpg").
        is_heif_supported (bool): Flag indicating if HEIF support is enabled.

    Returns:
        tuple: (needs_action: bool, is_heif_misnamed: bool)
               - needs_action: True if the image should be processed.
               - is_heif_misnamed: True if the image is an HEIF file but doesn't have a HEIF extension.
    """
    actual_format = img.format
    logger.info(f"Opened '{filename}'. Actual format: {actual_format}, Mode: {img.mode}")
    needs_action = False
    is_heif_misnamed = False # True if it's an HEIF file (e.g. misnamed as .jpg)

    if actual_format == "HEIF":
        if is_heif_supported:
            logger.info(f"'{filename}' is an HEIF file.")
            needs_action = True
            # is_heif_misnamed is True if the original extension was not .heic/.heif but format is HEIF
            # For simplicity, if it's HEIF, we mark it to ensure it becomes target_extension.
            is_heif_misnamed = True 
        else:
            logger.error(f"'{filename}' is HEIF, but pillow-heif is not functional. Cannot process.")
            return False, False # No action possible by this script

    # Check for problematic JPEGs (e.g., RGBA mode) that need re-saving,
    # only if it wasn't already identified as a HEIF to be converted.
    if not is_heif_misnamed and actual_format and actual_format.upper() == target_extension.upper():
        if target_extension.lower() == "jpg" and img.mode not in ('RGB', 'L', 'CMYK'):
            logger.info(f"'{filename}' is JPEG but mode {img.mode} might be problematic. Re-saving.")
            needs_action = True
            # is_heif_misnamed remains False here, it's a problematic JPEG, not a misnamed HEIF.

    # The original "Scenario 2" (convert all non-target-extension images) is intentionally omitted
    # as it was commented out, implying it's not the current desired behavior.
    # If, for example, all PNGs should be converted to JPGs when target_extension="jpg",
    # that logic would be added here:
    # elif actual_format and actual_format.upper() != target_extension.upper():
    #     logger.info(f"'{filename}' is {actual_format} but target is {target_extension.upper()}. Marking for conversion.")
    #     needs_action = True

    return needs_action, is_heif_misnamed


def process_image_folder(base_folder, target_extension="jpg"):
    """
    Processes all images in the base_folder and its subdirectories.
    Converts HEIF/HEIC files to the target_extension (default JPEG).
    Also re-saves problematic JPEGs (e.g. non-RGB modes if target is JPG).
    """
    if not HEIF_SUPPORT_ENABLED:
        logger.critical("HEIF support is not available. HEIF files will likely cause errors or be skipped.")
        # Depending on strictness, one might exit here:
        # sys.exit(1)

    processed_files = 0
    converted_files = 0
    error_files = 0

    # Common image extensions to look for.
    # HEIC/HEIF are included to catch them even if correctly named.
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif')

    for dirpath, dirnames, filenames in os.walk(base_folder):
        logger.info(f"Processing directory: {dirpath}")
        for filename in filenames:
            original_filepath = os.path.join(dirpath, filename)
            name_part, ext_part = os.path.splitext(filename)
            ext_part_lower = ext_part.lower()

            if ext_part_lower not in image_extensions:
                logger.debug(f"Skipping non-image or unrecognized extension: {filename}")
                continue

            logger.info(f"--- Examining: {filename} ---")
            processed_files += 1
            
            try:
                with Image.open(original_filepath) as img:
                    # Pass HEIF_SUPPORT_ENABLED to the helper function
                    needs_action, is_heif_misnamed = _determine_image_action(img, filename, target_extension, HEIF_SUPPORT_ENABLED)

                    # If _determine_image_action decided no action because HEIF support is off for an HEIF file
                    if img.format == "HEIF" and not HEIF_SUPPORT_ENABLED and not needs_action:
                        error_files += 1 # Error already logged by _determine_image_action
                        continue

                if needs_action: # is_heif_misnamed is implicitly True if it's a HEIF needing action
                    new_filename = f"{name_part}.{target_extension.lower()}"
                    new_filepath = os.path.join(dirpath, new_filename)

                    # The complex skip condition from before:
                    # `if original_filepath.lower() == new_filepath.lower() and not is_heif_misnamed and not (needs_action and actual_format and actual_format.upper() == target_extension.upper()):`
                    # This is simplified: if needs_action is True, we attempt conversion.
                    # convert_image_to_jpeg handles the case where source_path == target_path for deletion.
                    # is_heif_misnamed helps determine if deletion should occur even if paths are same (e.g. HEIC.jpg -> HEIC.jpg after conversion)
                    
                    if convert_image_to_jpeg(original_filepath, new_filepath, delete_original=(original_filepath != new_filepath or is_heif_misnamed)):
                        converted_files += 1
                    else: # Conversion failed
                        error_files += 1
                else: # No action needed
                    logger.info(f"'{filename}' requires no action (already correct format/mode or not targeted for conversion).")

            except UnidentifiedImageError:
                logger.error(f"Pillow cannot identify image file (even after HEIF check): '{original_filepath}'. Skipping.")
                error_files += 1
            except FileNotFoundError:
                logger.error(f"File not found: '{original_filepath}'. Skipping (maybe deleted during processing?).")
                error_files += 1
            except PermissionError as e:
                logger.error(f"Permission error processing file '{original_filepath}': {e}")
                error_files += 1
            except Exception as e:
                logger.exception(f"An unexpected error occurred with '{original_filepath}': {e}")
                error_files += 1

    logger.info("\n--- Processing Summary ---")
    logger.info(f"Total files examined (with image extensions): {processed_files}")
    logger.info(f"Files successfully converted/re-saved: {converted_files}")
    logger.info(f"Files with errors: {error_files}")

# --- Configuration from YAML ---
# Ensure that 'data_dir' from config is used, as 'preprocess_data_directory' was removed from config.yaml
try:
    DATA_DIRECTORY = config["paths"]["data_dir"]
except KeyError:
    logger.error("Key 'data_dir' not found in config['paths']. Please ensure config.yaml is correctly structured.")
    # Provide a default or exit if this path is critical and has no sensible default
    DATA_DIRECTORY = "./default_data_dir_preprocess" # Or sys.exit(1)
    logger.warning(f"Falling back to default data directory for preprocessing: {DATA_DIRECTORY}")

try:
    TARGET_EXTENSION = config["preprocessing"]["target_extension"]
except KeyError:
    logger.error("Key 'target_extension' not found in config['preprocessing']. Using default 'jpg'.")
    TARGET_EXTENSION = "jpg"


if __name__ == "__main__":
    logger.info("Starting image preprocessing script.")
    if not os.path.isdir(DATA_DIRECTORY):
        logger.error(f"The specified data directory '{DATA_DIRECTORY}' does not exist. Please check config.yaml.")
        sys.exit(1)
    else:
        logger.info(f"Processing images in folder: {DATA_DIRECTORY}, target extension: {TARGET_EXTENSION}")
        process_image_folder(DATA_DIRECTORY, TARGET_EXTENSION)
    logger.info("Image preprocessing script finished.")