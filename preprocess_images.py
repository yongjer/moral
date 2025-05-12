import os
from PIL import Image, UnidentifiedImageError

# Attempt to import and register HEIF support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    HEIF_SUPPORT_ENABLED = True
    print("INFO: HEIF support enabled via pillow-heif.")
except ImportError:
    HEIF_SUPPORT_ENABLED = False
    print("WARNING: pillow-heif not found. HEIF/HEIC files cannot be processed directly.")
    print("         Please install it: pip install pillow-heif")
    print("         And its dependencies: sudo apt install libheif1 libde265-0 (for Debian/Ubuntu)")


def convert_image_to_jpeg(source_path, target_path, delete_original=True):
    """
    Converts an image to JPEG format.
    If the source is HEIF, it will be handled if pillow-heif is installed.
    """
    try:
        with Image.open(source_path) as img:
            # Ensure the image is in RGB mode for JPEG compatibility
            if img.mode == 'RGBA' or img.mode == 'P': # P is paletted
                print(f"INFO: Converting '{source_path}' from mode {img.mode} to RGB for JPEG.")
                img = img.convert('RGB')
            elif img.mode != 'RGB': # For other modes like L (grayscale), CMYK etc.
                print(f"INFO: Image '{source_path}' is mode {img.mode}. Converting to RGB for JPEG if not already compatible.")
                # Forcing RGB, though for some like 'L' it might not be strictly necessary
                # but guarantees compatibility.
                img = img.convert('RGB')

            img.save(target_path, "JPEG", quality=90) # Adjust quality as needed
            print(f"CONVERTED: '{source_path}' to '{target_path}'")

            if delete_original and source_path != target_path:
                try:
                    os.remove(source_path)
                    print(f"DELETED original: '{source_path}'")
                except OSError as e:
                    print(f"ERROR: Could not delete original file '{source_path}': {e}")
            return True
    except UnidentifiedImageError:
        print(f"ERROR: Pillow (even with HEIF support if enabled) cannot identify image file: '{source_path}'. It might be corrupted or an unsupported format.")
        return False
    except FileNotFoundError:
        print(f"ERROR: File not found during conversion: '{source_path}'. Skipping.")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while converting '{source_path}': {e}")
        return False

def process_image_folder(base_folder, target_format_ext="jpg"):
    """
    Processes all images in the base_folder and its subdirectories.
    Converts HEIF/HEIC files (even with .JPG extension) to the target_format_ext (default JPEG).
    Also re-saves other images if their extension doesn't match their actual format or target format.
    """
    if not HEIF_SUPPORT_ENABLED:
        print("CRITICAL: HEIF support is not available. HEIF files will likely cause errors or be skipped.")
        # You might want to exit or handle this more strictly depending on your needs
        # return

    processed_files = 0
    converted_files = 0
    error_files = 0

    # Common image extensions to look for.
    # HEIC/HEIF are included to catch them even if correctly named.
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.heic', '.heif')

    for dirpath, dirnames, filenames in os.walk(base_folder):
        print(f"\nProcessing directory: {dirpath}")
        for filename in filenames:
            original_filepath = os.path.join(dirpath, filename)
            name_part, ext_part = os.path.splitext(filename)
            ext_part_lower = ext_part.lower()

            if ext_part_lower not in image_extensions:
                # print(f"Skipping non-image or unrecognized extension: {filename}")
                continue

            print(f"--- Examining: {filename} ---")
            processed_files += 1
            needs_action = False
            is_heif_misnamed = False

            try:
                with Image.open(original_filepath) as img:
                    actual_format = img.format
                    print(f"INFO: Opened '{filename}'. Actual format: {actual_format}, Mode: {img.mode}")

                    # Scenario 1: It's an HEIF file (regardless of its current extension)
                    if actual_format == "HEIF":
                        if HEIF_SUPPORT_ENABLED:
                            print(f"INFO: '{filename}' is an HEIF file.")
                            needs_action = True
                            is_heif_misnamed = True # Mark for conversion to target_format_ext
                        else:
                            print(f"ERROR: '{filename}' is HEIF, but pillow-heif is not functional. Cannot process.")
                            error_files +=1
                            continue # Skip to next file

                    # Scenario 2: Extension is wrong, or we want to standardize all to target_format_ext
                    # (e.g., a .png file that we want to convert to .jpg)
                    # For this script, we primarily focus on fixing HEIFs and standardizing their extension.
                    # If you want to convert ALL images to JPEG, uncomment and adjust:
                    # elif actual_format.upper() != target_format_ext.upper():
                    #     print(f"INFO: '{filename}' is {actual_format} but target is {target_format_ext.upper()}. Marking for conversion.")
                    #     needs_action = True

                    # Scenario 3: The extension IS the target format, but it might have been a misnamed HEIF
                    # This case is covered if actual_format was HEIF.
                    # If it's genuinely a JPG, and target is JPG, do nothing unless it's a problematic mode.
                    if actual_format and actual_format.upper() == target_format_ext.upper() and not is_heif_misnamed:
                         # Extra check for JPGs with problematic modes like RGBA or P if we strictly need RGB
                        if target_format_ext.lower() == "jpg" and img.mode not in ('RGB', 'L', 'CMYK'): # Common safe JPEG modes
                            print(f"INFO: '{filename}' is JPEG but mode {img.mode} might be problematic. Re-saving.")
                            needs_action = True


                if needs_action or is_heif_misnamed:
                    new_filename = f"{name_part}.{target_format_ext.lower()}"
                    new_filepath = os.path.join(dirpath, new_filename)

                    # Avoid converting if it's already the correct file (e.g. image.jpg -> image.jpg)
                    # unless it was a misnamed HEIF or problematic mode
                    if original_filepath.lower() == new_filepath.lower() and not is_heif_misnamed and not (needs_action and actual_format.upper() == target_format_ext.upper()):
                        print(f"INFO: '{filename}' seems to be in correct format and name already. Skipping conversion step.")
                    elif convert_image_to_jpeg(original_filepath, new_filepath, delete_original=(original_filepath != new_filepath or is_heif_misnamed)):
                        converted_files += 1
                    else:
                        error_files += 1
                else:
                    print(f"INFO: '{filename}' requires no action.")

            except UnidentifiedImageError:
                # This might catch files that pillow-heif also couldn't open, or other truly corrupt files.
                print(f"ERROR: Pillow cannot identify image file (even after HEIF check): '{original_filepath}'. Skipping.")
                error_files += 1
            except FileNotFoundError:
                print(f"ERROR: File not found: '{original_filepath}'. Skipping (maybe deleted during processing?).")
                error_files += 1
            except Exception as e:
                print(f"ERROR: An unexpected error occurred with '{original_filepath}': {e}")
                error_files += 1

    print("\n--- Processing Summary ---")
    print(f"Total files examined (with image extensions): {processed_files}")
    print(f"Files successfully converted/re-saved: {converted_files}")
    print(f"Files with errors: {error_files}")

# --- Configuration ---
DATA_DIRECTORY = './人工智慧第四組資料夾' # Change this to your top-level data folder
TARGET_EXTENSION = "jpg"  # Convert problematic files to this extension (e.g., "jpg", "png")

if __name__ == "__main__":
    if not os.path.isdir(DATA_DIRECTORY):
        print(f"ERROR: The specified data directory '{DATA_DIRECTORY}' does not exist.")
    else:
        process_image_folder(DATA_DIRECTORY, TARGET_EXTENSION)