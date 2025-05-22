from transformers import pipeline
from PIL import Image, UnidentifiedImageError
import logging
import sys
import os # For basename
import yaml

# Basic Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
_DEFAULT_MODEL_PATH = "./ai_vs_real_classifier" # Fallback if config key is missing
_DEFAULT_EXAMPLE_IMAGE_PATH = "test_image.jpg"

# Load configuration
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    logger.info("Configuration file 'config.yaml' loaded successfully for inference.py.")
except FileNotFoundError:
    logger.error("Configuration file 'config.yaml' not found in inference.py. Please ensure it exists.")
    sys.exit(1) 
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file 'config.yaml' in inference.py: {e}")
    sys.exit(1) 
except Exception as e:
    logger.error(f"An unexpected error occurred while loading 'config.yaml' in inference.py: {e}")
    sys.exit(1) 


def load_model(model_path: str):
    """
    Loads the image classification model using Hugging Face pipeline.

    Args:
        model_path (str): The path or Hugging Face model identifier for the model.

    Returns:
        transformers.pipeline or None: The loaded pipeline object, or None if loading fails.
    """
    try:
        logger.info(f"Loading image classification model from '{model_path}'...")
        # Using a more specific model path as seen in app.py and main.py
        # If "./outputs" is indeed the correct path for this script, it can be adjusted.
        # The prompt mentioned "./ai_vs_real_classifier" for app.py, let's assume consistency or use a config.
        # For now, sticking to the original "./outputs" for this file as per its content.
        classifier = pipeline(
            task="image-classification",
            model=model_path,
        )
        logger.info(f"Model from '{model_path}' loaded successfully.")
        return classifier
    except Exception as e:
        logger.exception(f"Error loading the model from '{model_path}': {e}")
        logger.warning("Consider using a known public model like 'google/vit-base-patch16-224' as a fallback if available.")
        return None

def predict_image(classifier, image_path: str = _DEFAULT_EXAMPLE_IMAGE_PATH):
    """
    Loads an image from the given path and gets predictions using the classifier.

    Args:
        classifier: The loaded Hugging Face pipeline for image classification.
        image_path (str, optional): Path to the image file. 
                                    Defaults to _DEFAULT_EXAMPLE_IMAGE_PATH.

    Returns:
        list or None: A list of prediction dictionaries from the classifier, 
                      or None if any error occurs.
    """
    if classifier is None:
        logger.error("Classifier model is not loaded. Cannot perform prediction.")
        return None

    logger.info(f"Loading image from '{image_path}'...")
    try:
        image = Image.open(image_path)
        logger.info(f"Image '{image_path}' loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Image file not found at path: {image_path}")
        return None
    except UnidentifiedImageError:
        logger.error(f"Cannot identify image file, it might be corrupted or an unsupported format: {image_path}")
        return None
    except Exception as e:
        logger.exception(f"An unexpected error occurred while loading image '{image_path}': {e}")
        return None

    logger.info("Getting predictions from the model...")
    try:
        predictions = classifier(image)
        logger.info(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        logger.exception(f"An error occurred during model prediction for image '{image_path}': {e}")
        return None

def main():
    """
    Main function to run the inference process.
    Loads the model, an example image (creating one if it doesn't exist),
    and performs prediction.
    """
    logger.info("Starting inference script...")
    
    # Get model path from config
    try:
        model_path_from_config = config['paths']['final_model_path']
        logger.info(f"Using model path from config: '{model_path_from_config}'")
    except KeyError:
        logger.error(f"Key 'final_model_path' not found in config['paths']. Using default '{_DEFAULT_MODEL_PATH}'.")
        model_path_from_config = _DEFAULT_MODEL_PATH

    classifier = load_model(model_path=model_path_from_config)

    if classifier:
        # Use the default example image path constant
        if not os.path.exists(_DEFAULT_EXAMPLE_IMAGE_PATH):
            logger.warning(f"Example image '{_DEFAULT_EXAMPLE_IMAGE_PATH}' not found. Creating a dummy image.")
            try:
                dummy_image = Image.new('RGB', (100, 100), color = 'red')
                dummy_image.save(_DEFAULT_EXAMPLE_IMAGE_PATH)
                logger.info(f"Created a dummy image at '{_DEFAULT_EXAMPLE_IMAGE_PATH}' for demonstration.")
            except Exception as e:
                logger.exception(f"Could not create dummy image '{_DEFAULT_EXAMPLE_IMAGE_PATH}': {e}")
                sys.exit(1) # Exit if dummy image creation fails, as prediction would fail.

        predictions = predict_image(classifier, image_path=_DEFAULT_EXAMPLE_IMAGE_PATH)
        if predictions:
            logger.info(f"Successfully processed '{_DEFAULT_EXAMPLE_IMAGE_PATH}'.")
        else:
            logger.error(f"Failed to process '{_DEFAULT_EXAMPLE_IMAGE_PATH}'.")
    else:
        logger.error("Model could not be loaded. Exiting inference script.")
        sys.exit(1)
        
    logger.info("Inference script finished.")

if __name__ == "__main__":
    main()