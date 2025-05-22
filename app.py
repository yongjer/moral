import gradio as gr
from transformers import pipeline
from PIL import Image, UnidentifiedImageError
import torch
import logging
import sys
import os # For os.path.basename
import yaml

# Basic Logging Configuration
# Configure logging early, before other imports if they might log too.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

# Load configuration
try:
    with open("config.yaml", "r") as f:
        app_config = yaml.safe_load(f) # Use a different name to avoid clash if 'config' is used elsewhere
    logger.info("Configuration file 'config.yaml' loaded successfully for app.py.")
except FileNotFoundError:
    logger.error("Configuration file 'config.yaml' not found for app.py. Critical features may fail.")
    app_config = {} # Allow app to run with defaults if config is missing for some reason
except yaml.YAMLError as e:
    logger.error(f"Error parsing configuration file 'config.yaml' in app.py: {e}")
    app_config = {}
except Exception as e:
    logger.error(f"An unexpected error occurred while loading 'config.yaml' in app.py: {e}")
    app_config = {}


# Global variable for the classifier
classifier = None

def load_model():
    """
    Loads the image classification model.
    It first tries to load the custom model specified in the configuration.
    If that fails, it attempts to load a default public model as a fallback.
    Sets the global `classifier` variable.
    """
    global classifier
    
    custom_model_path = app_config.get("paths", {}).get("final_model_path", "./ai_vs_real_classifier")
    logger.info(f"Attempting to load custom model from '{custom_model_path}'...")

    try:
        classifier = pipeline(
            task="image-classification",
            model=custom_model_path,
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        # These model optimizations might be better applied after successful pipeline creation
        # and potentially within another try-except if they are prone to errors.
        if hasattr(classifier, "model") and classifier.model is not None:
            logger.info("Compiling model and setting flash attention...")
            classifier.model = torch.compile(classifier.model, fullgraph=True)
            classifier.model.attn_implementation="flash_attention_2"
            logger.info("Custom model loaded and configured successfully.")
        else:
            logger.error("Pipeline created, but model attribute is missing or None.")
            # Fallback or exit, depending on desired behavior
            raise RuntimeError("Model attribute missing after pipeline creation.")

    except Exception as e:
        logger.exception(f"Error loading the custom model: {e}")
        logger.warning("Please ensure that your local model path './ai_vs_real_classifier' is correct and contains all necessary model files.")
        logger.info("Falling back to a default Hugging Face model: 'google/vit-base-patch16-224'")
        try:
            classifier = pipeline(
                task="image-classification",
                model="google/vit-base-patch16-224",
                torch_dtype="auto",
            )
            logger.info("Default Hugging Face model 'google/vit-base-patch16-224' loaded successfully.")
        except Exception as fallback_e:
            logger.exception(f"Error loading the fallback Hugging Face model: {fallback_e}")
            logger.critical("Both custom and fallback model loading failed. Application cannot start.")
            # If Gradio is already running, this might not stop it,
            # but it signals a critical failure.
            # Consider how to communicate this to the user in the UI if possible,
            # or exit if this script is run before Gradio launch.
            classifier = None # Ensure classifier is None if loading fails
            # sys.exit(1) # Uncomment if you want to exit the script entirely

# Load the model when the script starts
load_model()

def predict(image_path):
    """
    Takes an image path, opens the image, and returns predictions from the classifier.
    """
    if classifier is None:
        logger.error("Classifier model is not loaded. Cannot perform prediction.")
        return {"Error": "Model not loaded. Please check server logs."}

    if image_path is None:
        logger.warning("No image path provided to predict function.")
        return {"Error": "No image uploaded."} # Return an error message that Gradio can display
        
    logger.info(f"Received image for prediction: {os.path.basename(image_path)}") # Log only basename

    try:
        # Load image from the uploaded file path
        img = Image.open(image_path)
        logger.info(f"Image '{image_path}' opened successfully.")
        
        # Get predictions
        logger.info("Performing prediction...")
        predictions = classifier(img) # Returns sorted list of class probabilities
        logger.info(f"Prediction successful. Results: {predictions}")
        
        # Format predictions for Gradio output
        formatted_predictions = {pred["label"]: pred["score"] for pred in predictions}
        return formatted_predictions
        
    except FileNotFoundError:
        logger.error(f"Image file not found at path: {image_path}")
        return {"Error": f"Image file not found: {os.path.basename(image_path)}"}
    except UnidentifiedImageError:
        logger.error(f"Cannot identify image file, it might be corrupted or an unsupported format: {image_path}")
        return {"Error": f"Cannot open or read image file: {os.path.basename(image_path)}. It may be corrupted or an unsupported format."}
    except Exception as e:
        logger.exception(f"An unexpected error occurred during prediction for image '{image_path}': {e}")
        return {"Error": "An unexpected error occurred during prediction."}

# Create the Gradio interface
# Check if classifier is loaded before creating interface that depends on it.
if classifier is not None:
    iface = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="filepath", label="Upload Image"),
        outputs=gr.Label(num_top_classes=5, label="Predictions"),
        title="Custom Image Classifier",
        description="Upload an image and see the model's predictions. This model is loaded from a local directory if available, otherwise a default model is used.",
        examples=["test_image_0.jpg", "test_image_1.png", "test_image_2.png", "test_image_3.png"]
    )
else:
    # Fallback interface if model loading failed completely
    with gr.Blocks() as iface:
        gr.Markdown("## Model Loading Failed")
        gr.Markdown("The image classification model could not be loaded. Please check the server logs for more details. The application cannot process images at this time.")
    logger.critical("Gradio interface created with an error message because classifier is None.")


# Launch the app
if __name__ == "__main__":
    logger.info("Launching Gradio app...")
    iface.launch()
    logger.info("Gradio app launched.")