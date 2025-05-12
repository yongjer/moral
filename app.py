import gradio as gr
from transformers import pipeline
from PIL import Image
import torch 

# Load custom model from local directory
# Ensure the path "./outputs" is correct and contains your trained model files.
# These files typically include 'config.json', 'pytorch_model.bin' (or 'tf_model.h5'), 
# and 'preprocessor_config.json'.

try:
    classifier = pipeline(
        task="image-classification",
        model="./ai_vs_real_classifier",  # Path to custom-trained model
        torch_dtype=torch.bfloat16,  # Use 'auto' to automatically select the appropriate dtype
        device="cuda",  # Automatically select the device (CPU/GPU) for inference
    )
    classifier.model = torch.compile(classifier.model, fullgraph=True)
    classifier.model.attn_implementation="flash_attention_2"
    
except Exception as e:
    print(f"Error loading the model: {e}")
    print("Please ensure that your local model path './outputs' is correct and contains all necessary model files.")
    # Fallback to a default model if local loading fails, for demonstration purposes.
    # Replace this with your preferred error handling or a default model.
    print("Falling back to a default Hugging Face model: 'google/vit-base-patch16-224'")
    classifier = pipeline(
        task="image-classification",
        model="google/vit-base-patch16-224", 
        torch_dtype="auto",  # Use 'auto' to automatically select the appropriate dtype
    )

def predict(image_path):
    """
    Takes an image path, opens the image, and returns predictions from the classifier.
    """
    try:
        # Load image from the uploaded file path
        img = Image.open(image_path)
        
        # Get predictions
        predictions = classifier(img) # Returns sorted list of class probabilities
        
        # Format predictions for Gradio output
        # The pipeline output is a list of dictionaries, e.g., [{'label': 'LABEL_0', 'score': 0.99}, ...]
        # We need to convert this to a dictionary format that Gradio's Label component expects: {label: score}
        formatted_predictions = {pred["label"]: pred["score"] for pred in predictions}

        return formatted_predictions
    except Exception as e:
        return {"Error": str(e)}

# Create the Gradio interface
# The `Image` input component will pass the uploaded image as a file path to the `predict` function.
# The `Label` output component is suitable for displaying classification results. [1, 2]
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath", label="Upload Image"), # Pass image as a filepath
    outputs=gr.Label(num_top_classes=5, label="Predictions"), # Display top 5 predictions
    title="Custom Image Classifier",
    description="Upload an image and see the model's predictions. This model is loaded from a local directory.",
    examples=["test_image_0.jpg", "test_image_1.png", "test_image_2.png", "test_image_3.png"] # Add a path to an example image if you have one.
                                 # Make sure 'test_image.jpg' exists in the same directory as this script,
                                 # or provide an absolute path.
)

# Launch the app
if __name__ == "__main__":
    print("Launching Gradio app...")
    iface.launch()