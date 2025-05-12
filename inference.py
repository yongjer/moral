from transformers import pipeline

# Load custom model from local directory
classifier = pipeline(
    task="image-classification",
    model="./outputs",  # Path to custom-trained model
)

from PIL import Image

# Load image from file/URL/dataset
image = Image.open("test_image.jpg")

# Get predictions
predictions = classifier(image)  # Returns sorted list of class probabilities[1]

print(predictions)  # Print predictions