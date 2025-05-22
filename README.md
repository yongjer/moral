# AI vs. Real Image Classifier
<div align="center">

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/yongjer/moral)

</div>

This project contains a machine learning model and a Gradio web application to classify images as either AI-generated or real. It utilizes Python, the Hugging Face Transformers library, PyTorch for model training and inference, and Gradio for the interactive user interface.
The system is designed to help distinguish between photographic images and images created by artificial intelligence.

## Key Technologies:
*   Python
*   PyTorch
*   Hugging Face Transformers
*   Gradio
*   Scikit-learn
*   Pillow (PIL)
*   Matplotlib & Seaborn (for metrics visualization)

## Project Structure

Here's a brief overview of the key files and directories:

*   **`main.py`**: The main script for training the image classification model. It handles data loading, preprocessing, model training using PyTorch/Transformers, evaluation, and saving the trained model.
*   **`app.py`**: A Gradio web application that loads the trained model and provides an interactive interface for users to upload images and get predictions (AI-generated or real).
*   **`preprocess_images.py`**: A utility script to convert images in the dataset (especially HEIC/HEIF formats) to JPEG format. This is often a necessary step before training.
*   **`inference.py`**: A simple Python script for performing command-line inference with the trained model on a single image.
*   **`ai_vs_real_classifier/`**: This directory is created during training (by `main.py`) and stores the trained model files. `app.py` loads the model from here.
*   **`人工智慧第四組資料夾/`**: This is the expected root directory for the training dataset. It should contain two subdirectories:
    *   `ai/`: For AI-generated images.
    *   `real/`: For real photographic images.
*   **`confusion_matrix.png`**: An image file generated after training by `main.py`, showing the confusion matrix of the model's performance on the test set.
*   **`pyproject.toml`**: Defines project metadata and dependencies.
*   **`.python-version`**: Specifies the Python version for the project.
*   **`README.md`**: This file, providing information about the project.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd <repository-directory>
    ```

2.  **Python Version:**
    This project uses Python 3.12. It's recommended to use a virtual environment (e.g., venv, conda).
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    The project dependencies are listed in `pyproject.toml` and also provided in `requirements.txt` for convenience with pip.
    ```bash
    pip install -r requirements.txt
    ```
    This file contains all necessary Python packages for the project.
    For GPU support with PyTorch (recommended for training and faster inference), ensure your PyTorch installation is compatible with your CUDA setup. You might need to install a specific PyTorch version by following instructions on the [official PyTorch website](https://pytorch.org/get-started/locally/). The `requirements.txt` typically includes a CPU-compatible or a general PyTorch version.

4.  **Optional: HEIC/HEIF Support:**
    The `preprocess_images.py` script can convert HEIC/HEIF images. To enable this, you need to install `pillow-heif`:
    ```bash
    pip install pillow-heif
    ```
    You might also need to install system libraries for `libheif`:
    *   On Debian/Ubuntu: `sudo apt-get install libheif1 libde265-0`
    *   On macOS (using Homebrew): `brew install libheif`

## Project Configuration (`config.yaml`)

Key settings for the project are managed in the `config.yaml` file. This includes:

*   **`paths`**:
    *   `data_dir`: Path to the root dataset directory (e.g., `"./人工智慧第四組資料夾"`). **This is the primary path you might need to change.**
    *   `model_name`: The Hugging Face model identifier for the base Vision Transformer (e.g., `"google/vit-large-patch16-224-in21k"`).
    *   `output_dir`: Directory to save training results (like checkpoints).
    *   `logging_dir`: Directory for training logs.
    *   `final_model_path`: Path where the final trained model is saved (e.g., `"./ai_vs_real_classifier"`).
    *   `confusion_matrix_path`: Path to save the generated confusion matrix image.
*   **`training`**:
    *   Parameters like `num_train_epochs`, `per_device_train_batch_size`, `warmup_steps`, `weight_decay`, `bf16`, `torch_compile`, etc.
*   **`dataset`**:
    *   `val_ratio`, `test_ratio`: Proportions for splitting the dataset.
    *   `preprocess_batch_size`: Batch size for the dataset mapping/preprocessing step.
*   **`preprocessing`**:
    *   `target_extension`: The target image format (e.g., "jpg") for `preprocess_images.py`.

Before running preprocessing or training, review `config.yaml` and adjust paths and parameters as needed, especially `paths.data_dir`.

## Data Preparation

The model is trained to classify images as either AI-generated or real.

1.  **Dataset Directory Structure:**
    *   Your dataset root folder should be specified in `config.yaml` under `paths.data_dir`.
    *   Inside this folder, create two subdirectories:
        *   `ai/`: Place all your AI-generated images here.
        *   `real/`: Place all your real photographic images here.

    Example structure if `paths.data_dir` is set to `"./人工智慧第四組資料夾"`:
    ```
    <project-root>/
    ├── 人工智慧第四組資料夾/  <-- This path is set in config.yaml
    │   ├── ai/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── real/
    │       ├── image3.jpg
    │       └── ...
    ├── config.yaml
    ├── main.py
    └── ...
    ```

2.  **Image Formats & Preprocessing:**
    *   It's recommended to have images in common web formats like JPEG or PNG.
    *   If you have images in HEIC/HEIF format, use `preprocess_images.py`.
        *   Ensure `pillow-heif` is installed (see "Setup and Installation").
        *   The script uses `paths.data_dir` and `preprocessing.target_extension` from `config.yaml`.
        *   Run from the project root:
            ```bash
            python preprocess_images.py
            ```
    The script will convert images as configured, potentially overwriting originals if the target name is the same (e.g. a problematic JPG being re-saved as JPG) or deleting originals if `is_heif_misnamed` is true during HEIF to JPEG conversion. Review the script and `config.yaml` if you have concerns.

## Training the Model

Once your dataset is prepared, dependencies are installed, and `config.yaml` is reviewed:

1.  **Run the Training Script:**
    Execute `main.py` from the project's root directory:
    ```bash
    python main.py
    ```

2.  **Training Process:**
    *   The script loads images from the directory specified in `config.yaml` (`paths.data_dir`).
    *   It uses the Vision Transformer model defined in `config.yaml` (`paths.model_name`).
    *   A Focal Loss function is implemented to help address potential class imbalance.
    *   Training arguments (epochs, batch size, etc.) are now sourced from `config.yaml` (`training` section).
    *   The script outputs training progress and evaluation metrics.

3.  **Output:**
    *   **Trained Model:** Saved to the path specified in `config.yaml` (`paths.final_model_path`, e.g., `./ai_vs_real_classifier/`).
    *   **Confusion Matrix:** Saved to the path specified in `config.yaml` (`paths.confusion_matrix_path`, e.g., `confusion_matrix.png`).

**Note on Resources:** Training deep learning models can be computationally intensive. The script uses settings from `config.yaml` for `bf16` (mixed-precision) and `torch_compile` which can optimize training on compatible hardware.

## Usage (Inference)

Once the model is trained and saved, you can use it to classify new images.

### 1. Web Application (Gradio)

The primary way to interact with the classifier is through the Gradio web application:

*   **Run the App:**
    Execute the `app.py` script from the project's root directory:
    ```bash
    python app.py
    ```
*   **Using the Interface:**
    *   The script will typically print a local URL to your console (e.g., `Running on local URL:  http://127.0.0.1:7860`). Open this URL in your web browser.
    *   You'll see an interface titled "Custom Image Classifier".
    *   Upload an image using the provided upload box.
    *   The model will process the image, and the predicted label ("AI" or "Real") along with confidence scores for the top classes will be displayed.
    *   The app includes some example image paths (`test_image_0.jpg`, `test_image_1.png`, etc.). Ensure these example images exist if you want to use the example buttons in the Gradio interface, or update the paths in `app.py`.

### 2. Command-Line Inference (Basic)

For a simpler, non-interactive way to get a prediction for a single image, you can use `inference.py`:

*   **Prepare an image:** Place an image you want to test (e.g., `my_test_image.jpg`) in the project's root directory, or provide the correct path to it.
*   **Prepare an image:** Place an image you want to test (e.g., `my_test_image.jpg`) in the project's root directory, or provide the correct path to it.
*   **Configuration:** `inference.py` loads the model path from `config.yaml` (`paths.final_model_path`). Ensure this path is correct.
    *   The script will try to load an image named `test_image.jpg` by default (this can be changed in the script or by making `inference.py` accept command-line arguments).
*   **Run the script:**
    ```bash
    python inference.py
    ```
*   The script will print the raw prediction output to the console.

**Note:** The Gradio application (`app.py`) loads the model path from `config.yaml` (`paths.final_model_path`) and is generally recommended for ease of use.

## Model Information

*   **Base Model:** The classification model is based on the Vision Transformer (ViT) architecture. Specifically, the `main.py` script is configured to use `"google/vit-large-patch16-224-in21k"` as the starting point. This model is pre-trained on the ImageNet-21k dataset.
*   **Fine-Tuning:** The pre-trained ViT model is then fine-tuned on the custom "AI vs. Real" image dataset provided by the user.
*   **Output Layer:** The number of labels for the classification head is set to 2 (e.g., "AI" and "Real"). The `id2label` mapping is `{0: "AI", 1: "Real"}` and `label2id` is `{"AI": 0, "Real": 1}`.
*   **Loss Function:** To handle potential class imbalance in the dataset (where one class might have significantly more images than the other), a **Focal Loss** function is implemented in `main.py` and used during training. This helps the model pay more attention to hard-to-classify examples and down-weights the loss assigned to well-classified examples. The `alpha` parameter of the Focal Loss is dynamically calculated based on the inverse class frequencies in the training data.
*   **Performance:** The `main.py` script computes and logs various metrics during validation and testing, including accuracy, F1-score, precision, and recall. A confusion matrix is also generated and saved as `confusion_matrix.png`.

## Model Performance Visualization

After a successful training run using `python main.py`, a confusion matrix image named `confusion_matrix.png` is saved to the project's root directory.

This image visually represents the model's performance by showing the counts of true positive, true negative, false positive, and false negative predictions on the test set.

**Viewing the Confusion Matrix:**

*   You can open the `confusion_matrix.png` file directly from your file explorer.
*   If this README is being viewed on a platform that renders Markdown and can access local images (e.g., when the repository is hosted on GitHub), the image can be embedded directly.

**(Note: The image will only be visible below if `confusion_matrix.png` exists in the root of the repository and the Markdown viewer supports relative image paths.)**

![Confusion Matrix](./confusion_matrix.png)

## To-Do / Future Work

Here are some potential areas for future improvement and development:

*   **Completed / Addressed:**
    *   ~~`requirements.txt`~~: A `requirements.txt` file has been created and is the primary way to install dependencies.
    *   ~~Configuration File~~: Hardcoded paths and key training parameters have been moved to `config.yaml`.
    *   ~~Error Handling & Logging~~: Enhanced significantly across all scripts with structured logging and more robust error handling.
    *   ~~Testing~~: Unit tests have been added for core components of each script, covering config loading, preprocessing, main training logic (FocalLoss, metrics, data preprocessing), and inference/app model loading.

*   **Future Considerations:**
    *   **Expanded Model Evaluation:** Implement more detailed evaluation, such as per-class metrics if more categories are added, or ROC curves and AUC scores.
    *   **Model Checkpointing Options:** The current `TrainingArguments` save strategy is "epoch"; more flexible options (e.g., save best N, save every K steps) could be explored directly via Hugging Face Trainer capabilities or custom callbacks.
    *   **Data Augmentation:** Explore and implement more sophisticated data augmentation techniques during training.
    *   **Packaging:** Package the application for easier distribution (e.g., using Docker).
    *   **UI Enhancements:** Add more features to the Gradio UI, such as displaying example predictions or allowing adjustment of inference parameters.
    *   **CLI Arguments:** Add command-line arguments to scripts like `main.py` or `inference.py` to override specific `config.yaml` settings without direct file editing (e.g., for `inference.py` image path).
    *   **Shared Utilities:** Refactor common components like config loading into a shared `utils.py` module to reduce duplication.
