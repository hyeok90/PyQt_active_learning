# YOLO11-seg Active Learning-based Annotation Tool

This is a GUI-based image annotation tool developed with Python and PyQt5, designed for efficient object segmentation tasks. It leverages YOLOv11-seg models to implement an active learning workflow, significantly speeding up the labeling process.


## ✨ Features

- **🧠 Active Learning:** Automatically pre-annotates images without labels using a loaded YOLOv8-seg model.
- **🖋️ Polygon Annotation:** Supports creating, editing, and deleting polygon-shaped annotations with intuitive mouse controls.
- **🤖 YOLOv8 Model Integration:** Easily loads custom-trained YOLOv8-seg `.pt` models for inference and fine-tuning.
- **🚀 Model Fine-Tuning:** 
    - A dedicated dialog allows for detailed configuration of hyperparameters for training (e.g., epochs, batch size, learning rate, optimizer).
    - Supports extensive data augmentation options (geometry, color, etc.).
    - Training runs as a background process, with detailed logs printed directly to the console.
    - Upon successful completion, the original model file is automatically updated with the newly trained best weights.
- **📊 Confidence Score Visualization:** Displays the confidence score for each instance and the average score for the current image.
- **↔️ Flexible Export:** Allows exporting all annotated images and labels to user-selected destination folders for images and labels separately.
- **🖱️ User-Friendly Interface:**
  - Zoom in/out (mouse wheel) and pan (middle-click drag).
  - Stable layout with non-closable panels for file, class, and instance lists.
  - Rich support for keyboard shortcuts to accelerate workflow.

## ⚙️ Requirements

The main dependencies are listed in `requirements.txt`.

- Python 3.8+
- PyQt5
- ultralytics
- numpy
- rdp
- opencv-python-headless
- torch
- torchvision

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hyeok90/PyQt_active_learning
    cd PyQt_active_learning
    ```

2.  **Install PyTorch:**
    For GPU support (recommended), follow the official instructions on the [PyTorch website](https://pytorch.org/get-started/locally/) to install PyTorch matching your CUDA version. An example command is:
    ```bash
    # Example for CUDA 12.1
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```
    If you only have a CPU, you can install it as follows:
    ```bash
    pip3 install torch torchvision torchaudio
    ```

3.  **Install other dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🏃‍♂️ Usage

1.  **Run the application:**
    ```bash
    python main.py
    ```

2.  **Workflow:**
    - **1. Load Model:** Click `1. Load Model (.pt)` to load your trained YOLOv8 segmentation model.
    - **2. Open Image Folder:** Click `2. Open Image Folder` to open a directory containing your images.
    - **3. Annotate & Review:** Navigate through images (`A`/`D`), modify auto-generated labels, or create new ones (`W`). Changes are saved automatically or manually (`Ctrl+S`).
    - **4. Fine-Tune Model:** Click `Train`, select your dataset's `.yaml` file, adjust hyperparameters, and start training. Monitor the progress in the console where you launched the application.
    - **5. Export:** Click `3. Export` to move all images and labels to separate destination folders. The workspace will be cleared after the export.

## ⌨️ Shortcuts

| Key | Action |
| :--- | :--- |
| `A` | Previous Image |
| `D` | Next Image |
| `W` | Toggle Polygon Draw Mode |
| `Ctrl+S` | Save Current Labels |
| `Ctrl+Z` | Undo Last Shape Modification |
| `Delete` / `Backspace` | Delete Selected Instance(s) |
| `Mouse Wheel` | Zoom In / Out |
| `Middle Mouse Drag` | Pan Image |

