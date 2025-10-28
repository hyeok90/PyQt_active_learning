# YOLOv8-seg Active Learning-based Annotation Tool

This is a GUI-based image annotation tool developed with Python and PyQt5, designed for efficient object segmentation tasks. It leverages YOLOv8-seg models to implement an active learning workflow, significantly speeding up the labeling process.

![image](https://github.com/user-attachments/assets/b3f73189-7153-4111-85ba-f1183a08e59b)


## ✨ Features

- **🧠 Active Learning:** Automatically pre-annotates images without labels using a loaded YOLOv8-seg model.
- **🖋️ Polygon Annotation:** Supports creating, editing, and deleting polygon-shaped annotations with intuitive mouse controls.
- **🤖 YOLO Model Integration:** Easily loads custom-trained YOLOv8-seg `.pt` models.
- **📊 Confidence Score Visualization:** Displays the confidence score for each instance and the average score for the current image. Manually added instances are assigned a confidence of 1.0.
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
    git clone <your-repository-url>
    cd <repository-directory>
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
    - **1. Load Model:** Click the `1. Load Model (.pt)` button to load your trained YOLOv8 segmentation model.
    - **2. Open Image Folder:** Click `2. Open Image Folder` to open a directory containing your images. The tool assumes the following structure:
      ```
      - /your_folder/
        - /images/
          - image1.jpg
          - image2.png
          ...
      ```
      If a corresponding `/labels/` directory does not exist, the tool will create it. For any images without a label file, it will perform prediction and save the results as a new label file.
    - **3. Annotate & Review:**
      - Navigate through images using `A` (previous) and `D` (next) keys.
      - Modify auto-generated labels or create new ones using `W` to enter draw mode.
      - Changes are saved automatically when you move to another image. You can also save manually with `Ctrl+S`.
    - **4. Export:** Click `3. Export` to move all images and labels from the current session to separate destination folders of your choice. The workspace will be cleared after the export.

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

