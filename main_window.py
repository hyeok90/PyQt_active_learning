import os
import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QAction, QFileDialog, 
    QListWidget, QMessageBox, QDockWidget, QListWidgetItem, QInputDialog, QLabel, QMenu
)
from PyQt5.QtGui import QPixmap, QIcon, QColor
from PyQt5.QtCore import Qt, QPointF
from yolo_predictor import RealYOLOPredictor
from image_viewer import ImageViewer
from shape import Shape
from utils import load_yolo_labels, save_yolo_labels

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("YOLOv11-seg Active Learning Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        self.model = None
        self.image_paths = []
        self.current_image_index = -1
        self.class_names = []
        self.color_map = []

        self.viewer = ImageViewer(self)
        self.setCentralWidget(self.viewer)
        
        self.create_actions()
        self.create_tool_bar()
        self.create_docks()
        self.create_status_bar()
        
        self.set_actions_enabled(False)
        self.load_model_action.setEnabled(True)

        self.viewer.polygon_selected.connect(self.on_polygon_selected)
        self.viewer.new_polygon_drawn.connect(self.on_new_polygon_drawn)

    def create_actions(self):
        self.load_model_action = QAction(QIcon.fromTheme("document-open"), "1. Load Model (.pt)", self)
        self.load_model_action.triggered.connect(self.load_model)
        
        self.open_folder_action = QAction(QIcon.fromTheme("folder-open"), "2. Open Image Folder", self)
        self.open_folder_action.triggered.connect(self.open_folder)

        self.export_action = QAction(QIcon.fromTheme("document-send"), "3. Export", self)
        self.export_action.triggered.connect(self.export_files)
        
        self.save_labels_action = QAction(QIcon.fromTheme("document-save"), "Save Labels (Ctrl+S)", self)
        self.save_labels_action.triggered.connect(self.save_current_labels)
        self.save_labels_action.setShortcut("Ctrl+S")
        
        self.prev_image_action = QAction(QIcon.fromTheme("go-previous"), "Previous Image (A)", self)
        self.prev_image_action.triggered.connect(self.prev_image)
        self.prev_image_action.setShortcut("A")

        self.next_image_action = QAction(QIcon.fromTheme("go-next"), "Next Image (D)", self)
        self.next_image_action.triggered.connect(self.next_image)
        self.next_image_action.setShortcut("D")

        self.draw_poly_action = QAction(QIcon.fromTheme("edit-draw"), "Draw Polygon (W)", self)
        self.draw_poly_action.setCheckable(True)
        self.draw_poly_action.triggered.connect(self.toggle_draw_mode)
        self.draw_poly_action.setShortcut("W")
        
        self.fit_window_action = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit to Window", self)
        self.fit_window_action.triggered.connect(self.viewer.fit_to_window)

        self.undo_action = QAction(QIcon.fromTheme("edit-undo"), "Undo (Ctrl+Z)", self)
        self.undo_action.triggered.connect(self.undo_shape)
        self.undo_action.setShortcut("Ctrl+Z")

    def create_tool_bar(self):
        tool_bar = self.addToolBar("Main ToolBar")
        tool_bar.addAction(self.load_model_action)
        tool_bar.addAction(self.open_folder_action)
        tool_bar.addAction(self.export_action)
        tool_bar.addSeparator()
        tool_bar.addAction(self.save_labels_action)
        tool_bar.addAction(self.undo_action)
        tool_bar.addSeparator()
        tool_bar.addAction(self.prev_image_action)
        tool_bar.addAction(self.next_image_action)
        tool_bar.addSeparator()
        tool_bar.addAction(self.draw_poly_action)
        tool_bar.addAction(self.fit_window_action)

    def create_docks(self):
        file_list_dock = QDockWidget("File List", self)
        self.file_list_widget = QListWidget()
        self.file_list_widget.itemClicked.connect(self.on_file_item_clicked)
        file_list_dock.setWidget(self.file_list_widget)
        file_list_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, file_list_dock)
        
        class_list_dock = QDockWidget("Class List", self)
        self.class_list_widget = QListWidget()
        class_list_dock.setWidget(self.class_list_widget)
        class_list_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, class_list_dock)

        instance_list_dock = QDockWidget("Instance List", self)
        self.instance_list_widget = QListWidget()
        self.instance_list_widget.itemClicked.connect(self.on_instance_item_clicked)
        self.instance_list_widget.itemDoubleClicked.connect(self.on_instance_double_clicked)
        instance_list_dock.setWidget(self.instance_list_widget)
        instance_list_dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, instance_list_dock)

    def create_status_bar(self):
        self.statusBar().showMessage("Ready")
        self.conf_label = QLabel("Avg. Confidence: N/A")
        self.statusBar().addPermanentWidget(self.conf_label)

    def set_actions_enabled(self, enabled):
        self.open_folder_action.setEnabled(enabled)
        self.export_action.setEnabled(enabled)
        self.save_labels_action.setEnabled(enabled)
        self.prev_image_action.setEnabled(enabled)
        self.next_image_action.setEnabled(enabled)
        self.draw_poly_action.setEnabled(enabled)
        self.fit_window_action.setEnabled(enabled)
        self.undo_action.setEnabled(enabled)

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load YOLO Model", "", "PyTorch Models (*.pt)")
        if file_path:
            try:
                self.model = RealYOLOPredictor(file_path)
                class_map = self.model.get_class_names()
                self.class_names = [class_map[i] for i in sorted(class_map.keys())]
                self.class_list_widget.clear()
                self.class_list_widget.addItems(self.class_names)
                
                self.color_map = []
                hue_step = 360.0 / len(self.class_names)
                for i in range(len(self.class_names)):
                    color = QColor.fromHsv(int(i * hue_step), 200, 200)
                    self.color_map.append(color)

                self.set_actions_enabled(True)
                self.save_labels_action.setEnabled(False)
                self.prev_image_action.setEnabled(False)
                self.next_image_action.setEnabled(False)
                self.draw_poly_action.setEnabled(False)
                self.fit_window_action.setEnabled(False)
                self.undo_action.setEnabled(False)
                self.export_action.setEnabled(False)
                
                self.statusBar().showMessage(f"Model loaded: {os.path.basename(file_path)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load model: {e}")
                print(e)

    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if folder_path:
            if os.path.isdir(os.path.join(folder_path, "images")):
                folder_path = os.path.join(folder_path, "images")

            self.image_paths = []
            self.file_list_widget.clear()
            
            image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            labels_dir = os.path.join(os.path.dirname(folder_path), "labels")
            os.makedirs(labels_dir, exist_ok=True)
            
            for i, img_file in enumerate(image_files):
                self.statusBar().showMessage(f"Processing {i + 1}/{len(image_files)}: {img_file}")
                QApplication.processEvents()

                img_path = os.path.join(folder_path, img_file)
                txt_path = os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt")

                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error reading image {img_path}")
                    continue
                img_h, img_w = img.shape[:2]
                self.image_paths.append((img_path, (img_w, img_h)))

                if not os.path.exists(txt_path):
                    if self.model:
                        instances, _, _ = self.model.predict_and_optimize(img_path)
                        shapes_to_save = []
                        for inst in instances:
                            class_id, polygon_data, conf = inst
                            class_name = self.class_names[class_id]
                            shape = Shape(label=class_name, shape_type='polygon', score=conf)
                            shape.points = [QPointF(p[0], p[1]) for p in polygon_data]
                            shape.close()
                            shapes_to_save.append(shape)
                        if shapes_to_save:
                            save_yolo_labels(txt_path, shapes_to_save, img_w, img_h, self.class_names)

            self.statusBar().showMessage("Done processing folder.", 5000)
            self.file_list_widget.addItems([os.path.basename(p) for p, d in self.image_paths])
            
            if len(self.image_paths) > 0:
                self.load_image_by_index(0)
                self.set_actions_enabled(True)
                self.fit_window_action.setEnabled(True)
            else:
                self.set_actions_enabled(True)
                self.open_folder_action.setEnabled(True)
                self.load_model_action.setEnabled(True)

    def load_image_by_index(self, index):
        if not (0 <= index < len(self.image_paths)):
            return

        self.save_current_labels()
        
        self.current_image_index = index
        self.file_list_widget.setCurrentRow(index)
        
        img_path, (img_w, img_h) = self.image_paths[index]
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            QMessageBox.warning(self, "Error", f"Failed to load image: {img_path}")
            return

        txt_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        labels_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels")
        txt_path = os.path.join(labels_dir, txt_file)

        self.viewer.clear_polygons()
        self.instance_list_widget.clear()
        
        self.viewer.set_image(pixmap)
        
        shapes = load_yolo_labels(txt_path, img_w, img_h, self.class_names)
        self.viewer.shapes = shapes
        self.viewer.store_shapes() # Initial state for undo
        self.populate_instance_list()
        
        scores = [s.score for s in self.viewer.shapes if s.score is not None]
        avg_conf = sum(scores) / len(scores) if scores else 0.0
        self.conf_label.setText(f"Avg. Confidence: {avg_conf:.2f}")

        self.viewer.fit_to_window()
        self.viewer.update()

    def clear_viewer(self):
        self.viewer.clear_polygons()
        self.viewer.set_image(QPixmap())
        self.instance_list_widget.clear()
        self.conf_label.setText("Avg. Confidence: N/A")
        self.set_actions_enabled(False)
        self.open_folder_action.setEnabled(True)
        self.load_model_action.setEnabled(True)

    def populate_instance_list(self):
        self.instance_list_widget.clear()
        for i, shape in enumerate(self.viewer.shapes):
            score_text = f"({shape.score:.2f})" if shape.score is not None else ""
            item = QListWidgetItem(f"[{i}] {shape.label} {score_text}")
            item.setData(Qt.UserRole, i)
            
            try:
                class_index = self.class_names.index(shape.label)
                color = self.color_map[class_index % len(self.color_map)]
                item.setForeground(color)
                shape.line_color = color
            except ValueError:
                pass # Ignore if class name not in list

            self.instance_list_widget.addItem(item)

    def on_file_item_clicked(self, item):
        index = self.file_list_widget.row(item)
        self.load_image_by_index(index)

    def on_instance_item_clicked(self, item):
        instance_id = item.data(Qt.UserRole)
        shape = self.viewer.shapes[instance_id]
        self.viewer.select_shape(shape)

    def on_polygon_selected(self, shape):
        if not shape:
            self.instance_list_widget.clearSelection()
            return
        instance_id = self.viewer.shapes.index(shape)
        for i in range(self.instance_list_widget.count()):
            item = self.instance_list_widget.item(i)
            if item.data(Qt.UserRole) == instance_id:
                item.setSelected(True)
                break

    def prev_image(self):
        if self.current_image_index > 0:
            self.load_image_by_index(self.current_image_index - 1)

    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.load_image_by_index(self.current_image_index + 1)

    def toggle_draw_mode(self, checked):
        self.viewer.set_draw_mode(checked)

    def on_new_polygon_drawn(self, shape):
        self.draw_poly_action.setChecked(False)
        self.toggle_draw_mode(False)
        self.viewer.store_shapes()
        class_name, ok = QInputDialog.getItem(self, "Select Class", "Class:", self.class_names, 0, False)
        if ok and class_name:
            shape.label = class_name
            shape.score = 1.0
            self.viewer.shapes.append(shape)
            self.populate_instance_list()
            self.viewer.update()
            self.viewer.store_shapes()
        
    def save_current_labels(self):
        if self.current_image_index == -1:
            return

        img_path, (img_w, img_h) = self.image_paths[self.current_image_index]
        
        txt_file = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        labels_dir = os.path.join(os.path.dirname(os.path.dirname(img_path)), "labels")
        txt_path = os.path.join(labels_dir, txt_file)

        save_yolo_labels(txt_path, self.viewer.shapes, img_w, img_h, self.class_names)
        self.statusBar().showMessage(f"Saved labels for {os.path.basename(img_path)}", 2000)

    def export_files(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Warning", "No images to export.")
            return

        dest_images_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory for Images")
        if not dest_images_dir:
            return

        dest_labels_dir = QFileDialog.getExistingDirectory(self, "Select Export Directory for Labels")
        if not dest_labels_dir:
            return

        try:
            total_files = len(self.image_paths)
            for i, (source_img_path, _) in enumerate(self.image_paths):
                self.statusBar().showMessage(f"Exporting {i + 1}/{total_files}: {os.path.basename(source_img_path)}")
                QApplication.processEvents()

                img_filename = os.path.basename(source_img_path)
                txt_filename = os.path.splitext(img_filename)[0] + ".txt"
                
                source_labels_dir = os.path.join(os.path.dirname(os.path.dirname(source_img_path)), "labels")
                source_txt_path = os.path.join(source_labels_dir, txt_filename)

                dest_img_path = os.path.join(dest_images_dir, img_filename)
                dest_txt_path = os.path.join(dest_labels_dir, txt_filename)

                os.rename(source_img_path, dest_img_path)
                if os.path.exists(source_txt_path):
                    os.rename(source_txt_path, dest_txt_path)

            QMessageBox.information(self, "Success", f"{total_files} image(s) and their labels have been exported successfully.")

            # Clear workspace
            self.image_paths = []
            self.file_list_widget.clear()
            self.clear_viewer()
            self.current_image_index = -1

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export files: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            if self.viewer.selected_shapes:
                reply = QMessageBox.question(self, "Delete", "Delete selected instances?", 
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.delete_selected_instances()
        else:
            super().keyPressEvent(event)

    def delete_selected_instances(self):
        self.viewer.store_shapes()
        for shape in self.viewer.selected_shapes:
            self.viewer.shapes.remove(shape)
        self.viewer.deselect_shape()
        self.populate_instance_list()
        self.viewer.update()
        
    def on_instance_double_clicked(self, item):
        instance_id = item.data(Qt.UserRole)
        shape = self.viewer.shapes[instance_id]
        self.change_instance_class(shape)
        
    def change_instance_class(self, shape):
        self.viewer.store_shapes()
        current_class_name = shape.label
        class_name, ok = QInputDialog.getItem(self, "Select Class", "Class:", self.class_names, 
                                            self.class_names.index(current_class_name), False)
        if ok and class_name and class_name != current_class_name:
            shape.label = class_name
            self.populate_instance_list()
            self.viewer.update()
            self.viewer.store_shapes()

    def undo_shape(self):
        self.viewer.restore_shape()
        self.populate_instance_list()
            
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
