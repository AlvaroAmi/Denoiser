import sys
import torch
import numpy as np
from PIL import Image
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QScrollArea, 
                           QLabel, QSplitter)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor
from torchvision import transforms
import os

from denoiser_functions import EnhancedUNet, create_patches, reconstruct_image

class ImageViewer(QWidget):
    def __init__(self, title=""):
        super().__init__()
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.pressing = False
        self.last_pos = None
        self.title = title

        self.layout = QVBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.title_label)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.scroll_area.setWidget(self.image_label)
        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)
        self.pixmap = None
        self.original_pixmap = None

    def set_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)
        self.update_view()

    def set_image_from_pil(self, pil_image):
        img = pil_image.convert('RGB')
        data = img.tobytes('raw', 'RGB')
        qimg = QImage(data, img.size[0], img.size[1], img.size[0] * 3, QImage.Format.Format_RGB888)
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.update_view()

    def update_view(self):
        if self.original_pixmap:
            scaled_size = self.original_pixmap.size() * self.zoom
            self.pixmap = self.original_pixmap.scaled(
                scaled_size.width(), 
                scaled_size.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.image_label.setPixmap(self.pixmap)

    def wheelEvent(self, event):
        if self.original_pixmap:
            old_zoom = self.zoom
            if event.angleDelta().y() > 0:
                self.zoom *= 1.2
            else:
                self.zoom /= 1.2
            self.zoom = min(max(0.1, self.zoom), 10.0)
            
            # Ajustar el desplazamiento para mantener el punto bajo el cursor
            if old_zoom != self.zoom:
                self.update_view()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressing = True
            self.last_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.pressing = False

    def mouseMoveEvent(self, event):
        if self.pressing and self.last_pos:
            delta = event.pos() - self.last_pos
            self.scroll_area.horizontalScrollBar().setValue(
                self.scroll_area.horizontalScrollBar().value() - delta.x())
            self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().value() - delta.y())
            self.last_pos = event.pos()

class DenoiserApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.load_model()

    def init_ui(self):
        self.setWindowTitle('Denoiser App')
        self.setGeometry(100, 100, 1200, 800)

        # Widget principal
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()

        # Botones
        button_layout = QHBoxLayout()
        self.load_button = QPushButton('Cargar Imagen(es)')
        self.load_button.clicked.connect(self.load_images)
        self.process_button = QPushButton('Procesar')
        self.process_button.clicked.connect(self.process_images)
        self.process_button.setEnabled(False)
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.process_button)
        layout.addLayout(button_layout)

        # Área de visualización
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_viewer = ImageViewer("Imagen Original")
        self.processed_viewer = ImageViewer("Imagen Procesada")
        self.splitter.addWidget(self.original_viewer)
        self.splitter.addWidget(self.processed_viewer)
        layout.addWidget(self.splitter)

        main_widget.setLayout(layout)
        
        self.current_image_path = None
        self.show()

    def load_model(self):
        # Cargar el modelo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedUNet().to(self.device)
        checkpoint = torch.load('checkpoints/best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("Modelo cargado correctamente")

    def load_images(self):
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            "Seleccionar Imagen(es)",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_paths:
            self.current_image_path = file_paths[0]  # Por ahora solo mostramos la primera imagen
            self.original_viewer.set_image(self.current_image_path)
            self.process_button.setEnabled(True)

    def process_images(self):
        if self.current_image_path:
            # Preparar la imagen
            transform = transforms.Compose([transforms.ToTensor()])
            
            # Crear patches y procesar
            patches, positions, _, original_size, window = create_patches(
                self.current_image_path, 
                output_dir=None
            )
            
            # Procesar cada patch
            denoised_patches = []
            with torch.no_grad():
                for patch in patches:
                    patch_tensor = transform(patch).unsqueeze(0).to(self.device)
                    denoised_patch = self.model(patch_tensor)
                    denoised_patch = transforms.ToPILImage()(denoised_patch.squeeze().cpu())
                    denoised_patches.append(denoised_patch)
            
            # Reconstruir la imagen completa
            denoised_full = reconstruct_image(denoised_patches, positions, original_size, window)
            
            # Mostrar la imagen procesada
            self.processed_viewer.set_image_from_pil(denoised_full)

def main():
    app = QApplication(sys.argv)
    ex = DenoiserApp()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()