import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QComboBox, QSlider, 
                           QGroupBox, QListWidget, QListWidgetItem, QCheckBox,
                           QSpinBox, QDoubleSpinBox, QTabWidget, QTextEdit,
                           QFileDialog, QMessageBox, QProgressBar, QSplitter)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
import json
import os
from typing import Dict, Any
from .video_processor import VideoProcessor

class FilterControlWidget(QWidget):
    def __init__(self, filter_manager, parent=None):
        super().__init__(parent)
        self.filter_manager = filter_manager
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Filter presets
        preset_group = QGroupBox("Filter Presets")
        preset_layout = QVBoxLayout(preset_group)
        
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(self.filter_manager.get_available_presets())
        self.preset_combo.currentTextChanged.connect(self.load_preset)
        
        preset_layout.addWidget(QLabel("Preset:"))
        preset_layout.addWidget(self.preset_combo)
        
        # Individual filters
        filter_group = QGroupBox("Individual Filters")
        filter_layout = QVBoxLayout(filter_group)
        
        # Available filters
        available_filters = self.filter_manager.gpu_manager.get_available_filters()
        
        # Gaussian Blur controls
        blur_widget = self.create_blur_controls()
        filter_layout.addWidget(blur_widget)
        
        # Color Temperature controls
        temp_widget = self.create_temperature_controls()
        filter_layout.addWidget(temp_widget)
        
        # Edge Detection
        edge_btn = QPushButton("Toggle Sobel Edge")
        edge_btn.clicked.connect(self.toggle_edge_filter)
        filter_layout.addWidget(edge_btn)
        
        # Emboss Effect
        emboss_btn = QPushButton("Toggle Emboss")
        emboss_btn.clicked.connect(self.toggle_emboss_filter)
        filter_layout.addWidget(emboss_btn)
        
        # Bilateral Filter controls
        bilateral_widget = self.create_bilateral_controls()
        filter_layout.addWidget(bilateral_widget)
        
        # Clear all filters
        clear_btn = QPushButton("Clear All Filters")
        clear_btn.clicked.connect(self.filter_manager.clear_filters)
        clear_btn.setStyleSheet("QPushButton { background-color: #ff6b6b; color: white; font-weight: bold; }")
        filter_layout.addWidget(clear_btn)

        # Sharpen
        sharpen_btn = QPushButton("Toggle Sharpen")
        sharpen_btn.clicked.connect(self.toggle_sharpen_filter)
        filter_layout.addWidget(sharpen_btn)

        # Grayscale
        grayscale_btn = QPushButton("Toggle Grayscale")
        grayscale_btn.clicked.connect(self.toggle_grayscale_filter)
        filter_layout.addWidget(grayscale_btn)

        # Brightness/Contrast controls
        bc_group = QGroupBox("Brightness / Contrast")
        bc_layout = QVBoxLayout(bc_group)
        # Contrast (alpha)
        alpha_row = QHBoxLayout()
        alpha_row.addWidget(QLabel("Contrast (alpha):"))
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setRange(5, 200)  # 0.5 - 2.0
        self.alpha_slider.setValue(100)
        self.alpha_label = QLabel("1.0")
        self.alpha_slider.valueChanged.connect(lambda v: self.alpha_label.setText(f"{v/100:.2f}"))
        alpha_row.addWidget(self.alpha_slider)
        alpha_row.addWidget(self.alpha_label)
        # Brightness (beta)
        beta_row = QHBoxLayout()
        beta_row.addWidget(QLabel("Brightness (beta):"))
        self.beta_slider = QSlider(Qt.Horizontal)
        self.beta_slider.setRange(-100, 100)
        self.beta_slider.setValue(0)
        self.beta_label = QLabel("0")
        self.beta_slider.valueChanged.connect(lambda v: self.beta_label.setText(str(v)))
        beta_row.addWidget(self.beta_slider)
        beta_row.addWidget(self.beta_label)
        bc_apply = QPushButton("Apply Brightness/Contrast")
        bc_apply.clicked.connect(self.apply_brightness_contrast)
        bc_layout.addLayout(alpha_row)
        bc_layout.addLayout(beta_row)
        bc_layout.addWidget(bc_apply)
        filter_layout.addWidget(bc_group)
        
        layout.addWidget(preset_group)
        layout.addWidget(filter_group)
        
    def create_blur_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Gaussian Blur"))
        
        # Kernel size
        kernel_layout = QHBoxLayout()
        kernel_layout.addWidget(QLabel("Kernel Size:"))
        self.kernel_slider = QSlider(Qt.Horizontal)
        self.kernel_slider.setRange(3, 31)
        self.kernel_slider.setValue(15)
        self.kernel_slider.setSingleStep(2)
        self.kernel_label = QLabel("15")
        
        self.kernel_slider.valueChanged.connect(
            lambda v: self.kernel_label.setText(str(v))
        )
        
        kernel_layout.addWidget(self.kernel_slider)
        kernel_layout.addWidget(self.kernel_label)
        
        # Sigma
        sigma_layout = QHBoxLayout()
        sigma_layout.addWidget(QLabel("Sigma:"))
        self.sigma_slider = QSlider(Qt.Horizontal)
        self.sigma_slider.setRange(1, 100)
        self.sigma_slider.setValue(20)
        self.sigma_label = QLabel("2.0")
        
        self.sigma_slider.valueChanged.connect(
            lambda v: self.sigma_label.setText(f"{v/10:.1f}")
        )
        
        sigma_layout.addWidget(self.sigma_slider)
        sigma_layout.addWidget(self.sigma_label)
        
        # Apply button
        blur_btn = QPushButton("Apply Gaussian Blur")
        blur_btn.clicked.connect(self.apply_blur_filter)
        
        layout.addLayout(kernel_layout)
        layout.addLayout(sigma_layout)
        layout.addWidget(blur_btn)
        
        return widget
        
    def create_temperature_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Color Temperature"))
        
        temp_layout = QHBoxLayout()
        temp_layout.addWidget(QLabel("Temperature:"))
        self.temp_slider = QSlider(Qt.Horizontal)
        self.temp_slider.setRange(-50, 50)
        self.temp_slider.setValue(0)
        self.temp_label = QLabel("0")
        
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(str(v))
        )
        
        temp_layout.addWidget(self.temp_slider)
        temp_layout.addWidget(self.temp_label)
        
        temp_btn = QPushButton("Apply Color Temperature")
        temp_btn.clicked.connect(self.apply_temperature_filter)
        
        layout.addLayout(temp_layout)
        layout.addWidget(temp_btn)
        
        return widget
        
    def create_bilateral_controls(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        layout.addWidget(QLabel("Bilateral Filter"))
        
        # Radius
        radius_layout = QHBoxLayout()
        radius_layout.addWidget(QLabel("Radius:"))
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(1, 15)
        self.radius_slider.setValue(5)
        self.radius_label = QLabel("5")
        
        self.radius_slider.valueChanged.connect(
            lambda v: self.radius_label.setText(str(v))
        )
        
        radius_layout.addWidget(self.radius_slider)
        radius_layout.addWidget(self.radius_label)
        
        bilateral_btn = QPushButton("Apply Bilateral Filter")
        bilateral_btn.clicked.connect(self.apply_bilateral_filter)
        
        layout.addLayout(radius_layout)
        layout.addWidget(bilateral_btn)
        
        return widget
        
    def load_preset(self, preset_name: str):
        if preset_name:
            self.filter_manager.load_preset(preset_name)
            
    def apply_blur_filter(self):
        kernel_size = self.kernel_slider.value()
        sigma = self.sigma_slider.value() / 10.0
        
        self.filter_manager.add_filter('gaussian_blur', {
            'kernel_size': kernel_size,
            'sigma': sigma
        })
        
    def apply_temperature_filter(self):
        temperature = float(self.temp_slider.value())
        
        self.filter_manager.add_filter('color_temperature', {
            'temperature': temperature
        })
        
    def apply_bilateral_filter(self):
        radius = self.radius_slider.value()
        
        self.filter_manager.add_filter('bilateral_filter', {
            'radius': radius,
            'sigma_color': 50.0,
            'sigma_space': 50.0
        })
        
    def toggle_edge_filter(self):
        self.filter_manager.add_filter('sobel_edge')
        
    def toggle_emboss_filter(self):
        self.filter_manager.add_filter('emboss')

    def toggle_sharpen_filter(self):
        self.filter_manager.add_filter('sharpen')

    def toggle_grayscale_filter(self):
        self.filter_manager.add_filter('grayscale')

    def apply_brightness_contrast(self):
        alpha = self.alpha_slider.value() / 100.0
        beta = float(self.beta_slider.value())
        self.filter_manager.add_filter('brightness_contrast', {
            'alpha': alpha,
            'beta': beta,
        })

class PerformanceWidget(QWidget):
    def __init__(self, video_processor, parent=None):
        super().__init__(parent)
        self.video_processor = video_processor
        self.setup_ui()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(1000)  # Update every second
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Performance metrics
        metrics_group = QGroupBox("Performance Metrics")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setFont(QFont("Arial", 12, QFont.Bold))
        
        self.gpu_usage_label = QLabel("GPU Usage: N/A")
        self.memory_label = QLabel("GPU Memory: N/A")
        self.filter_time_label = QLabel("Filter Time: 0.0ms")
        
        metrics_layout.addWidget(self.fps_label)
        metrics_layout.addWidget(self.gpu_usage_label)
        metrics_layout.addWidget(self.memory_label)
        metrics_layout.addWidget(self.filter_time_label)
        
        # Filter performance
        filter_group = QGroupBox("Filter Performance")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_list = QTextEdit()
        self.filter_list.setMaximumHeight(150)
        self.filter_list.setReadOnly(True)
        
        filter_layout.addWidget(self.filter_list)
        
        layout.addWidget(metrics_group)
        layout.addWidget(filter_group)
        
    def update_stats(self):
        stats = self.video_processor.get_filter_manager().get_performance_stats()
        
        # Update main metrics
        self.fps_label.setText(f"FPS: {stats.get('fps', 0):.1f}")
        self.filter_time_label.setText(f"Total Time: {stats.get('total_time', 0)*1000:.1f}ms")
        
        # Update GPU info
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # GPU utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            self.gpu_usage_label.setText(f"GPU Usage: {utilization.gpu}%")
            
            # Memory usage
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / 1024**3
            total_gb = mem_info.total / 1024**3
            self.memory_label.setText(f"GPU Memory: {used_gb:.1f}GB / {total_gb:.1f}GB")
            
        except Exception:
            # Backend fallback for memory
            try:
                mem = self.video_processor.get_filter_manager().gpu_manager.get_memory_info()  # type: ignore[attr-defined]
                used_gb = mem.get('used_gb')
                total_gb = mem.get('total_gb')
                if used_gb is not None and total_gb is not None:
                    self.gpu_usage_label.setText("GPU Usage: N/A")
                    self.memory_label.setText(f"GPU Memory: {used_gb:.1f}GB / {total_gb:.1f}GB")
                else:
                    raise RuntimeError("no backend mem info")
            except Exception:
                # Fallback to nvidia-smi via subprocess
                try:
                    import subprocess
                    out = subprocess.check_output([
                        'nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'
                    ], stderr=subprocess.DEVNULL, shell=False, timeout=1.0)
                    parts = out.decode('utf-8').strip().split(',')
                    if len(parts) >= 3:
                        util = parts[0].strip()
                        used = float(parts[1].strip()) / 1024.0
                        total = float(parts[2].strip()) / 1024.0
                        self.gpu_usage_label.setText(f"GPU Usage: {util}%")
                        self.memory_label.setText(f"GPU Memory: {used:.1f}GB / {total:.1f}GB")
                    else:
                        raise RuntimeError("unexpected nvidia-smi output")
                except Exception:
                    self.gpu_usage_label.setText("GPU Usage: N/A")
                    self.memory_label.setText("GPU Memory: N/A")
            
        # Update filter performance
        filter_times = stats.get('filter_times', {})
        filter_text = ""
        for filter_name, time_ms in filter_times.items():
            filter_text += f"{filter_name}: {time_ms*1000:.2f}ms\n"
        self.filter_list.setText(filter_text)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_processor = VideoProcessor()
        self.current_frame = None
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        self.setWindowTitle("Real-Time GPU Image Filter Application")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget with splitter
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        
        # Video display area
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Camera")
        self.start_btn.setMinimumHeight(40)
        self.start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-weight: bold; }")
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setEnabled(False)
        self.pause_btn.setMinimumHeight(40)
        
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.setMinimumHeight(40)
        
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.load_video_btn)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("QLabel { border: 2px solid gray; background-color: black; }")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("Video will appear here")
        
        video_layout.addLayout(control_layout)
        video_layout.addWidget(self.video_label)
        
        # Control panel
        control_panel = QTabWidget()
        control_panel.setMaximumWidth(400)
        
        # Filter controls tab
        self.filter_controls = FilterControlWidget(self.video_processor.get_filter_manager())
        control_panel.addTab(self.filter_controls, "Filters")
        
        # Performance tab
        self.performance_widget = PerformanceWidget(self.video_processor)
        control_panel.addTab(self.performance_widget, "Performance")
        
        # Add to splitter
        splitter.addWidget(video_widget)
        splitter.addWidget(control_panel)
        splitter.setSizes([1000, 400])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
        # Display update timer
        self.display_timer = QTimer()
        self.display_timer.timeout.connect(self.update_display)
        
    def setup_connections(self):
        self.start_btn.clicked.connect(self.start_camera)
        self.stop_btn.clicked.connect(self.stop_camera)
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.load_video_btn.clicked.connect(self.load_video_file)
        
    def start_camera(self):
        try:
            self.video_processor.start_processing()
            self.display_timer.start(16)  # ~60 FPS display update
            
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.pause_btn.setEnabled(True)
            
            self.statusBar().showMessage("Camera started")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start camera: {str(e)}")
            
    def stop_camera(self):
        self.display_timer.stop()
        self.video_processor.stop_processing()
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        
        self.video_label.setText("Video stopped")
        self.statusBar().showMessage("Camera stopped")
        
    def toggle_pause(self):
        if self.video_processor.is_paused:
            self.video_processor.resume()
            self.pause_btn.setText("Pause")
            self.statusBar().showMessage("Resumed")
        else:
            self.video_processor.pause()
            self.pause_btn.setText("Resume")
            self.statusBar().showMessage("Paused")
            
    def load_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        
        if file_path:
            self.stop_camera()
            self.video_processor = VideoProcessor(source=file_path)
            
            # Update control widgets with new processor
            self.filter_controls.filter_manager = self.video_processor.get_filter_manager()
            self.performance_widget.video_processor = self.video_processor
            
            self.statusBar().showMessage(f"Loaded: {os.path.basename(file_path)}")
            
    def update_display(self):
        frame = self.video_processor.get_processed_frame()
        if frame is not None:
            self.current_frame = frame
            self.display_frame(frame)
            
    def display_frame(self, frame):
        # Convert frame to Qt format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        label_size = self.video_label.size()
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        self.video_label.setPixmap(scaled_pixmap)
        
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern look
    
    # Set application icon and style
    app.setApplicationName("GPU Image Filter")
    app.setApplicationVersion("1.0")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
