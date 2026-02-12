import numpy as np
from typing import Dict, List, Any, Callable
import time
import os
from dataclasses import dataclass

# Prefer CuPy backend; fallback to PyCUDA; final fallback is CPU (OpenCV/Numpy)
try:
    from .gpu_filters_cupy import CuPyFilterManager  # type: ignore
    _HAS_CUPY = True
except Exception:
    CuPyFilterManager = None  # type: ignore
    _HAS_CUPY = False

try:
    from .gpu_filters import GPUFilterManager  # type: ignore
    _HAS_PYCUDA = True
except Exception:
    GPUFilterManager = None  # type: ignore
    _HAS_PYCUDA = False

try:
    import cv2
except Exception:
    cv2 = None  # type: ignore


@dataclass
class FilterConfig:
    name: str
    function: Callable
    parameters: Dict[str, Any]
    enabled: bool = True
    processing_time: float = 0.0


class _CPUFilterManager:
    """Minimal CPU fallback implementation to keep the app functional without GPU libs."""
    def __init__(self):
        pass

    def apply_gaussian_blur(self, frame: np.ndarray, kernel_size: int = 15, sigma: float = 2.0) -> np.ndarray:
        if cv2 is None:
            return frame
        k = max(3, kernel_size | 1)
        return cv2.GaussianBlur(frame, (k, k), sigmaX=sigma, sigmaY=sigma)

    def apply_sobel_edge(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return frame
        grad_x = cv2.Sobel(frame, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(frame, cv2.CV_16S, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    def apply_color_temperature(self, frame: np.ndarray, temperature: float = 0.0) -> np.ndarray:
        scale_r = 1.0 + (temperature * 0.01)
        scale_b = 1.0 - (temperature * 0.01)
        result = frame.astype(np.float32).copy()
        # BGR ordering in OpenCV
        result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)
        result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
        return result.astype(np.uint8)

    def apply_bilateral_filter(self, frame: np.ndarray, radius: int = 5, sigma_color: float = 50.0, sigma_space: float = 50.0) -> np.ndarray:
        if cv2 is None:
            return frame
        d = max(1, int(radius) * 2 + 1)
        return cv2.bilateralFilter(frame, d=d, sigmaColor=float(sigma_color), sigmaSpace=float(sigma_space))

    def apply_emboss(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return frame
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        embossed = cv2.filter2D(frame, -1, kernel) + 128
        return np.clip(embossed, 0, 255).astype(np.uint8)

    def apply_sharpen(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return frame
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(frame, -1, kernel)

    def apply_grayscale(self, frame: np.ndarray) -> np.ndarray:
        if cv2 is None:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def apply_brightness_contrast(self, frame: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
        result = frame.astype(np.float32) * float(alpha) + float(beta)
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_available_filters(self) -> Dict[str, Callable]:
        return {
            'gaussian_blur': self.apply_gaussian_blur,
            'sobel_edge': self.apply_sobel_edge,
            'color_temperature': self.apply_color_temperature,
            'bilateral_filter': self.apply_bilateral_filter,
            'emboss': self.apply_emboss,
            'sharpen': self.apply_sharpen,
            'grayscale': self.apply_grayscale,
            'brightness_contrast': self.apply_brightness_contrast,
        }


class FilterManager:
    def __init__(self):
        # Select backend with environment override and runtime probe
        self.gpu_manager = self._select_backend()

        # CPU fallback for filters missing in the active backend
        self._cpu_fallback = _CPUFilterManager()
        self._cpu_fallback_map: Dict[str, Callable] = self._cpu_fallback.get_available_filters()

        self.active_filters: List[FilterConfig] = []
        self.filter_presets = self._create_presets()
        self.performance_stats = {
            'total_frames': 0,
            'total_time': 0.0,
            'fps': 0.0,
            'filter_times': {}
        }

    def _select_backend(self):
        backend_pref = os.environ.get('RTGF_BACKEND', '').strip().lower()

        def safe_create_cupy():
            if not (_HAS_CUPY and CuPyFilterManager is not None):
                return None
            try:
                mgr = CuPyFilterManager()
                # Probe a tiny op to ensure nvrtc/cuDNN stack is loadable
                import cupy as cp  # type: ignore
                _ = (cp.ones((1, 1), dtype=cp.float32) + 1).sum()
                return mgr
            except Exception:
                return None

        def safe_create_pycuda():
            if not (_HAS_PYCUDA and GPUFilterManager is not None):
                return None
            try:
                return GPUFilterManager()
            except Exception:
                return None

        if backend_pref == 'cpu':
            return _CPUFilterManager()
        if backend_pref == 'cupy':
            return safe_create_cupy() or _CPUFilterManager()
        if backend_pref == 'pycuda':
            return safe_create_pycuda() or _CPUFilterManager()

        # Auto: try CuPy, then PyCUDA, then CPU
        return safe_create_cupy() or safe_create_pycuda() or _CPUFilterManager()
        
    def _create_presets(self) -> Dict[str, List[FilterConfig]]:
        """Create predefined filter combinations"""
        available_filters = self.gpu_manager.get_available_filters()
        
        return {
            'Beauty': [
                FilterConfig('bilateral_filter', available_filters['bilateral_filter'], 
                           {'radius': 3, 'sigma_color': 30, 'sigma_space': 30}),
                FilterConfig('color_temperature', available_filters['color_temperature'], 
                           {'temperature': 5.0})
            ],
            'Artistic': [
                FilterConfig('emboss', available_filters['emboss'], {}),
                FilterConfig('color_temperature', available_filters['color_temperature'], 
                           {'temperature': -10.0})
            ],
            'Edge_Enhancement': [
                FilterConfig('sobel_edge', available_filters['sobel_edge'], {}),
                FilterConfig('gaussian_blur', available_filters['gaussian_blur'], 
                           {'kernel_size': 3, 'sigma': 0.5})
            ],
            'Cinematic': [
                FilterConfig('bilateral_filter', available_filters['bilateral_filter'], 
                           {'radius': 5, 'sigma_color': 50, 'sigma_space': 50}),
                FilterConfig('color_temperature', available_filters['color_temperature'], 
                           {'temperature': -15.0})
            ],
            'SharpenAndTone': [
                FilterConfig('sharpen', available_filters.get('sharpen', available_filters['gaussian_blur']), {}),
                FilterConfig('brightness_contrast', available_filters.get('brightness_contrast', available_filters['color_temperature']), 
                           {'alpha': 1.2, 'beta': 10.0})
            ]
        }
        
    def apply_filters(self, frame: np.ndarray) -> np.ndarray:
        """Apply all active filters to frame"""
        start_time = time.time()
        result = frame.copy()
        
        for filter_config in self.active_filters:
            if not filter_config.enabled:
                continue
                
            filter_start = time.time()
            try:
                result = filter_config.function(result, **filter_config.parameters)
            except Exception as e:
                # Try CPU fallback for this filter if GPU path failed
                cpu_func = self._cpu_fallback_map.get(filter_config.name)
                if cpu_func is not None:
                    try:
                        result = cpu_func(result, **filter_config.parameters)
                        # Switch to CPU func for subsequent frames to avoid repeated errors
                        filter_config.function = cpu_func
                    except Exception as e2:
                        print(f"Error applying filter {filter_config.name} (CPU fallback): {e2}")
                        continue
                else:
                    print(f"Error applying filter {filter_config.name}: {e}")
                    continue
            
            filter_time = time.time() - filter_start
            filter_config.processing_time = filter_time
            self.performance_stats['filter_times'][filter_config.name] = filter_time
        
        total_time = time.time() - start_time
        self.performance_stats['total_frames'] += 1
        self.performance_stats['total_time'] += total_time
        
        if self.performance_stats['total_frames'] % 30 == 0:  # Update every 30 frames
            avg_time = self.performance_stats['total_time'] / self.performance_stats['total_frames']
            self.performance_stats['fps'] = 1.0 / avg_time if avg_time > 0 else 0
        
        return result
        
    def add_filter(self, filter_name: str, parameters: Dict[str, Any] = None):
        """Add a filter to the active pipeline"""
        available_filters = self.gpu_manager.get_available_filters()
        
        if filter_name not in available_filters:
            # Try CPU fallback for missing filters (e.g., 'sharpen', 'grayscale', etc.)
            if filter_name in self._cpu_fallback_map:
                def cpu_wrapper(frame: np.ndarray, **kwargs):
                    return self._cpu_fallback_map[filter_name](frame, **kwargs)
                func = cpu_wrapper
            else:
                raise ValueError(f"Filter {filter_name} not available")
        else:
            func = available_filters[filter_name]
            
        params = parameters or {}
        filter_config = FilterConfig(
            name=filter_name,
            function=func,
            parameters=params
        )
        
        self.active_filters.append(filter_config)
        
    def remove_filter(self, index: int):
        """Remove filter by index"""
        if 0 <= index < len(self.active_filters):
            del self.active_filters[index]
            
    def toggle_filter(self, index: int):
        """Toggle filter on/off"""
        if 0 <= index < len(self.active_filters):
            self.active_filters[index].enabled = not self.active_filters[index].enabled
            
    def load_preset(self, preset_name: str):
        """Load a filter preset"""
        if preset_name in self.filter_presets:
            self.active_filters = self.filter_presets[preset_name].copy()
        else:
            raise ValueError(f"Preset {preset_name} not found")
            
    def update_filter_parameter(self, filter_index: int, param_name: str, value: Any):
        """Update a filter parameter"""
        if 0 <= filter_index < len(self.active_filters):
            self.active_filters[filter_index].parameters[param_name] = value
            
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.performance_stats.copy()
        
    def clear_filters(self):
        """Remove all active filters"""
        self.active_filters.clear()
        
    def get_available_presets(self) -> List[str]:
        """Get list of available filter presets"""
        return list(self.filter_presets.keys())
