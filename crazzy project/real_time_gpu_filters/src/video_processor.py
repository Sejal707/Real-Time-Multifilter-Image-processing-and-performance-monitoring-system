import cv2
import numpy as np
import threading
import queue
import time
from typing import Optional, Callable, Tuple
from .filter_manager import FilterManager

class VideoProcessor:
    def __init__(self, source=0, target_fps: int = 60):
        self.source = source
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        self.filter_manager = FilterManager()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.is_paused = False
        
        # Threading components
        self.frame_queue = queue.Queue(maxsize=5)
        self.processed_queue = queue.Queue(maxsize=5)
        self.capture_thread: Optional[threading.Thread] = None
        self.processing_thread: Optional[threading.Thread] = None
        
        # Frame statistics
        self.frame_count = 0
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0.0
        
        # Frame callback
        self.frame_callback: Optional[Callable] = None
        
    def initialize_capture(self) -> bool:
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            
            # Optimize capture settings for RTX 5070Ti
            if self.cap.isOpened():
                # Set high performance settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                
                # GPU acceleration if available
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                return True
        except Exception as e:
            print(f"Failed to initialize capture: {e}")
            
        return False
        
    def start_processing(self):
        """Start video processing pipeline"""
        if not self.initialize_capture():
            raise RuntimeError("Failed to initialize video capture")
            
        self.is_running = True
        
        # Start threads
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        
        self.capture_thread.start()
        self.processing_thread.start()
        
    def stop_processing(self):
        """Stop video processing"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
            
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
            
        if self.cap:
            self.cap.release()
            
    def pause(self):
        """Pause processing"""
        self.is_paused = True
        
    def resume(self):
        """Resume processing"""
        self.is_paused = False
        
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str):  # Video file ended
                    break
                continue
                
            # Add frame to queue (non-blocking)
            try:
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                # Drop oldest frame if queue is full
                try:
                    self.frame_queue.get(block=False)
                    self.frame_queue.put(frame, block=False)
                except queue.Empty:
                    pass
                    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        while self.is_running:
            if self.is_paused:
                time.sleep(0.1)
                continue
                
            try:
                # Get frame from capture queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Apply filters
                start_time = time.time()
                processed_frame = self.filter_manager.apply_filters(frame)
                processing_time = time.time() - start_time
                
                # Add timestamp and performance info
                processed_frame = self._add_performance_overlay(processed_frame, processing_time)
                
                # Add to output queue
                try:
                    self.processed_queue.put(processed_frame, block=False)
                except queue.Full:
                    # Drop oldest processed frame
                    try:
                        self.processed_queue.get(block=False)
                        self.processed_queue.put(processed_frame, block=False)
                    except queue.Empty:
                        pass
                
                # Update FPS counter
                self._update_fps_counter()
                
                # Call frame callback if set
                if self.frame_callback:
                    self.frame_callback(processed_frame)
                    
            except queue.Empty:
                continue
                
    def _add_performance_overlay(self, frame: np.ndarray, processing_time: float) -> np.ndarray:
        """Add performance information overlay to frame"""
        overlay_frame = frame.copy()
        
        # Performance text
        fps_text = f"FPS: {self.current_fps:.1f}"
        process_text = f"Process: {processing_time*1000:.1f}ms"
        
        # GPU memory info (NVML or backend fallback)
        gpu_text = "GPU: N/A"
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_text = f"GPU: {mem_info.used/1024**3:.1f}GB/{mem_info.total/1024**3:.1f}GB"
        except Exception:
            # Try backend-provided memory info (CuPy mempool)
            try:
                mem = self.filter_manager.gpu_manager.get_memory_info()  # type: ignore[attr-defined]
                used = mem.get('used_gb')
                total = mem.get('total_gb')
                if used is not None and total is not None:
                    gpu_text = f"GPU: {used:.1f}GB/{total:.1f}GB"
            except Exception:
                pass
        
        # Draw text overlay
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thickness = 2
        
        cv2.putText(overlay_frame, fps_text, (10, 30), font, font_scale, color, thickness)
        cv2.putText(overlay_frame, process_text, (10, 60), font, font_scale, color, thickness)
        cv2.putText(overlay_frame, gpu_text, (10, 90), font, font_scale, color, thickness)
        
        return overlay_frame
        
    def _update_fps_counter(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_time)
            self.fps_counter = 0
            self.fps_time = current_time
            
    def get_processed_frame(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the latest processed frame"""
        try:
            return self.processed_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def set_frame_callback(self, callback: Callable):
        """Set callback function for processed frames"""
        self.frame_callback = callback
        
    def get_filter_manager(self) -> FilterManager:
        """Get the filter manager instance"""
        return self.filter_manager
        
    def get_frame_info(self) -> Tuple[int, int]:
        """Get current frame dimensions"""
        if self.cap:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0
