"""
Memory management utilities for GPU operations.
"""
import os
import gc
import time
from typing import Optional
import psutil
import torch

class MemoryManager:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.warning_threshold = 0.8  # 80% memory usage warning
        self.critical_threshold = 0.9  # 90% memory usage critical
        
    def get_gpu_memory_info(self) -> tuple[float, float, float]:
        """Get current GPU memory usage information."""
        if self.device != "cuda" or not torch.cuda.is_available():
            return 0.0, 0.0, 0.0
            
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        return allocated, reserved, total
        
    def get_system_memory_info(self) -> tuple[float, float]:
        """Get system memory usage information."""
        vm = psutil.virtual_memory()
        return vm.used / 1024**3, vm.total / 1024**3
        
    def check_memory_pressure(self) -> tuple[bool, str]:
        """
        Check both GPU and system memory pressure.
        
        Returns:
            tuple[bool, str]: (is_critical, status_message)
        """
        if self.device == "cuda":
            allocated, reserved, total = self.get_gpu_memory_info()
            gpu_usage = allocated / total
            
            if gpu_usage > self.critical_threshold:
                return True, f"Critical GPU memory pressure: {gpu_usage:.1%} used"
            elif gpu_usage > self.warning_threshold:
                return False, f"High GPU memory pressure: {gpu_usage:.1%} used"
                
        sys_used, sys_total = self.get_system_memory_info()
        sys_usage = sys_used / sys_total
        
        if sys_usage > self.critical_threshold:
            return True, f"Critical system memory pressure: {sys_usage:.1%} used"
        elif sys_usage > self.warning_threshold:
            return False, f"High system memory pressure: {sys_usage:.1%} used"
            
        return False, "Memory usage normal"
        
    def optimize_memory_usage(self):
        """Optimize memory usage through various techniques."""
        if self.device == "cuda":
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
        # Run garbage collection
        gc.collect()
        
        # Suggest to OS to release memory
        if hasattr(os, 'malloc_trim'):  # Linux only
            os.malloc_trim(0)
            
    def wait_for_memory_release(self, timeout: int = 30):
        """
        Wait for memory pressure to decrease.
        
        Args:
            timeout: Maximum seconds to wait
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            is_critical, _ = self.check_memory_pressure()
            if not is_critical:
                return
            self.optimize_memory_usage()
            time.sleep(1)
