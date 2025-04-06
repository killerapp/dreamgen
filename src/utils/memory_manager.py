"""
Memory management utilities for GPU operations.
"""
import os
import gc
import time
import platform
import subprocess
from typing import Optional, Literal, Tuple
import psutil
import torch
import logging

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, device: Literal["cpu", "cuda", "mps"] = "cuda"):
        self.device = device
        self.warning_threshold = 0.8  # 80% memory usage warning
        self.critical_threshold = 0.9  # 90% memory usage critical
        self.is_apple_silicon = platform.processor() == 'arm' and platform.system() == 'Darwin'
        
    def get_gpu_memory_info(self) -> tuple[float, float, float]:
        """Get current GPU memory usage information."""
        # For CUDA devices
        if self.device == "cuda" and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return allocated, reserved, total
            
        # For MPS devices (Apple Silicon)
        # MPS doesn't have built-in memory tracking like CUDA
        if self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                # Get more accurate memory info for Apple Silicon
                # Using Activity Monitor's data via command line
                if platform.system() == 'Darwin':
                    # Get total GPU memory from system_profiler
                    total_gpu_mem = self._get_apple_gpu_total_memory()
                    
                    # Get current process memory usage
                    process = psutil.Process(os.getpid())
                    process_memory = process.memory_info().rss / 1024**3
                    
                    # Get memory pressure info
                    mem_pressure = self._get_memory_pressure()
                    
                    # Calculate an approximation based on memory pressure and process memory
                    # This is more accurate than just using process memory
                    if mem_pressure > 0.8:  # High memory pressure
                        allocated = process_memory * 1.5  # Assume more is allocated than RSS shows
                    else:
                        allocated = process_memory * 1.2  # Normal case
                        
                    # For reserved, use slightly higher than allocated
                    reserved = allocated * 1.2
                    
                    logger.debug(f"MPS memory: allocated={allocated:.2f}GB, total={total_gpu_mem:.2f}GB, pressure={mem_pressure:.2f}")
                    return allocated, reserved, total_gpu_mem
                    
            except Exception as e:
                logger.warning(f"Failed to get detailed Apple GPU memory info: {e}")
                
            # Fallback to basic estimation using process memory
            vm = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_memory = process.memory_info().rss / 1024**3
            
            # For Apple Silicon fallback, estimate GPU portion of unified memory
            return process_memory, process_memory, vm.total / 1024**3 * 0.7  # Assume 70% of system memory available to GPU
            
        return 0.0, 0.0, 0.0
        
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
        if self.device in ["cuda", "mps"]:
            allocated, reserved, total = self.get_gpu_memory_info()
            if total > 0:  # Ensure we have valid GPU memory info
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
        
    def _get_apple_gpu_total_memory(self) -> float:
        """Get the total memory of Apple GPU."""
        if not self.is_apple_silicon:
            return 0.0
            
        try:
            # Use system_profiler to get GPU info
            cmd = ["system_profiler", "SPDisplaysDataType"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse the output to find memory information
            for line in result.stdout.split("\n"):
                if "VRAM" in line or "Total VRAM" in line:
                    # Extract the number and convert to GB
                    parts = line.split(":")
                    if len(parts) >= 2:
                        mem_str = parts[1].strip()
                        if "GB" in mem_str:
                            return float(mem_str.replace("GB", "").strip())
                        elif "MB" in mem_str:
                            return float(mem_str.replace("MB", "").strip()) / 1024
            
            # If we couldn't find VRAM info, estimate based on device model
            # M1: 8GB or 16GB, M1 Pro/Max: 16-64GB, M2: 8-24GB
            # Default to a conservative estimate
            return 8.0  # Assume at least 8GB unified memory for GPU
            
        except Exception as e:
            logger.warning(f"Error getting Apple GPU memory: {str(e)}")
            # Return a reasonable default
            return 8.0
            
    def _get_memory_pressure(self) -> float:
        """Get memory pressure on macOS (0.0-1.0)."""
        if platform.system() != 'Darwin':
            return 0.0
            
        try:
            cmd = ["memory_pressure"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse the output
            pressure = 0.0
            for line in result.stdout.split("\n"):
                if "System-wide memory pressure" in line:
                    if "normal" in line.lower():
                        pressure = 0.3
                    elif "warning" in line.lower():
                        pressure = 0.7
                    elif "critical" in line.lower():
                        pressure = 0.9
                    break
                    
            return pressure
        except Exception:
            # If command fails, fall back to psutil
            vm = psutil.virtual_memory()
            return vm.percent / 100.0

    def optimize_memory_usage(self):
        """Optimize memory usage through various techniques."""
        if self.device == "cuda":
            # Clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        elif self.device == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS doesn't have explicit cache clearing functions like CUDA
            # But we can implement some MPS-specific optimizations
            
            # For Apple Silicon, we can force garbage collection more aggressively
            for _ in range(3):
                gc.collect()
                
            # On macOS, we can request memory pressure relief
            if platform.system() == 'Darwin':
                try:
                    # Use mach_vm_pressure_monitor if available
                    subprocess.run(["sudo", "purge"], capture_output=True, timeout=5)
                except (subprocess.SubprocessError, OSError, FileNotFoundError):
                    # If the command fails, we can't do much else for MPS memory
                    pass
            
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
