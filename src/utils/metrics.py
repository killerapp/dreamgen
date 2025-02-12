"""
Metrics collection and monitoring for the image generation system.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List, Optional, Union

@dataclass
class GenerationMetrics:
    """Metrics for a single image generation."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    prompt: str = ""
    model_name: str = ""
    generation_time: float = 0.0
    prompt_tokens: int = 0
    gpu_memory_peak: float = 0.0
    success: bool = True
    error: Optional[str] = None

@dataclass
class BatchMetrics:
    """Metrics for a batch of generations."""
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    total_images: int = 0
    successful_images: int = 0
    failed_images: int = 0
    total_time: float = 0.0
    average_time: float = 0.0
    generations: List[GenerationMetrics] = field(default_factory=list)

class MetricsCollector:
    def __init__(self, metrics_dir: Path):
        """Initialize metrics collector."""
        self.metrics_dir = metrics_dir
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.current_batch: Optional[BatchMetrics] = None
        
    def start_batch(self):
        """Start collecting metrics for a new batch."""
        self.current_batch = BatchMetrics()
        
    def end_batch(self):
        """End the current batch and save metrics."""
        if self.current_batch:
            self.current_batch.end_time = datetime.now().isoformat()
            if self.current_batch.total_images > 0:
                self.current_batch.average_time = (
                    self.current_batch.total_time / self.current_batch.total_images
                )
            self._save_batch_metrics()
            self.current_batch = None
            
    def add_generation(self, metrics: GenerationMetrics):
        """Add metrics for a single generation to the current batch."""
        if not self.current_batch:
            self.start_batch()
            
        self.current_batch.total_images += 1
        if metrics.success:
            self.current_batch.successful_images += 1
        else:
            self.current_batch.failed_images += 1
            
        self.current_batch.total_time += metrics.generation_time
        self.current_batch.generations.append(metrics)
        
    def _save_batch_metrics(self):
        """Save batch metrics to a JSON file."""
        if not self.current_batch:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"batch_metrics_{timestamp}.json"
        
        metrics_dict = {
            "start_time": self.current_batch.start_time,
            "end_time": self.current_batch.end_time,
            "total_images": self.current_batch.total_images,
            "successful_images": self.current_batch.successful_images,
            "failed_images": self.current_batch.failed_images,
            "total_time": self.current_batch.total_time,
            "average_time": self.current_batch.average_time,
            "generations": [
                {
                    "timestamp": g.timestamp,
                    "prompt": g.prompt,
                    "model_name": g.model_name,
                    "generation_time": g.generation_time,
                    "prompt_tokens": g.prompt_tokens,
                    "gpu_memory_peak": g.gpu_memory_peak,
                    "success": g.success,
                    "error": g.error
                }
                for g in self.current_batch.generations
            ]
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
            
    def get_summary(self) -> Dict[str, Union[int, float]]:
        """Get summary statistics for the current batch."""
        if not self.current_batch:
            return {}
            
        return {
            "total_images": self.current_batch.total_images,
            "successful_images": self.current_batch.successful_images,
            "failed_images": self.current_batch.failed_images,
            "average_time": self.current_batch.average_time,
            "total_time": self.current_batch.total_time
        }
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for the current batch."""
        if not self.current_batch or not self.current_batch.generations:
            return {}
            
        successful_gens = [g for g in self.current_batch.generations if g.success]
        if not successful_gens:
            return {}
            
        return {
            "avg_generation_time": sum(g.generation_time for g in successful_gens) / len(successful_gens),
            "avg_gpu_memory": sum(g.gpu_memory_peak for g in successful_gens) / len(successful_gens),
            "success_rate": len(successful_gens) / len(self.current_batch.generations)
        }
