"""Enhanced web interface with image rotation and system control."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from PIL import Image

from ..generators.image_generator import ImageGenerator
from ..generators.mock_image_generator import MockImageGenerator
from ..generators.prompt_generator import PromptGenerator
from ..plugins.plugin_manager import PluginManager
from ..utils.config import Config
from ..utils.storage import StorageManager


class EnhancedWebUI:
    """Enhanced web UI with image gallery, rotation display, and system controls."""

    def __init__(self, config: Config, mock: bool = False):
        """Initialize the enhanced web UI.
        
        Args:
            config: Application configuration
            mock: Use mock generators for testing
        """
        self.config = config
        self.mock = mock
        self.storage = StorageManager(str(config.system.output_dir))
        self.plugin_manager = PluginManager(config)
        self.generator_cls = MockImageGenerator if mock else ImageGenerator
        self.is_generating = False
        self.generation_task: Optional[asyncio.Task] = None
        self.generation_history: List[Dict[str, Any]] = []
        self.max_history = 100
        
    def get_recent_images(self, limit: int = 20) -> List[Tuple[str, str, str]]:
        """Get recent generated images from storage.
        
        Args:
            limit: Maximum number of images to return
            
        Returns:
            List of tuples (image_path, prompt, timestamp)
        """
        output_dir = Path(self.config.system.output_dir)
        images = []
        
        # Scan for recent images
        for img_file in sorted(output_dir.glob("**/*.png"), key=os.path.getmtime, reverse=True)[:limit]:
            # Try to find associated prompt file
            prompt_file = img_file.with_suffix(".txt")
            prompt = "No prompt available"
            if prompt_file.exists():
                try:
                    prompt = prompt_file.read_text().strip()
                except Exception:
                    pass
            
            timestamp = datetime.fromtimestamp(os.path.getmtime(img_file))
            images.append((str(img_file), prompt, timestamp.strftime("%Y-%m-%d %H:%M:%S")))
        
        return images
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics."""
        status = {
            "is_generating": self.is_generating,
            "mock_mode": self.mock,
            "total_generated": len(self.generation_history),
            "output_directory": str(self.config.system.output_dir),
            "active_plugins": [],
            "gpu_available": False,
            "models_loaded": False
        }
        
        # Get active plugins
        for plugin_name, plugin in self.plugin_manager.plugins.items():
            if self.plugin_manager.is_enabled(plugin_name):
                status["active_plugins"].append(plugin_name)
        
        # Check GPU availability
        try:
            import torch
            status["gpu_available"] = torch.cuda.is_available() or torch.backends.mps.is_available()
        except ImportError:
            pass
        
        # Check if models are loaded (mock always true)
        status["models_loaded"] = self.mock or Path(self.config.system.model_cache_dir).exists()
        
        return status
    
    async def generate_single_image(self, prompt: str, enhance_prompt: bool = True) -> Tuple[str, str, Dict]:
        """Generate a single image.
        
        Args:
            prompt: Text prompt for generation
            enhance_prompt: Whether to enhance prompt with plugins
            
        Returns:
            Tuple of (image_path, final_prompt, metadata)
        """
        generator = self.generator_cls(self.config)
        
        # Enhance prompt if requested
        final_prompt = prompt
        if enhance_prompt and not self.mock:
            try:
                prompt_gen = PromptGenerator(self.config, self.plugin_manager)
                final_prompt = await prompt_gen.generate_prompt(base_prompt=prompt)
            except Exception as e:
                print(f"Prompt enhancement failed: {e}")
        
        # Generate image
        output_path = self.storage.get_output_path(final_prompt)
        image_path, prompt_used, gen_time = await generator.generate_image(final_prompt, output_path)
        
        # Clean up
        generator.cleanup()
        
        # Record in history
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "generation_time": gen_time,
            "enhanced": enhance_prompt,
            "mock": self.mock
        }
        
        self.generation_history.append({
            "image": str(image_path),
            "prompt": prompt_used,
            "metadata": metadata
        })
        
        # Trim history if needed
        if len(self.generation_history) > self.max_history:
            self.generation_history = self.generation_history[-self.max_history:]
        
        return str(image_path), prompt_used, metadata
    
    async def continuous_generation_loop(
        self,
        base_prompt: str,
        interval_minutes: int,
        batch_size: int,
        enhance_prompts: bool
    ):
        """Run continuous generation loop.
        
        Args:
            base_prompt: Base prompt for variations
            interval_minutes: Minutes between generation batches
            batch_size: Number of images per batch
            enhance_prompts: Whether to enhance prompts
        """
        self.is_generating = True
        
        try:
            while self.is_generating:
                # Generate batch
                for i in range(batch_size):
                    if not self.is_generating:
                        break
                    
                    try:
                        await self.generate_single_image(base_prompt, enhance_prompts)
                    except Exception as e:
                        print(f"Generation error: {e}")
                
                # Wait for next batch
                if self.is_generating:
                    await asyncio.sleep(interval_minutes * 60)
        finally:
            self.is_generating = False
    
    def start_continuous_generation(
        self,
        base_prompt: str,
        interval: int,
        batch_size: int,
        enhance: bool
    ) -> str:
        """Start continuous generation in background.
        
        Args:
            base_prompt: Base prompt for generation
            interval: Minutes between batches
            batch_size: Images per batch
            enhance: Whether to enhance prompts
            
        Returns:
            Status message
        """
        if self.is_generating:
            return "⚠️ Generation already in progress"
        
        # Start generation task
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        self.generation_task = loop.create_task(
            self.continuous_generation_loop(base_prompt, interval, batch_size, enhance)
        )
        
        return f"✅ Started continuous generation: {batch_size} images every {interval} minutes"
    
    def stop_continuous_generation(self) -> str:
        """Stop continuous generation.
        
        Returns:
            Status message
        """
        if not self.is_generating:
            return "ℹ️ No generation in progress"
        
        self.is_generating = False
        
        if self.generation_task:
            self.generation_task.cancel()
        
        return "🛑 Stopped continuous generation"
    
    def toggle_plugin(self, plugin_name: str, enabled: bool) -> str:
        """Toggle a plugin on/off.
        
        Args:
            plugin_name: Name of the plugin
            enabled: Whether to enable or disable
            
        Returns:
            Status message
        """
        if enabled:
            self.plugin_manager.enable_plugin(plugin_name)
            return f"✅ Enabled {plugin_name}"
        else:
            self.plugin_manager.disable_plugin(plugin_name)
            return f"❌ Disabled {plugin_name}"
    
    def export_history(self) -> str:
        """Export generation history as JSON.
        
        Returns:
            JSON string of history
        """
        return json.dumps(self.generation_history, indent=2)
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface.
        
        Returns:
            Configured Gradio Blocks interface
        """
        with gr.Blocks(title="Continuous Image Generator - CSO Module", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎨 Continuous Image Generator")
            gr.Markdown("AI-powered image generation system with plugin architecture")
            
            with gr.Tabs():
                # Single Generation Tab
                with gr.Tab("Generate"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            prompt_input = gr.Textbox(
                                label="Prompt",
                                placeholder="Enter your image prompt...",
                                lines=3
                            )
                            enhance_checkbox = gr.Checkbox(
                                label="Enhance prompt with plugins",
                                value=True
                            )
                            generate_btn = gr.Button("Generate Image", variant="primary")
                        
                        with gr.Column(scale=3):
                            output_image = gr.Image(label="Generated Image")
                            output_prompt = gr.Textbox(label="Final Prompt Used", lines=2)
                            generation_info = gr.JSON(label="Generation Metadata")
                
                # Gallery Tab
                with gr.Tab("Gallery"):
                    refresh_gallery_btn = gr.Button("Refresh Gallery")
                    gallery = gr.Gallery(
                        label="Recent Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=4,
                        rows=2,
                        height="auto"
                    )
                    gallery_info = gr.Dataframe(
                        headers=["Image", "Prompt", "Timestamp"],
                        label="Image Details"
                    )
                
                # Continuous Generation Tab
                with gr.Tab("Continuous Mode"):
                    with gr.Row():
                        with gr.Column():
                            continuous_prompt = gr.Textbox(
                                label="Base Prompt",
                                placeholder="Base prompt for variations...",
                                lines=2
                            )
                            interval_slider = gr.Slider(
                                minimum=1,
                                maximum=60,
                                value=5,
                                step=1,
                                label="Interval (minutes)"
                            )
                            batch_slider = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=3,
                                step=1,
                                label="Batch Size"
                            )
                            continuous_enhance = gr.Checkbox(
                                label="Enhance prompts",
                                value=True
                            )
                        
                        with gr.Column():
                            start_btn = gr.Button("Start Generation", variant="primary")
                            stop_btn = gr.Button("Stop Generation", variant="stop")
                            continuous_status = gr.Textbox(
                                label="Status",
                                value="Not running",
                                interactive=False
                            )
                
                # System Control Tab
                with gr.Tab("System Control"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### Plugin Management")
                            plugin_states = {}
                            for plugin_name in self.plugin_manager.plugins:
                                plugin_states[plugin_name] = gr.Checkbox(
                                    label=plugin_name,
                                    value=self.plugin_manager.is_enabled(plugin_name)
                                )
                            
                            gr.Markdown("### System Status")
                            status_display = gr.JSON(label="Current Status")
                            refresh_status_btn = gr.Button("Refresh Status")
                        
                        with gr.Column():
                            gr.Markdown("### Export & Analytics")
                            export_btn = gr.Button("Export History")
                            history_export = gr.Textbox(
                                label="Generation History (JSON)",
                                lines=10,
                                max_lines=20
                            )
                            
                            stats_display = gr.Dataframe(
                                headers=["Metric", "Value"],
                                label="Statistics"
                            )
            
            # Event handlers
            def generate_wrapper(prompt, enhance):
                """Wrapper for single image generation."""
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                img_path, final_prompt, metadata = loop.run_until_complete(
                    self.generate_single_image(prompt, enhance)
                )
                return img_path, final_prompt, metadata
            
            def refresh_gallery():
                """Refresh the image gallery."""
                images = self.get_recent_images()
                image_paths = [img[0] for img in images]
                details = [[img[0].split("/")[-1], img[1][:50] + "...", img[2]] for img in images]
                return image_paths, details
            
            def get_statistics():
                """Get system statistics."""
                status = self.get_system_status()
                stats = [
                    ["Total Generated", status["total_generated"]],
                    ["GPU Available", "Yes" if status["gpu_available"] else "No"],
                    ["Models Loaded", "Yes" if status["models_loaded"] else "No"],
                    ["Active Plugins", len(status["active_plugins"])],
                    ["Mock Mode", "Yes" if status["mock_mode"] else "No"]
                ]
                return stats
            
            # Connect event handlers
            generate_btn.click(
                generate_wrapper,
                inputs=[prompt_input, enhance_checkbox],
                outputs=[output_image, output_prompt, generation_info]
            )
            
            refresh_gallery_btn.click(
                refresh_gallery,
                outputs=[gallery, gallery_info]
            )
            
            start_btn.click(
                self.start_continuous_generation,
                inputs=[continuous_prompt, interval_slider, batch_slider, continuous_enhance],
                outputs=continuous_status
            )
            
            stop_btn.click(
                self.stop_continuous_generation,
                outputs=continuous_status
            )
            
            refresh_status_btn.click(
                lambda: (self.get_system_status(), get_statistics()),
                outputs=[status_display, stats_display]
            )
            
            export_btn.click(
                self.export_history,
                outputs=history_export
            )
            
            # Plugin toggles
            for plugin_name, checkbox in plugin_states.items():
                checkbox.change(
                    lambda enabled, name=plugin_name: self.toggle_plugin(name, enabled),
                    inputs=checkbox
                )
            
            # Load initial data
            interface.load(
                lambda: (refresh_gallery(), self.get_system_status(), get_statistics()),
                outputs=[
                    [gallery, gallery_info],
                    status_display,
                    stats_display
                ]
            )
        
        return interface


def launch_enhanced_ui(config: Config, mock: bool = False, share: bool = False, port: int = 7860):
    """Launch the enhanced web UI.
    
    Args:
        config: Application configuration
        mock: Use mock generators
        share: Create public Gradio link
        port: Port to run on
    """
    ui = EnhancedWebUI(config, mock)
    interface = ui.create_interface()
    interface.launch(share=share, server_port=port, server_name="0.0.0.0")