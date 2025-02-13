"""
Prompt generator using Ollama for local inference.
Incorporates temporal context (time of day, day of week, and holidays)
for more contextually aware prompts.
"""
import json
import os
from typing import Optional
import time

from ..utils.error_handler import handle_errors, PromptError
from ..utils.config import Config
from ..utils.metrics import GenerationMetrics
from ..plugins import get_temporal_descriptor

class PromptGenerator:
    def __init__(self, config: Config):
        """Initialize prompt generator with configuration."""
        self.config = config
        self.model_name = config.model.ollama_model
        self.example_prompts = [
            "Morning with clear sky, 18°C: Cozy cafe scene, steam rising from coffee cups mingles with warm sunbeams, readers bask in gentle morning light.",
            "Afternoon with light rain, 22°C: Market stalls protected by iridescent force fields, shoppers in flowing raincoats, puddles reflecting neon signs.",
            "Evening with strong winds, 15°C: Enchanted post office, letters swirling in wind currents, magical lanterns swaying, autumn leaves dancing."
        ]
        self.conversation_history = []
        
    @handle_errors(error_type=PromptError, retries=2)
    async def generate_prompt(self) -> str:
        """Generate a 60 word image prompt using Ollama with conversation context."""
        try:
            import ollama
            
            metrics = GenerationMetrics(model_name=self.model_name)
            start_time = time.time()
            
            # Get temporal context
            temporal_context = get_temporal_descriptor()
            
            # Build system context
            system_context = "\n".join([
                "You are a creative prompt generator for image generation.",
                "Generate unique and imaginative prompts that would inspire beautiful AI-generated images.",
                "IMPORTANT: Prompts MUST be concise and fit within 77 tokens (approximately 60 words).",
                "IMPORTANT: Do not have a preamble or explain the prompt, output ONLY the prompt itself.",
                "Focus on vivid, impactful descriptions using fewer, carefully chosen words.",
                f"\nCurrent temporal context: {temporal_context}",
                # If temporal context contains meme formatting, provide meme-specific guidance
                "If the context includes meme formatting, create a scene that emphasizes the text placement and style,",
                "making sure the text is clearly visible and follows classic meme aesthetics.",
                "For non-meme contexts, begin with the temporal context and incorporate weather naturally.",
                "Let the context influence the mood, lighting, and atmosphere of the scene.",
                "Example format: '[context]: [concise scene description]'",
                "Keep the final combined prompt (including context) within the 77 token limit."
            ])
            
            # Initialize conversation if empty
            if not self.conversation_history:
                self.conversation_history = [{
                    "role": "system",
                    "content": system_context
                }, {
                    "role": "user",
                    "content": "\n".join([
                        "Here are some example prompts:",
                        *[f"Example {i+1}: {prompt}" for i, prompt in enumerate(self.example_prompts)],
                        "\nGenerate a new prompt that is different from these examples but equally creative."
                    ])
                }]
            
            # Generate prompt
            response = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                options={"temperature": self.config.model.ollama_temperature}
            )
            
            # Add new prompt to conversation history
            new_prompt = response.message.content.strip()
            self.conversation_history.append({
                "role": "assistant",
                "content": new_prompt
            })
            self.conversation_history.append({
                "role": "user",
                "content": "Generate another unique prompt, different from previous ones."
            })
            
            # Update metrics
            metrics.generation_time = time.time() - start_time
            metrics.prompt = new_prompt
            metrics.prompt_tokens = len(new_prompt.split())
            
            return new_prompt
            
        except ImportError:
            raise PromptError("Please install ollama-python: pip install ollama")
        except Exception as e:
            raise PromptError(f"Error generating prompt: {str(e)}")
    
    @handle_errors(error_type=PromptError, retries=1)
    async def get_prompt_with_feedback(self) -> str:
        """Interactive prompt generation with user feedback."""
        while True:
            prompt = await self.generate_prompt()
            print("\nGenerated prompt:")
            print("-" * 80)
            print(prompt)
            print("-" * 80)
            
            choice = input("\nOptions:\n1. Use this prompt\n2. Generate new prompt\n3. Edit this prompt\nChoice (1-3): ")
            
            if choice == "1":
                return prompt
            elif choice == "2":
                continue
            elif choice == "3":
                edited = input("\nEdit the prompt:\n")
                return edited.strip()
            else:
                print("Invalid choice, please try again.")
    
    def cleanup(self):
        """Clean up resources."""
        self.conversation_history = []
