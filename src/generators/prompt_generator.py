"""
Prompt generator using Ollama for local inference.
Incorporates temporal context (time of day, day of week, and holidays)
for more contextually aware prompts.
"""
import json
import os
from typing import Optional

from ..plugins import get_temporal_descriptor

class PromptGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'phi4:latest')
        self.example_prompts = [
            "Cozy cafe: Steam from coffee cups, readers in corners, frost patterns on windows cast golden morning light, prismatic reflections dance.",
            "Futuristic market: Holographic stalls mix with traditional ones, sci-fi foods under crystal dome, rainbow light filters through.",
            "Magical post office: Elves sort letters on floating belts, mechanical reindeer power machines, fiber-optic antlers glow."
        ]
        self.conversation_history = []
        
    async def generate_prompt(self) -> str:
        """Generate a 60 word image prompt using Ollama with conversation context."""
        try:
            import ollama
            
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
                "Begin the prompt with this temporal context, then add a concise but vivid scene description.",
                "Example format: '[temporal/style context]: [concise scene description]'",
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
            
            # Get temperature from environment with default
            temperature = float(os.getenv('OLLAMA_TEMPERATURE', 0.7))
            
            # Generate prompt
            response = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                options={"temperature": temperature}
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
            
            return new_prompt
            
        except ImportError:
            raise ImportError("Please install ollama-python: pip install ollama")
        except Exception as e:
            raise Exception(f"Error generating prompt: {str(e)}")
    
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
