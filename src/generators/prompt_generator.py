"""
Prompt generator using Ollama for local inference.
"""
import json
import os
from typing import Optional

class PromptGenerator:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv('OLLAMA_MODEL', 'phi4:latest')
        self.example_prompts = [
            "A bioluminescent underwater library where jellies float between shelves of waterproof books, casting ethereal blue-green light on ancient texts. Marine archaeologists in vintage diving suits study scrolls while schools of translucent fish wind between the stacks.",
            "A cross-section view of a towering termite mound reimagined as a retrofuturistic apartment complex, with tiny robots instead of termites going about their daily routines. Each chamber shows a different aspect of their mechanical society, from power generation to data processing.",
            "An impossible MC Escher-style train station during rush hour, where Victorian-era commuters walk on stairs that loop impossibly in multiple directions. The architecture blends Art Nouveau with mathematical impossibilities, while steam from locomotives rises in golden fractals."
        ]
        self.conversation_history = []
        
    async def generate_prompt(self) -> str:
        """Generate a 60 word image prompt using Ollama with conversation context."""
        try:
            import ollama
            
            # Build system context
            system_context = "\n".join([
                "You are a creative prompt generator for image generation.",
                "Generate unique and imaginative prompts that would inspire beautiful AI-generated images.",
                "Each prompt should be distinct from previous ones while maintaining high quality.",
                "Prompts should be detailed and descriptive, painting a vivid picture."
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
