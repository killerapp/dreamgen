"""
Utilities for handling text embeddings for long prompts.
"""
import re
import torch
from typing import Union, List, Tuple

def split_prompt(prompt: str, chunk_size: int = 77) -> List[str]:
    """Split a long prompt into semantically meaningful chunks.
    
    Args:
        prompt: The prompt text to split
        chunk_size: Maximum number of tokens per chunk
        
    Returns:
        List[str]: List of prompt chunks
    """
    # Split on common delimiters while preserving meaningful phrases
    chunks = []
    current_chunk = []
    words = prompt.split()
    current_length = 0
    
    for word in words:
        # Rough estimate: each word is ~1-2 tokens
        word_length = len(word.split()) + 1  # +1 for space
        if current_length + word_length > chunk_size:
            if current_chunk:  # Only add non-empty chunks
                chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:  # Add the last chunk
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_flux_embeddings(
    pipe,
    prompt: Union[str, List[str]],
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get text embeddings for Flux model, handling long prompts.
    
    Args:
        pipe: The Flux pipeline instance
        prompt: The prompt text or list of prompts
        device: The device to use for processing
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing 
        (clip_embeddings, t5_embeddings)
    """
    # Split prompt into CLIP-sized chunks
    if isinstance(prompt, str):
        prompt_chunks = split_prompt(prompt)
    else:
        prompt_chunks = prompt
        
    # Process with CLIP encoder
    clip_embeds = []
    for chunk in prompt_chunks:
        text_inputs = pipe.tokenizer(
            chunk,
            padding="max_length",
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        with torch.no_grad():
            clip_embed = pipe.text_encoder(**text_inputs)[0]
            clip_embeds.append(clip_embed)
    
    # Average CLIP embeddings
    clip_embeddings = torch.cat(clip_embeds).mean(dim=0, keepdim=True)
    
    # Process full prompt with T5 encoder
    text_inputs_2 = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,  # T5's max length
        truncation=True,
        return_tensors="pt",
    ).to(device)
    
    with torch.no_grad():
        t5_embeddings = pipe.text_encoder_2(**text_inputs_2)[0]
    
    return clip_embeddings, t5_embeddings
