"""
Utilities for handling text embeddings for long prompts.
"""
import re
import torch
from typing import Union, List, Tuple

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    This is a more accurate estimation than simply counting words.
    
    Args:
        text: The text to estimate tokens for
        
    Returns:
        int: Estimated token count
    """
    # Common token patterns and their counts
    # This is a simplification but more accurate than word count
    
    # 1. Count normal words (roughly 1 token each)
    words = len(re.findall(r'\b\w+\b', text))
    
    # 2. Count special characters (roughly 1 token each)
    special_chars = len(re.findall(r'[^\w\s]', text))
    
    # 3. Count numbers (roughly 1 token per 2-3 digits)
    numbers = len(re.findall(r'\d+', text))
    digits = sum(len(match) for match in re.findall(r'\d+', text))
    number_tokens = max(numbers, digits // 3)
    
    # 4. Account for uncommon words/specialized tokens (longer words often split into subtokens)
    long_words = len(re.findall(r'\b\w{10,}\b', text))
    long_word_overhead = long_words  # Each long word might cost +1 token
    
    # Final estimate
    total = words + special_chars + number_tokens + long_word_overhead
    
    # Apply a small safety factor
    return int(total * 1.1)

def split_prompt(prompt: str, chunk_size: int = 77) -> List[str]:
    """Split a long prompt into semantically meaningful chunks.
    
    Args:
        prompt: The prompt text to split
        chunk_size: Maximum number of tokens per chunk
        
    Returns:
        List[str]: List of prompt chunks
    """
    # Try to split on natural boundaries first
    natural_splits = re.split(r'(?<=[.!?;])\s+', prompt)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for split in natural_splits:
        # If this natural split is already too long, we'll need to break it further
        if estimate_tokens(split) > chunk_size:
            # Process and add any accumulated chunks before handling this long one
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Process the long split by words
            words = split.split()
            for word in words:
                # Estimate tokens more accurately
                word_token_count = estimate_tokens(word + " ")
                
                if current_length + word_token_count > chunk_size:
                    if current_chunk:  # Only add non-empty chunks
                        chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = word_token_count
                else:
                    current_chunk.append(word)
                    current_length += word_token_count
        else:
            # This natural split fits within the chunk size
            estimated_tokens = estimate_tokens(split + " ")
            
            if current_length + estimated_tokens > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [split]
                current_length = estimated_tokens
            else:
                current_chunk.append(split)
                current_length += estimated_tokens
    
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
