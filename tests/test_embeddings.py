"""
Tests for CLIP embedding utilities.
"""
import pytest
import torch
from src.utils.embeddings import check_prompt_length, get_pipeline_embeds

class MockTokenizer:
    def __init__(self, model_max_length=77):
        self.model_max_length = model_max_length
        
    def __call__(self, text, **kwargs):
        # Simulate tokenization by counting words and adding some padding
        words = text.split()
        # Simulate each word taking ~1.5 tokens on average
        token_count = int(len(words) * 1.5)
        
        # Handle padding if max_length is provided
        if 'max_length' in kwargs:
            max_length = kwargs['max_length']
            if kwargs.get('padding') == 'max_length':
                token_count = max_length
                
        return type('Tokens', (), {
            'input_ids': torch.zeros((1, token_count))
        })

class MockPipeline:
    def __init__(self):
        self.tokenizer = MockTokenizer()
        # Return both hidden states and pooled output
        self.text_encoder = lambda x: (
            torch.zeros((1, x.shape[1], 768)),  # hidden states
            torch.zeros((1, 768))  # pooled output
        )

def test_check_prompt_length():
    tokenizer = MockTokenizer()
    
    # Test short prompt
    short_prompt = "A simple test prompt"
    assert not check_prompt_length(tokenizer, short_prompt)
    
    # Test long prompt
    long_prompt = "A " + "very " * 100 + "long prompt"
    assert check_prompt_length(tokenizer, long_prompt)

def test_get_pipeline_embeds():
    pipeline = MockPipeline()
    device = "cpu"  # Use CPU for testing
    
    # Test with short prompts
    prompt = "A test prompt"
    negative_prompt = "A negative prompt"
    
    prompt_embeds, neg_embeds, pooled_prompt, pooled_neg = get_pipeline_embeds(
        pipeline,
        prompt,
        negative_prompt,
        device
    )
    
    # Check that embeddings were generated
    assert isinstance(prompt_embeds, torch.Tensor)
    assert isinstance(neg_embeds, torch.Tensor)
    assert isinstance(pooled_prompt, torch.Tensor)
    assert isinstance(pooled_neg, torch.Tensor)
    
    # Test with longer prompts
    long_prompt = "A " + "very " * 100 + "long prompt"
    long_neg = "A " + "very " * 50 + "long negative prompt"
    
    prompt_embeds, neg_embeds, pooled_prompt, pooled_neg = get_pipeline_embeds(
        pipeline,
        long_prompt,
        long_neg,
        device
    )
    
    # Check that embeddings were generated for long prompts
    assert isinstance(prompt_embeds, torch.Tensor)
    assert isinstance(neg_embeds, torch.Tensor)
    assert isinstance(pooled_prompt, torch.Tensor)
    assert isinstance(pooled_neg, torch.Tensor)
    
    # Verify shapes match between positive and negative embeddings
    assert prompt_embeds.shape == neg_embeds.shape
    assert pooled_prompt.shape == pooled_neg.shape
