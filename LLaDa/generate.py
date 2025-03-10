import torch
import numpy as np
import torch.nn.functional as F
import sys
import os
import re

# Add path to import from Structured_MDM
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Structured_MDM'))
from structure import StructuredGeneration

from transformers import AutoTokenizer, AutoModel


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


@ torch.no_grad()
def generate_structured(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
                       cfg_scale=0., remasking='low_confidence', mask_id=126336,
                       section_start_token_id=None, section_end_token_id=None,
                       max_section_length=None, min_section_length=None,
                       regex_patterns=None, tokenizer=None, section_names=None):
    '''
    Structured generation for MDM using the StructuredGeneration class.
    
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The token id of [MASK] is 126336.
        section_start_token_id: Token ID for section start marker.
        section_end_token_id: Token ID for section end marker.
        max_section_length: Maximum tokens per section.
        min_section_length: Minimum tokens per section.
        regex_patterns: Dictionary mapping section names to regex patterns.
        tokenizer: Tokenizer for converting between tokens and text.
        section_names: List of section names corresponding to section_markers.
    '''
    # Initialize structured generation
    structured_gen = StructuredGeneration(
        section_start_token_id=section_start_token_id,
        section_end_token_id=section_end_token_id,
        mask_token_id=mask_id,
        max_section_length=max_section_length,
        min_section_length=min_section_length,
        regex_patterns=regex_patterns,
        tokenizer=tokenizer
    )
    
    # Prepare input for structured generation
    x, mask_index = structured_gen.prepare_for_structured_generation(prompt, gen_length)
    x = x.to(model.device)
    
    # Detect section markers for structured generation
    section_markers = structured_gen.detect_section_markers(prompt)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        
        # Create mask for current block
        block_mask_index = mask_index.clone()
        block_mask_index[:, :block_start] = False
        block_mask_index[:, block_end:] = False
        
        # Get number of transfer tokens for this block
        block_section_markers = [(max(start, block_start), min(end, block_end)) 
                              for start, end in section_markers 
                              if start < block_end and end > block_start]
        
        num_transfer_tokens = structured_gen.get_structured_num_transfer_tokens(
            block_mask_index, steps, block_section_markers)
        
        for i in range(steps):
            current_mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                prompt_index = torch.zeros_like(x, dtype=torch.bool)
                prompt_index[:, :prompt.shape[1]] = True
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x).logits

            # Apply regex constraints if provided
            if regex_patterns is not None and section_names is not None:
                for pos in range(x.size(1)):
                    if current_mask_index[0, pos]:
                        logits[:, pos, :] = structured_gen.constrain_generation_with_regex(
                            logits[:, pos, :], x, section_markers, section_names, pos)

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits.to(torch.float64), dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            # Don't generate beyond current block
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(current_mask_index, x0, x)
            confidence = torch.where(current_mask_index, x0_p, -np.inf)

            # Apply structured confidence remasking
            x = structured_gen.apply_structured_confidence_remasking(
                x, x0, current_mask_index, confidence, num_transfer_tokens, i, 
                block_section_markers, section_names)

    return x


@ torch.no_grad()
def generate_structured_autoregressive(model, prompt, tokenizer, gen_length=128, temperature=0.,
                                     cfg_scale=0., section_start_token_id=126338, section_end_token_id=126339,
                                     regex_patterns=None, section_names=None):
    '''
    Autoregressive structured generation for LLMs using regex pattern constraints.
    
    Args:
        model: Language model.
        prompt: A tensor of shape (1, L).
        tokenizer: Tokenizer for converting between tokens and text.
        gen_length: Generated answer length.
        temperature: Sampling temperature.
        cfg_scale: Classifier-free guidance scale.
        section_start_token_id: Token ID for section start marker.
        section_end_token_id: Token ID for section end marker.
        regex_patterns: Dictionary mapping section names to regex patterns.
        section_names: List of section names corresponding to section_markers.
    '''
    # Initialize structured generation
    structured_gen = StructuredGeneration(
        section_start_token_id=section_start_token_id,
        section_end_token_id=section_end_token_id,
        regex_patterns=regex_patterns,
        tokenizer=tokenizer
    )
    
    # Detect section markers in the prompt
    section_markers = structured_gen.detect_section_markers(prompt)
    
    # Initialize generation
    x = prompt.clone()
    batch_size = x.size(0)
    
    # Generate tokens autoregressively
    for i in range(gen_length):
        # Get model predictions
        if cfg_scale > 0.:
            # Apply classifier-free guidance
            un_x = x.clone()
            prompt_index = torch.zeros_like(x, dtype=torch.bool)
            prompt_index[:, :prompt.shape[1]] = True
            un_x[prompt_index] = structured_gen.mask_token_id
            x_ = torch.cat([x, un_x], dim=0)
            outputs = model(x_)
            logits, un_logits = torch.chunk(outputs.logits, 2, dim=0)
            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        else:
            outputs = model(x)
            logits = outputs.logits
        
        # Get logits for the next token
        next_token_logits = logits[:, -1, :]
        
        # Apply regex constraints if provided
        if regex_patterns is not None and section_names is not None:
            current_pos = x.size(1)
            next_token_logits = structured_gen.constrain_generation_with_regex(
                next_token_logits, x, section_markers, section_names, current_pos)
        
        # Apply temperature
        if temperature > 0:
            next_token_logits = next_token_logits / temperature
        
        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append next token to the sequence
        x = torch.cat([x, next_token], dim=1)
        
        # Check for end of generation (e.g., EOS token)
        if (next_token == tokenizer.eos_token_id).all():
            break
    
    return x


def main():
    device = 'cuda'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # Example of structured prompt with section markers
    # Note: You would need to define actual token IDs for section markers in your tokenizer
    section_start_token_id = 126338  # Example token ID for section start
    section_end_token_id = 126339   # Example token ID for section end
    
    # Example regex patterns for structured generation
    regex_patterns = {
        "number": r"\d+",
        "calculation": r"[\d\+\-\*\/\(\)\s]+",
        "answer": r"\d+(\.\d+)?"
    }
    
    # Section names corresponding to the sections in the prompt
    section_names = ["calculation", "answer"]

    # Standard prompt
    standard_prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": standard_prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    # Standard generation
    print("Running standard generation...")
    out_standard = generate(model, input_ids, steps=128, gen_length=128, block_length=32, 
                          temperature=0., cfg_scale=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out_standard[:, input_ids.shape[1]:], skip_special_tokens=True)[0])

    # Structured generation with MDM
    print("\nRunning structured MDM generation...")
    out_structured = generate_structured(
        model, input_ids, steps=128, gen_length=128, block_length=32,
        temperature=0., cfg_scale=0., remasking='low_confidence',
        section_start_token_id=section_start_token_id,
        section_end_token_id=section_end_token_id,
        max_section_length=64, min_section_length=8,
        regex_patterns=regex_patterns,
        tokenizer=tokenizer,
        section_names=section_names
    )
    print(tokenizer.batch_decode(out_structured[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
    
    # Structured autoregressive generation
    print("\nRunning structured autoregressive generation...")
    out_autoregressive = generate_structured_autoregressive(
        model, input_ids, tokenizer, gen_length=128,
        temperature=0.7, cfg_scale=0.,
        section_start_token_id=section_start_token_id,
        section_end_token_id=section_end_token_id,
        regex_patterns=regex_patterns,
        section_names=section_names
    )
    print(tokenizer.batch_decode(out_autoregressive[:, input_ids.shape[1]:], skip_special_tokens=True)[0])


if __name__ == '__main__':
    main()