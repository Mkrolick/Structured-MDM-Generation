import torch
import torch.nn.functional as F
import numpy as np
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Pattern


@dataclass
class StructuredGeneration:
    """
    Implementation of structured generation for Multimodal Diffusion Models (MDM).
    This allows for generating content in a structured manner, controlling the generation
    process through specified sections or formats.
    """
    # Structure token ids
    section_start_token_id: int = 126338  # Example token ID for section start
    section_end_token_id: int = 126339  # Example token ID for section end
    mask_token_id: int = 126336  # Default mask token ID
    
    # Generation parameters
    max_section_length: int = 32  # Maximum tokens per section
    min_section_length: int = 4  # Minimum tokens per section
    
    # Regex pattern constraints
    regex_patterns: Dict[str, str] = field(default_factory=dict)  # Section name to regex pattern mapping
    compiled_patterns: Dict[str, Pattern] = field(default_factory=dict)  # Compiled regex patterns
    tokenizer: Optional[object] = None  # Tokenizer for converting between tokens and text
    
    def __init__(self, 
                 section_start_token_id: Optional[int] = None,
                 section_end_token_id: Optional[int] = None,
                 mask_token_id: Optional[int] = None,
                 max_section_length: Optional[int] = None,
                 min_section_length: Optional[int] = None,
                 regex_patterns: Optional[Dict[str, str]] = None,
                 tokenizer: Optional[object] = None):
        """
        Initialize structured generation with custom parameters if provided.
        
        Args:
            section_start_token_id: Token ID for section start marker
            section_end_token_id: Token ID for section end marker
            mask_token_id: Token ID for mask token
            max_section_length: Maximum tokens per section
            min_section_length: Minimum tokens per section
            regex_patterns: Dictionary mapping section names to regex patterns
            tokenizer: Tokenizer for converting between tokens and text
        """
        if section_start_token_id is not None:
            self.section_start_token_id = section_start_token_id
        if section_end_token_id is not None:
            self.section_end_token_id = section_end_token_id
        if mask_token_id is not None:
            self.mask_token_id = mask_token_id
        if max_section_length is not None:
            self.max_section_length = max_section_length
        if min_section_length is not None:
            self.min_section_length = min_section_length
        if regex_patterns is not None:
            self.regex_patterns = regex_patterns
            self.compiled_patterns = {
                name: re.compile(pattern) for name, pattern in regex_patterns.items()
            }
        if tokenizer is not None:
            self.tokenizer = tokenizer
    
    def create_section_masks(self, input_ids: torch.Tensor, section_markers: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Create masks for structured sections in the input sequence.
        
        Args:
            input_ids: Input token ids (batch_size, sequence_length)
            section_markers: List of (start_pos, end_pos) tuples for each section
            
        Returns:
            A tensor of masks where 1 indicates positions to be generated in the structured format
        """
        batch_size, seq_len = input_ids.size()
        masks = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
        
        for start_pos, end_pos in section_markers:
            if start_pos < seq_len and end_pos <= seq_len:
                masks[:, start_pos:end_pos] = True
                
        return masks
    
    def detect_section_markers(self, input_ids: torch.Tensor) -> List[Tuple[int, int]]:
        """
        Detect section markers in the input sequence based on special tokens.
        
        Args:
            input_ids: Input token ids (batch_size, sequence_length)
            
        Returns:
            List of (start_pos, end_pos) tuples for each section
        """
        sections = []
        batch_idx = 0  # Assuming a batch size of 1 for simplicity
        
        start_pos = None
        for pos in range(input_ids.size(1)):
            token = input_ids[batch_idx, pos].item()
            
            if token == self.section_start_token_id:
                start_pos = pos + 1  # Start after the marker
            elif token == self.section_end_token_id and start_pos is not None:
                sections.append((start_pos, pos))
                start_pos = None
        
        # If a section started but never closed, close it at the end
        if start_pos is not None:
            sections.append((start_pos, input_ids.size(1)))
            
        return sections
    
    def prepare_for_structured_generation(self, input_ids: torch.Tensor, gen_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare input for structured generation by adding masks based on detected sections.
        
        Args:
            input_ids: Input token ids (batch_size, sequence_length)
            gen_length: Length of generation
            
        Returns:
            (prepared_input, mask_index) where prepared_input has mask tokens in structured sections
        """
        batch_size, seq_len = input_ids.size()
        
        # Create space for generation with mask tokens
        extended_input = torch.full((batch_size, seq_len + gen_length), 
                                    self.mask_token_id, 
                                    dtype=input_ids.dtype, 
                                    device=input_ids.device)
        extended_input[:, :seq_len] = input_ids.clone()
        
        # Detect section markers in the original input
        section_markers = self.detect_section_markers(input_ids)
        
        # If no markers found, default to standard generation
        if not section_markers:
            mask_index = (extended_input == self.mask_token_id)
            return extended_input, mask_index
        
        # Create structured masks for the extended input
        # Adjust positions to account for the added generation space
        adjusted_markers = []
        marker_idx = 0
        gen_space_used = 0
        
        for start_pos, end_pos in section_markers:
            section_length = min(self.max_section_length, 
                                gen_length // len(section_markers) if len(section_markers) > 0 else gen_length)
            section_length = max(section_length, self.min_section_length)
            
            # Don't exceed available generation space
            if gen_space_used + section_length > gen_length:
                section_length = gen_length - gen_space_used
                if section_length <= 0:
                    break
            
            # Adjust for already generated content
            adjusted_start = seq_len + gen_space_used
            adjusted_end = adjusted_start + section_length
            adjusted_markers.append((adjusted_start, adjusted_end))
            
            gen_space_used += section_length
            marker_idx += 1
        
        # Create the mask for structured generation
        mask_index = self.create_section_masks(extended_input, adjusted_markers)
        
        return extended_input, mask_index
    
    def get_structured_num_transfer_tokens(self, mask_index: torch.Tensor, steps: int, 
                                          section_markers: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Calculate the number of tokens to transfer at each step, respecting the section structure.
        
        Args:
            mask_index: Boolean tensor indicating which positions are masked
            steps: Number of sampling steps
            section_markers: List of (start_pos, end_pos) tuples for each section
            
        Returns:
            Tensor of shape (batch_size, steps) indicating tokens to transfer per step
        """
        batch_size = mask_index.size(0)
        num_transfer_tokens = torch.zeros(batch_size, steps, device=mask_index.device, dtype=torch.int64)
        
        # If no sections, use standard transfer token computation
        if not section_markers:
            mask_num = mask_index.sum(dim=1, keepdim=True)
            base = mask_num // steps
            remainder = mask_num % steps
            
            num_transfer_tokens = torch.zeros(batch_size, steps, device=mask_index.device, dtype=torch.int64) + base
            
            for i in range(batch_size):
                num_transfer_tokens[i, :remainder[i]] += 1
                
            return num_transfer_tokens
        
        # Compute per section and then aggregate
        for start_pos, end_pos in section_markers:
            section_mask = mask_index.clone()
            # Zero out everything outside this section
            section_mask[:, :start_pos] = False
            section_mask[:, end_pos:] = False
            
            section_mask_num = section_mask.sum(dim=1, keepdim=True)
            section_base = section_mask_num // steps
            section_remainder = section_mask_num % steps
            
            section_transfers = torch.zeros(batch_size, steps, device=mask_index.device, dtype=torch.int64) + section_base
            
            for i in range(batch_size):
                section_transfers[i, :section_remainder[i]] += 1
            
            num_transfer_tokens += section_transfers
            
        return num_transfer_tokens
    
    def validate_against_regex(self, token_ids: torch.Tensor, section_name: str) -> torch.Tensor:
        """
        Validate a sequence of token IDs against a regex pattern for a specific section.
        
        Args:
            token_ids: Tensor of token IDs
            section_name: Name of the section with associated regex pattern
            
        Returns:
            Boolean tensor indicating whether the sequence matches the pattern
        """
        if self.tokenizer is None or section_name not in self.compiled_patterns:
            # If no tokenizer or pattern, everything is valid
            return torch.ones_like(token_ids, dtype=torch.bool)
        
        # Convert tokens to text
        text = self.tokenizer.decode(token_ids.tolist())
        
        # Check if text matches the pattern
        match = self.compiled_patterns[section_name].match(text)
        
        # Return True if matches, False otherwise
        return torch.tensor(match is not None, device=token_ids.device)
    
    def filter_logits_with_regex(self, logits: torch.Tensor, token_ids: torch.Tensor, 
                                section_name: str, position: int) -> torch.Tensor:
        """
        Filter logits based on regex pattern constraints for a specific section.
        
        Args:
            logits: Logits tensor (batch_size, vocab_size)
            token_ids: Current token IDs
            section_name: Name of the section with associated regex pattern
            position: Current position in the sequence
            
        Returns:
            Filtered logits tensor
        """
        if self.tokenizer is None or section_name not in self.compiled_patterns:
            return logits
        
        batch_size, vocab_size = logits.size()
        filtered_logits = logits.clone()
        
        # For each token in the vocabulary, check if adding it would still match the pattern
        for token_id in range(vocab_size):
            # Create a new sequence with the candidate token
            new_tokens = token_ids.clone()
            new_tokens[:, position] = token_id
            
            # Check if the new sequence matches the pattern
            valid = self.validate_against_regex(new_tokens, section_name)
            
            # If not valid, set logits to a very negative value
            if not valid:
                filtered_logits[:, token_id] = -float('inf')
        
        return filtered_logits
    
    def apply_structured_confidence_remasking(self, 
                                             x: torch.Tensor, 
                                             x0: torch.Tensor, 
                                             mask_index: torch.Tensor,
                                             confidence: torch.Tensor,
                                             num_transfer_tokens: torch.Tensor,
                                             step_idx: int,
                                             section_markers: List[Tuple[int, int]],
                                             section_names: Optional[List[str]] = None) -> torch.Tensor:
        """
        Apply confidence-based remasking that respects the section structure.
        
        Args:
            x: Current state of the sequence
            x0: Predicted x0 (token ids)
            mask_index: Boolean tensor indicating masked positions
            confidence: Confidence scores for each position
            num_transfer_tokens: Number of tokens to transfer at each step
            step_idx: Current step index
            section_markers: List of (start_pos, end_pos) tuples for each section
            section_names: Optional list of section names for regex validation
            
        Returns:
            Updated sequence with newly predicted tokens
        """
        batch_size = x.size(0)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        
        if not section_markers:
            # Standard confidence-based remasking
            for j in range(batch_size):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, step_idx])
                transfer_index[j, select_index] = True
        else:
            # Structured section-by-section remasking
            total_transfers_left = num_transfer_tokens[:, step_idx].clone()
            
            for i, (start_pos, end_pos) in enumerate(section_markers):
                # Skip if section is out of bounds
                if start_pos >= x.size(1) or start_pos >= end_pos:
                    continue
                    
                # Create a mask limited to this section
                section_mask = mask_index.clone()
                section_mask[:, :start_pos] = False
                section_mask[:, end_pos:] = False
                
                # Create a confidence tensor that only has values for this section
                section_confidence = confidence.clone()
                section_confidence[~section_mask] = -float('inf')
                
                # Calculate how many tokens to transfer in this section
                section_mask_count = section_mask.sum(dim=1)
                # Proportion of transfers based on section size relative to total masks
                total_mask_count = mask_index.sum(dim=1)
                prop_transfers = torch.where(
                    total_mask_count > 0,
                    section_mask_count.float() / torch.max(total_mask_count.float(), torch.ones_like(total_mask_count.float())),
                    torch.zeros_like(section_mask_count.float())
                )
                
                section_transfers = torch.max(
                    torch.floor(prop_transfers * total_transfers_left.float()).to(torch.int64),
                    torch.ones_like(total_transfers_left, dtype=torch.int64)  # At least 1 token per section if possible
                )
                
                # Don't transfer more than available
                section_transfers = torch.min(section_transfers, section_mask_count)
                
                # Don't transfer more than we have left for this step
                section_transfers = torch.min(section_transfers, total_transfers_left)
                
                # Update the remaining transfers
                total_transfers_left -= section_transfers
                
                # Apply regex validation if section names are provided
                if section_names is not None and i < len(section_names) and section_names[i] in self.regex_patterns:
                    # For each candidate token, check if it would satisfy the regex pattern
                    for j in range(batch_size):
                        if section_transfers[j] > 0:
                            # Get indices of top-k confidence tokens
                            _, candidate_indices = torch.topk(section_confidence[j], k=section_transfers[j] * 2)  # Get more candidates than needed
                            
                            valid_indices = []
                            for idx in candidate_indices:
                                # Create a temporary sequence with this token
                                temp_x = x[j].clone()
                                temp_x[idx] = x0[j, idx]
                                
                                # Check if it satisfies the regex
                                section_text = self.tokenizer.decode(temp_x[start_pos:end_pos].tolist())
                                if self.compiled_patterns[section_names[i]].match(section_text):
                                    valid_indices.append(idx)
                                    if len(valid_indices) >= section_transfers[j]:
                                        break
                            
                            # If we found valid indices, use them
                            if valid_indices:
                                for idx in valid_indices[:section_transfers[j]]:
                                    transfer_index[j, idx] = True
                            else:
                                # Fall back to top confidence if no valid tokens found
                                _, select_index = torch.topk(section_confidence[j], k=section_transfers[j])
                                for idx in select_index:
                                    transfer_index[j, idx] = True
                else:
                    # Standard top-k selection without regex validation
                    for j in range(batch_size):
                        if section_transfers[j] > 0:
                            _, select_index = torch.topk(section_confidence[j], k=section_transfers[j])
                            transfer_index[j, select_index] = True
        
        # Apply the transfers
        x_new = x.clone()
        x_new[transfer_index] = x0[transfer_index]
        
        return x_new
    
    def constrain_generation_with_regex(self, logits: torch.Tensor, input_ids: torch.Tensor, 
                                       section_markers: List[Tuple[int, int]], 
                                       section_names: List[str], position: int) -> torch.Tensor:
        """
        Constrain generation logits using regex patterns for autoregressive generation.
        
        Args:
            logits: Logits tensor (batch_size, vocab_size)
            input_ids: Current token IDs
            section_markers: List of (start_pos, end_pos) tuples for each section
            section_names: List of section names for regex validation
            position: Current position in the sequence
            
        Returns:
            Constrained logits tensor
        """
        if self.tokenizer is None or not section_names:
            return logits
        
        # Find which section the current position belongs to
        section_idx = -1
        for i, (start_pos, end_pos) in enumerate(section_markers):
            if start_pos <= position < end_pos:
                section_idx = i
                break
        
        # If not in any section or section doesn't have a regex pattern, return original logits
        if section_idx == -1 or section_idx >= len(section_names) or section_names[section_idx] not in self.regex_patterns:
            return logits
        
        # Apply regex filtering to constrain generation
        return self.filter_logits_with_regex(
            logits, input_ids, section_names[section_idx], position)
