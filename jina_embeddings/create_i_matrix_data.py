#!/usr/bin/env python3
"""
Script to create randomized calibration data for llama.cpp importance matrix.
Based on research showing that near-random data significantly outperforms structured data.
Uses NLPAUG library for professional-grade text augmentation.

Requirements: pip install nlpaug pandas click
"""

import click # type: ignore
import pandas as pd # type: ignore
import random
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# NLPAUG imports - required
import nlpaug.augmenter.char as nac # type: ignore
import nlpaug.augmenter.word as naw # type: ignore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextScrambler:
    """Professional text scrambling using NLPAUG for importance matrix calibration."""
    
    def __init__(self, char_prob=0.05, word_prob=0.1, keyboard_prob=0.03,
                 remove_stopwords=False, remove_punctuation=False):
        """
        Initialize NLPAUG augmenters with research-backed parameters.
        
        Args:
            char_prob: Probability of character-level changes
            word_prob: Probability of word-level changes  
            keyboard_prob: Probability of keyboard error simulation
            remove_stopwords: Whether to remove stop words
            remove_punctuation: Whether to remove punctuation
        """
        # Character-level augmenters
        self.char_substitute = nac.RandomCharAug(action="substitute", aug_char_p=char_prob)
        self.char_insert = nac.RandomCharAug(action="insert", aug_char_p=char_prob/2)
        self.char_delete = nac.RandomCharAug(action="delete", aug_char_p=char_prob/2)
        self.keyboard_aug = nac.KeyboardAug(aug_char_p=keyboard_prob)
        self.ocr_aug = nac.OcrAug(aug_char_p=keyboard_prob)
        
        # Word-level augmenters (only using implemented actions)
        self.word_swap = naw.RandomWordAug(action="swap", aug_p=word_prob)
        self.word_substitute = naw.RandomWordAug(action="substitute", aug_p=word_prob/2)
        self.word_delete = naw.RandomWordAug(action="delete", aug_p=word_prob/3)
        
        # Preprocessing augmenters
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        
        if remove_stopwords:
            # Using reserved word augmenter for stop word removal (multilingual support)
            self.stopword_aug = naw.ReservedAug(
                reserved_words=None,  # Uses default multilingual stopwords
                action="delete"
            )
            logger.info("✓ Stop word removal enabled (multilingual)")
        
        logger.info("✓ NLPAUG augmenters initialized for high-entropy scrambling")
    
    def _safe_augment(self, augmenter, text: str) -> str:
        """Safely apply augmentation and handle list/string returns."""
        try:
            result = augmenter.augment(text)
            # NLPAUG returns a list, we want the first result
            if isinstance(result, list):
                return result[0] if result else text
            return result
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}, returning original text")
            return text
    
    def preprocess(self, text: str) -> str:
        """Apply preprocessing (stop word and punctuation removal)."""
        result = text
        
        if self.remove_punctuation:
            # Use regex for multilingual punctuation removal
            # This pattern removes all non-letter, non-digit, non-space characters
            # Works across all Unicode categories including multilingual punctuation
            import re
            result = re.sub(r'[^\w\s]', ' ', result, flags=re.UNICODE)
            # Clean up multiple spaces
            result = re.sub(r'\s+', ' ', result).strip()
        
        if self.remove_stopwords:
            result = self._safe_augment(self.stopword_aug, result)
        
        return result
    
    def light_scramble(self, text: str) -> str:
        """Light scrambling - maintains more readability."""
        result = self.preprocess(text)
        result = self._safe_augment(self.keyboard_aug, result)
        result = self._safe_augment(self.word_swap, result)
        return result
    
    def medium_scramble(self, text: str) -> str:
        """Medium scrambling - balanced chaos and structure."""
        result = self.preprocess(text)
        result = self._safe_augment(self.char_substitute, result)
        result = self._safe_augment(self.keyboard_aug, result)
        result = self._safe_augment(self.word_swap, result)
        result = self._safe_augment(self.word_substitute, result)
        return result
    
    def high_entropy_scramble(self, text: str) -> str:
        """
        Maximum entropy scrambling - recommended for importance matrices.
        Applies multiple NLPAUG transformations for optimal randomization.
        """
        result = self.preprocess(text)
        
        # Character-level chaos
        result = self._safe_augment(self.char_substitute, result)
        result = self._safe_augment(self.char_insert, result)
        result = self._safe_augment(self.char_delete, result)
        result = self._safe_augment(self.keyboard_aug, result)
        result = self._safe_augment(self.ocr_aug, result)
        
        # Word-level chaos (only implemented actions)
        result = self._safe_augment(self.word_swap, result)
        result = self._safe_augment(self.word_substitute, result)
        result = self._safe_augment(self.word_delete, result)
        
        return result
    
    def extreme_scramble(self, text: str) -> str:
        """Extreme scrambling - maximum chaos for testing."""
        # Apply high entropy twice for maximum randomization
        result = self.high_entropy_scramble(text)
        result = self.high_entropy_scramble(result)
        return result


def get_parquet_files(directory: Path) -> List[Path]:
    """Get all parquet files from a directory."""
    return list(directory.glob("*.parquet"))


def read_random_samples(directory: Path, num_samples: int) -> List[Tuple[str, str]]:
    """Read random samples from parquet files in a directory."""
    parquet_files = get_parquet_files(directory)
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {directory}")
        return []
    
    samples = []
    samples_per_file = max(1, num_samples // len(parquet_files))
    remaining_samples = num_samples % len(parquet_files)
    
    random.shuffle(parquet_files)
    
    for i, parquet_file in enumerate(parquet_files):
        try:
            df = pd.read_parquet(parquet_file)
            
            if 'left' not in df.columns or 'right' not in df.columns:
                logger.warning(f"Required columns 'left' and 'right' not found in {parquet_file}")
                continue
            
            current_samples = samples_per_file
            if i < remaining_samples:
                current_samples += 1
            
            if len(df) > 0:
                sample_size = min(current_samples, len(df))
                sampled_rows = df.sample(n=sample_size, random_state=random.randint(0, 10000))
                
                for _, row in sampled_rows.iterrows():
                    samples.append((str(row['left']), str(row['right'])))
            
            if len(samples) >= num_samples:
                break
                
        except Exception as e:
            logger.error(f"Error reading {parquet_file}: {e}")
            continue
    
    return samples[:num_samples]


def format_for_llama_cpp(samples: List[Tuple[str, str]], 
                        left_prefix: str = "", 
                        right_prefix: str = "",
                        scrambler: Optional[TextScrambler] = None,
                        scramble_method: str = 'high_entropy') -> str:
    """
    Format samples for llama.cpp importance matrix with NLPAUG scrambling.
    """
    separator = "\n\n"
    formatted_lines = []
    
    for left, right in samples:
        # Apply prefixes
        formatted_line = f"{left_prefix}{left}\n{right_prefix}{right}"
        
        # Apply NLPAUG scrambling
        if scrambler and scramble_method != 'none':
            if scramble_method == 'light':
                formatted_line = scrambler.light_scramble(formatted_line)
            elif scramble_method == 'medium':
                formatted_line = scrambler.medium_scramble(formatted_line)
            elif scramble_method == 'high_entropy':
                formatted_line = scrambler.high_entropy_scramble(formatted_line)
            elif scramble_method == 'extreme':
                formatted_line = scrambler.extreme_scramble(formatted_line)
        
        formatted_lines.append(formatted_line)
    
    return f"{separator}".join(formatted_lines)


@click.command()
@click.option('--folders', '-f', multiple=True, required=True,
              help='List of folders containing parquet files.')
@click.option('--samples-per-folder', '-s', default=10, type=int,
              help='Number of samples to read from each folder (default: 10)')
@click.option('--output', '-o', default='importance_matrix_random.txt',
              help='Output text file path (default: importance_matrix_random.txt)')
@click.option('--left-prefix', default='', help='Prefix for left column content')
@click.option('--right-prefix', default='', help='Prefix for right column content')
@click.option('--scramble-method', 
              type=click.Choice(['none', 'light', 'medium', 'high_entropy', 'extreme']),
              default='high_entropy',
              help='NLPAUG scrambling intensity (default: high_entropy - recommended for importance matrices)')
@click.option('--char-prob', default=0.05, type=float,
              help='Character-level scrambling probability (default: 0.05)')
@click.option('--word-prob', default=0.1, type=float,
              help='Word-level scrambling probability (default: 0.1)')
@click.option('--keyboard-prob', default=0.03, type=float,
              help='Keyboard error simulation probability (default: 0.03)')
@click.option('--remove-stopwords', is_flag=True,
              help='Remove stop words before scrambling (multilingual support)')
@click.option('--remove-punctuation', is_flag=True,
              help='Remove punctuation before scrambling')
@click.option('--seed', default=42, type=int, help='Random seed for reproducibility')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(folders: Tuple[str], samples_per_folder: int, output: str, 
         left_prefix: str, right_prefix: str, scramble_method: str, 
         char_prob: float, word_prob: float, keyboard_prob: float,
         remove_stopwords: bool, remove_punctuation: bool,
         seed: int, verbose: bool):
    """
    Create randomized calibration data for llama.cpp importance matrix using NLPAUG.
    
    Based on research showing that near-random data significantly outperforms
    structured data for importance matrix calculations.
    
    Requires: pip install nlpaug pandas click
    
    Examples:
      # High-entropy scrambling (recommended)
      python script.py -f folder1 folder2 --scramble-method high_entropy
      
      # With stop word and punctuation removal
      python script.py -f folder1 --scramble-method high_entropy --remove-stopwords --remove-punctuation
      
      # Custom probability settings
      python script.py -f folder1 --char-prob 0.1 --word-prob 0.2 --scramble-method medium
      
      # Maximum chaos for testing
      python script.py -f folder1 --scramble-method extreme
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    random.seed(seed)
    
    logger.info(f"Creating {'randomized' if scramble_method != 'none' else 'structured'} calibration data")
    logger.info(f"NLPAUG scrambling method: {scramble_method}")
    logger.info(f"Parameters: char_prob={char_prob}, word_prob={word_prob}, keyboard_prob={keyboard_prob}")
    if remove_stopwords:
        logger.info("✓ Stop word removal enabled (multilingual)")
    if remove_punctuation:
        logger.info("✓ Punctuation removal enabled")
    logger.info(f"Random seed: {seed}")
    
    # Initialize NLPAUG scrambler
    scrambler = None
    if scramble_method != 'none':
        scrambler = TextScrambler(
            char_prob=char_prob, 
            word_prob=word_prob, 
            keyboard_prob=keyboard_prob,
            remove_stopwords=remove_stopwords,
            remove_punctuation=remove_punctuation
        )
        logger.info("✓ NLPAUG scrambler initialized for research-backed randomization")
    else:
        logger.warning("⚠ Using structured data - research shows this performs worse for importance matrices!")
        logger.info("💡 Recommendation: Use --scramble-method high_entropy")
    
    all_samples = []
    
    for folder_path in folders:
        folder = Path(folder_path)
        
        if not folder.exists():
            logger.error(f"Folder does not exist: {folder}")
            continue
        
        if not folder.is_dir():
            logger.error(f"Path is not a directory: {folder}")
            continue
        
        logger.info(f"Processing folder: {folder}")
        samples = read_random_samples(folder, samples_per_folder)
        logger.info(f"Read {len(samples)} samples from {folder}")
        all_samples.extend(samples)
    
    if not all_samples:
        logger.error("No samples were collected. Please check your input folders and parquet files.")
        return
    
    logger.info(f"Total samples collected: {len(all_samples)}")
    
    # Shuffle all samples for additional randomness
    random.shuffle(all_samples)
    
    # Format samples with NLPAUG scrambling
    formatted_content = format_for_llama_cpp(
        all_samples, left_prefix, right_prefix, scrambler, scramble_method
    )
    
    # Write to output file
    output_path = Path(output)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        
        logger.info(f"✓ Successfully wrote {len(all_samples)} samples to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size:,} bytes")
        
        if scramble_method != 'none':
            logger.info("✓ Calibration data optimized using NLPAUG (research-backed for importance matrices)")
        else:
            logger.warning("⚠ Consider using --scramble-method high_entropy for better quantization results")
        
    except Exception as e:
        logger.error(f"Error writing to output file: {e}")
        return


if __name__ == "__main__":
    main() # type: ignore