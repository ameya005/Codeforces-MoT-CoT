#!/usr/bin/env python
"""
Utility functions for model training and evaluation.
"""

import os
import argparse
import subprocess
import sys
import shutil


def convert_deepspeed_checkpoint(checkpoint_path, output_dir):
    """
    Convert a DeepSpeed checkpoint using zero_to_fp32.py (recommended for ZeRO stage 3).
    
    Args:
        checkpoint_path (str): Path to the DeepSpeed checkpoint directory
        output_dir (str): Output directory for the converted HuggingFace model
    """
    print(f"Converting DeepSpeed checkpoint using zero_to_fp32.py...")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if zero_to_fp32.py exists in the current environment
    try:
        # Try to find zero_to_fp32.py in the current Python environment
        import deepspeed
        deepspeed_path = os.path.dirname(deepspeed.__file__)
        zero_to_fp32_script = os.path.join(deepspeed_path, "utils", "zero_to_fp32.py")
        
        if not os.path.exists(zero_to_fp32_script):
            print(f"zero_to_fp32.py not found at {zero_to_fp32_script}")
            print("Please install DeepSpeed or check your environment")
            return False
            
        print(f"Found zero_to_fp32.py at: {zero_to_fp32_script}")
        
        # Run zero_to_fp32.py
        cmd = [
            sys.executable, zero_to_fp32_script,
            checkpoint_path,  # checkpoint directory
            output_dir,       # output directory
            "fp32"           # precision (fp32, fp16, bf16)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Model successfully converted using zero_to_fp32.py")
            print(f"Output saved to: {output_dir}")
            
            # Copy tokenizer files if they exist
            tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
            for file in tokenizer_files:
                src = os.path.join(checkpoint_path, file)
                dst = os.path.join(output_dir, file)
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f"   Copied {file}")
            
            return True
        else:
            print(f"zero_to_fp32.py failed with return code {result.returncode}")
            print(f"   stdout: {result.stdout}")
            print(f"   stderr: {result.stderr}")
            return False
            
    except ImportError:
        print("DeepSpeed not found. Please install it first:")
        print("   pip install deepspeed")
        return False
    except Exception as e:
        print(f"Error running zero_to_fp32.py: {e}")
        return False


def main():
    """Command line interface for converting checkpoints."""
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to HuggingFace format using zero_to_fp32.py")
    parser.add_argument("--checkpoint_path", required=True, help="Path to DeepSpeed checkpoint directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for converted model")
    
    args = parser.parse_args()
    
    if convert_deepspeed_checkpoint(args.checkpoint_path, args.output_dir):
        print("Conversion completed successfully!")
        print(f"You can now use {args.output_dir} with --checkpoint_path in cf_eval.py")
    else:
        print("Conversion failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
