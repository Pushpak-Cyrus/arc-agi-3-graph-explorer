"""
Test the Frame Processor on a REAL captured game frame,
instead of random dummy data.

Loads sample_frame.txt and tries multiple parsing strategies
since we don't know the exact saved format.
"""

import json
import ast
import numpy as np
from frame_processor import FrameProcessor


def load_real_frame(filepath: str) -> np.ndarray:
    """
    Try to load a real frame from a text file.
    Tries JSON first, then Python literal (ast.literal_eval) as fallback.
    """
    with open(filepath, "r") as f:
        content = f.read().strip()

    # If the file has multiple lines (one row per line), try joining/parsing differently
    try:
        # Try parsing the whole file as one JSON/list structure
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(content)
        except (ValueError, SyntaxError):
            # If it's multiple lines, each being a row, try line-by-line
            lines = content.split("\n")
            data = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    row = ast.literal_eval(line)
                data.append(row)

    arr = np.array(data)
    print(f"Loaded array with shape: {arr.shape}")
    return arr


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Frame Processor on REAL captured frame")
    print("=" * 60)

    # Try to load the real frame
    try:
        real_frame = load_real_frame("sample_frame.txt")
    except FileNotFoundError:
        print("\nERROR: sample_frame.txt not found in this directory.")
        print("Make sure it's in the same folder as this script,")
        print("or update the filepath below.")
        exit(1)
    except Exception as e:
        print(f"\nERROR loading frame: {e}")
        print("The file format might be different than expected.")
        print("Please paste the first few lines of sample_frame.txt")
        print("so the loader can be adjusted.")
        exit(1)

    # Reshape if needed
    print(f"\nOriginal shape: {real_frame.shape}")
    print(f"Value range: min={real_frame.min()}, max={real_frame.max()}")
    print(f"Unique values: {sorted(np.unique(real_frame).tolist())}")

    # Ensure it's 1x64x64
    if real_frame.shape != (1, 64, 64):
        real_frame = real_frame.reshape(1, 64, 64)
        print(f"Reshaped to: {real_frame.shape}")

    # Process it
    processor = FrameProcessor("ls20")
    state = processor.process(real_frame)

    print(f"\n--- RESULTS ---")
    print(f"State hash: {state['hash']}")
    print(f"Number of blobs found: {len(state['blobs'])}")
    print(f"\nTop 10 blobs (by priority tier, then size):")
    for i, blob in enumerate(state['blobs'][:10]):
        print(f"  Blob {i}: size={blob['size']}, value={blob['primary_value']}, tier={blob['priority_tier']}")

    print(f"\nBlob count by tier:")
    tier_counts = {}
    for blob in state['blobs']:
        tier = blob['priority_tier']
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
    for tier in sorted(tier_counts.keys()):
        print(f"  Tier {tier}: {tier_counts[tier]} blobs")