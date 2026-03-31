"""
compare_weights.py
------------------
Run BEFORE and AFTER main.py fine-tuning to confirm whether
storage/checkpoint.h5 was trained on more data than the 16 sample pickles.

Usage:
  # Step 1 - BEFORE training (snapshot original weights)
  python compare_weights.py --mode snapshot

  # Step 2 - Run training normally
  python main.py --config_file='configs/config_WC_TAN-MS.yml'

  # Step 3 - AFTER training (compare original vs fine-tuned)
  python compare_weights.py --mode compare
"""

import argparse
import numpy as np
import os

def read_h5_weights(h5_path):
    """Read all weights from an .h5 checkpoint using h5py (no TF/addons needed)."""
    import h5py
    weights = {}
    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            weights[name] = obj[()]
    with h5py.File(h5_path, 'r') as f:
        f.visititems(visitor)
    return weights


def snapshot(args):
    """Save original checkpoint weights as numpy arrays for later comparison."""
    print("=== SNAPSHOT MODE: Saving original pre-trained weights ===")

    pretrained_path = 'storage/checkpoint.h5'
    if not os.path.exists(pretrained_path):
        print(f"ERROR: Pre-trained checkpoint not found at {pretrained_path}")
        return

    weights = read_h5_weights(pretrained_path)
    os.makedirs('log/weight_comparison', exist_ok=True)
    np.savez('log/weight_comparison/original_weights.npz', **{
        k.replace('/', '__'): v for k, v in weights.items()
    })

    total_params = sum(v.size for v in weights.values())
    print(f"Loaded {len(weights)} weight tensors from: {pretrained_path}")
    print(f"Total parameters snapshotted: {total_params:,}")
    print(f"\nSnapshot saved → log/weight_comparison/original_weights.npz")
    print("\nNext step — run training:")
    print("  python main.py --config_file='configs/config_WC_TAN-MS.yml'")
    print("Then run: python compare_weights.py --mode compare")


def compare(args):
    """Compare original weights to post-fine-tuning weights."""
    print("=== COMPARE MODE: Measuring weight changes after fine-tuning ===\n")

    snapshot_path = 'log/weight_comparison/original_weights.npz'
    finetuned_path = 'log/WC_TAN-MS/checkpoint.h5'

    if not os.path.exists(snapshot_path):
        print(f"ERROR: Original snapshot not found. Run --mode snapshot first.")
        return
    if not os.path.exists(finetuned_path):
        print(f"ERROR: Fine-tuned checkpoint not found at {finetuned_path}")
        print("Run training first: python main.py --config_file='configs/config_WC_TAN-MS.yml'")
        return

    # Load snapshots
    snap = np.load(snapshot_path, allow_pickle=True)
    orig_weights = {k.replace('__', '/'): snap[k] for k in snap.files}

    # Load fine-tuned weights
    new_weights = read_h5_weights(finetuned_path)

    # Match keys
    common = set(orig_weights.keys()) & set(new_weights.keys())
    print(f"Comparing {len(common)} matched weight tensors\n")
    print(f"{'Layer':<55} {'L2 Change':>12} {'% Change':>10} {'Type':>12}")
    print("-" * 92)

    all_pct = []
    lora_pct = []
    non_lora_pct = []

    for key in sorted(common):
        orig = orig_weights[key].astype(np.float32)
        new  = new_weights[key].astype(np.float32)
        l2_change = float(np.linalg.norm(orig - new))
        l2_orig   = float(np.linalg.norm(orig))
        pct = (l2_change / (l2_orig + 1e-10)) * 100

        is_lora = 'lora' in key.lower()
        label = "LoRA" if is_lora else "Backbone"

        display_key = key[-55:] if len(key) > 55 else key
        print(f"{display_key:<55} {l2_change:>12.6f} {pct:>9.2f}% {label:>12}")

        all_pct.append(pct)
        (lora_pct if is_lora else non_lora_pct).append(pct)

    print("\n" + "=" * 92)
    print(f"\n{'Statistic':<40} {'Value':>15}")
    print(f"{'Overall avg weight change':<40} {np.mean(all_pct):>14.4f}%")
    print(f"{'Backbone (frozen) layers avg':<40} {np.mean(non_lora_pct) if non_lora_pct else 0:>14.4f}%")
    print(f"{'LoRA adapter layers avg':<40} {np.mean(lora_pct) if lora_pct else 0:>14.4f}%")

    backbone_avg = np.mean(non_lora_pct) if non_lora_pct else np.mean(all_pct)

    print("\n=== VERDICT ===")
    if backbone_avg < 2.0:
        print("RESULT: Backbone weights barely changed (<2%).")
        print("   The checkpoint was TRAINED ON A MUCH LARGER DATASET than the 16 sample pickles.")
        print("   -> The authors pre-trained model is from their full private dataset.")
        print("   -> Tell Dr. Mazen: checkpoint.h5 contains full pre-training, not just sample data.")
    elif backbone_avg < 10.0:
        print("WARNING: Moderate backbone change (2-10%).")
        print("   Inconclusive. Either checkpoint had limited pre-training or LoRA leaked.")
    else:
        print("LARGE CHANGE: Backbone changed >10%.")
        print("   The model learned a lot from the 16 sample pickles.")
        print("   The checkpoint may NOT be substantially pre-trained on a larger dataset.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare DeepMachining weights before/after fine-tuning')
    parser.add_argument('--mode', choices=['snapshot', 'compare'], required=True)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if args.mode == 'snapshot':
        snapshot(args)
    else:
        compare(args)
