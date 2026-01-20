#!/usr/bin/env python3
"""
Plot the relationship between matrix sparsity ratio and time ratio.

Helps determine if m1_nvals / m3_nvals predicts which direction is faster.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Plot matrix sparsity ratio vs time ratio'
    )
    parser.add_argument('--input', default='direction_analysis.tsv',
                        help='Input TSV file from direction analysis')
    parser.add_argument('--output', default='matrix_ratio_vs_time_ratio.png',
                        help='Output PNG file')
    parser.add_argument('--use-amortized', action='store_true',
                        help='Use amortized times instead of total times (default: total times)')

    args = parser.parse_args()

    # Read data
    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, sep='\t')

    # Calculate ratios
    print("Calculating ratios...")

    # Matrix sparsity ratio (m1 / m3)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    df['matrix_ratio'] = df['m1_nvals'] / (df['m3_nvals'] + epsilon)

    # Time ratio (forward / reverse)
    if args.use_amortized:
        df['time_ratio'] = df['forward_amortized_time'] / (df['reverse_amortized_time'] + epsilon)
        time_label = 'Forward Amortized Time / Reverse Amortized Time'
        title_suffix = ' (Amortized)'
        better_col = 'amortized_better'
    else:
        df['time_ratio'] = df['forward_total_time'] / (df['reverse_total_time'] + epsilon)
        time_label = 'Forward Total Time / Reverse Total Time'
        title_suffix = ' (Total Time)'
        better_col = 'better_direction'

    # Filter out infinite/NaN values
    df = df[np.isfinite(df['matrix_ratio']) & np.isfinite(df['time_ratio'])]

    print(f"Plotting {len(df)} samples...")

    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Color by which direction is faster
    colors = df[better_col].map({
        'forward': 'blue',
        'reverse': 'red',
        'equal': 'gray'
    })

    plt.scatter(df['matrix_ratio'], df['time_ratio'],
                c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add reference line at y=1 (equal time)
    plt.axhline(y=1, color='black', linestyle='--', linewidth=1, alpha=0.5,
                label='Equal time (ratio=1)')

    # Add reference line at x=1 (equal sparsity)
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1, alpha=0.5,
                label='Equal sparsity (ratio=1)')

    # Log scale for both axes (ratios can span large ranges)
    plt.xscale('log')
    plt.yscale('log')

    plt.xlabel('M1 Sparsity / M3 Sparsity (m1_nvals / m3_nvals)', fontsize=12)
    plt.ylabel(time_label, fontsize=12)
    plt.title(f'Matrix Sparsity Ratio vs Time Ratio{title_suffix}', fontsize=14, fontweight='bold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', edgecolor='black', label='Forward faster'),
        Patch(facecolor='red', edgecolor='black', label='Reverse faster'),
        Patch(facecolor='gray', edgecolor='black', label='Equal'),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    # Add grid
    plt.grid(True, alpha=0.3, which='both')

    # Add text annotation explaining quadrants
    plt.text(0.02, 0.98,
             'Upper left: M1 sparse, forward slow\n'
             'Upper right: M1 dense, forward slow\n'
             'Lower left: M1 sparse, forward fast\n'
             'Lower right: M1 dense, forward fast',
             transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    # Correlation
    correlation = df['matrix_ratio'].corr(df['time_ratio'])
    print(f"\nPearson correlation: {correlation:.3f}")

    # Quadrant analysis
    forward_faster = df['time_ratio'] < 1
    m1_denser = df['matrix_ratio'] > 1

    print(f"\nQuadrant breakdown:")
    print(f"  M1 denser + forward faster: {sum(m1_denser & forward_faster)} ({sum(m1_denser & forward_faster)/len(df)*100:.1f}%)")
    print(f"  M1 denser + reverse faster: {sum(m1_denser & ~forward_faster)} ({sum(m1_denser & ~forward_faster)/len(df)*100:.1f}%)")
    print(f"  M3 denser + forward faster: {sum(~m1_denser & forward_faster)} ({sum(~m1_denser & forward_faster)/len(df)*100:.1f}%)")
    print(f"  M3 denser + reverse faster: {sum(~m1_denser & ~forward_faster)} ({sum(~m1_denser & ~forward_faster)/len(df)*100:.1f}%)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
