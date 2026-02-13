#!/usr/bin/env python3
"""
Plot the relationship between matrix sparsity ratio and memory ratio.

Helps determine if m1_nvals / m3_nvals predicts which direction uses less memory.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Plot matrix sparsity ratio vs memory ratio'
    )
    parser.add_argument('--input', default='direction_analysis.tsv',
                        help='Input TSV file from direction analysis')
    parser.add_argument('--output', default='matrix_ratio_vs_memory_ratio.png',
                        help='Output PNG file')

    args = parser.parse_args()

    # Read data
    print(f"Reading {args.input}...")
    df = pd.read_csv(args.input, sep='\t')

    # Calculate ratios
    print("Calculating ratios...")

    # Matrix sparsity ratio (m1 / m3)
    epsilon = 1e-10
    df['matrix_ratio'] = df['m1_nvals'] / (df['m3_nvals'] + epsilon)

    # Maximum memory used in each direction
    df['forward_max_mem'] = df[['forward_step1_mem_mb', 'forward_step2_mem_mb']].max(axis=1)
    df['reverse_max_mem'] = df[['reverse_step1_mem_mb', 'reverse_step2_mem_mb']].max(axis=1)

    # Memory ratio (forward / reverse)
    df['memory_ratio'] = df['forward_max_mem'] / (df['reverse_max_mem'] + epsilon)

    # Determine which direction uses less memory
    df['memory_better'] = 'equal'
    df.loc[df['forward_max_mem'] < df['reverse_max_mem'], 'memory_better'] = 'forward'
    df.loc[df['reverse_max_mem'] < df['forward_max_mem'], 'memory_better'] = 'reverse'

    # Filter out infinite/NaN values
    df = df[np.isfinite(df['matrix_ratio']) & np.isfinite(df['memory_ratio'])]

    print(f"Plotting {len(df)} samples...")

    # Create 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Subplot 1: forward_step1_mem vs m1_nvals
    ax = axes[0, 0]
    ax.scatter(df['m1_nvals'], df['forward_step1_mem_mb'], alpha=0.6, s=50,
               edgecolors='black', linewidth=0.5, color='blue')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('M1 Sparsity (m1_nvals)', fontsize=11)
    ax.set_ylabel('Forward Step 1 Memory (MB)', fontsize=11)
    ax.set_title('Forward: First Intermediate Memory vs M1 Sparsity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    # Calculate correlation
    valid = (df['m1_nvals'] > 0) & (df['forward_step1_mem_mb'] > 0)
    if valid.sum() > 1:
        corr1 = df.loc[valid, 'm1_nvals'].corr(df.loc[valid, 'forward_step1_mem_mb'])
        ax.text(0.05, 0.95, f'Correlation: {corr1:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 2: forward_step2_mem vs m1_nvals
    ax = axes[0, 1]
    ax.scatter(df['m1_nvals'], df['forward_step2_mem_mb'], alpha=0.6, s=50,
               edgecolors='black', linewidth=0.5, color='green')
    ax.set_xscale('log')
    # Only use log scale if there are meaningful non-zero values
    if (df['forward_step2_mem_mb'] > 0.01).sum() > 5:
        ax.set_yscale('log')
    ax.set_xlabel('M1 Sparsity (m1_nvals)', fontsize=11)
    ax.set_ylabel('Forward Step 2 Memory (MB)', fontsize=11)
    ax.set_title('Forward: Final Result Memory vs M1 Sparsity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    valid = (df['m1_nvals'] > 0) & (df['forward_step2_mem_mb'] > 0)
    if valid.sum() > 1:
        corr2 = df.loc[valid, 'm1_nvals'].corr(df.loc[valid, 'forward_step2_mem_mb'])
        ax.text(0.05, 0.95, f'Correlation: {corr2:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 3: reverse_step1_mem vs m3_nvals
    ax = axes[1, 0]
    ax.scatter(df['m3_nvals'], df['reverse_step1_mem_mb'], alpha=0.6, s=50,
               edgecolors='black', linewidth=0.5, color='red')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('M3 Sparsity (m3_nvals)', fontsize=11)
    ax.set_ylabel('Reverse Step 1 Memory (MB)', fontsize=11)
    ax.set_title('Reverse: First Intermediate Memory vs M3 Sparsity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    valid = (df['m3_nvals'] > 0) & (df['reverse_step1_mem_mb'] > 0)
    if valid.sum() > 1:
        corr3 = df.loc[valid, 'm3_nvals'].corr(df.loc[valid, 'reverse_step1_mem_mb'])
        ax.text(0.05, 0.95, f'Correlation: {corr3:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Subplot 4: reverse_step2_mem vs m3_nvals
    ax = axes[1, 1]
    ax.scatter(df['m3_nvals'], df['reverse_step2_mem_mb'], alpha=0.6, s=50,
               edgecolors='black', linewidth=0.5, color='orange')
    ax.set_xscale('log')
    # Only use log scale if there are meaningful non-zero values
    if (df['reverse_step2_mem_mb'] > 0.01).sum() > 5:
        ax.set_yscale('log')
    ax.set_xlabel('M3 Sparsity (m3_nvals)', fontsize=11)
    ax.set_ylabel('Reverse Step 2 Memory (MB)', fontsize=11)
    ax.set_title('Reverse: Final Result Memory vs M3 Sparsity', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')

    valid = (df['m3_nvals'] > 0) & (df['reverse_step2_mem_mb'] > 0)
    if valid.sum() > 1:
        corr4 = df.loc[valid, 'm3_nvals'].corr(df.loc[valid, 'reverse_step2_mem_mb'])
        ax.text(0.05, 0.95, f'Correlation: {corr4:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")

    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")

    # Correlation
    correlation = df['matrix_ratio'].corr(df['memory_ratio'])
    print(f"\nPearson correlation: {correlation:.3f}")

    # Overall memory usage comparison
    print(f"\nMemory usage:")
    print(f"  Forward max memory - Mean: {df['forward_max_mem'].mean():.1f} MB, Median: {df['forward_max_mem'].median():.1f} MB")
    print(f"  Reverse max memory - Mean: {df['reverse_max_mem'].mean():.1f} MB, Median: {df['reverse_max_mem'].median():.1f} MB")

    # Which direction uses less memory
    forward_better = (df['memory_better'] == 'forward').sum()
    reverse_better = (df['memory_better'] == 'reverse').sum()
    equal = (df['memory_better'] == 'equal').sum()

    print(f"\nDirection with lower memory:")
    print(f"  Forward: {forward_better} ({forward_better/len(df)*100:.1f}%)")
    print(f"  Reverse: {reverse_better} ({reverse_better/len(df)*100:.1f}%)")
    print(f"  Equal: {equal} ({equal/len(df)*100:.1f}%)")

    # Quadrant analysis
    forward_less_memory = df['memory_ratio'] < 1
    m1_denser = df['matrix_ratio'] > 1

    print(f"\nQuadrant breakdown:")
    print(f"  M1 denser + forward less memory: {sum(m1_denser & forward_less_memory)} ({sum(m1_denser & forward_less_memory)/len(df)*100:.1f}%)")
    print(f"  M1 denser + reverse less memory: {sum(m1_denser & ~forward_less_memory)} ({sum(m1_denser & ~forward_less_memory)/len(df)*100:.1f}%)")
    print(f"  M3 denser + forward less memory: {sum(~m1_denser & forward_less_memory)} ({sum(~m1_denser & forward_less_memory)/len(df)*100:.1f}%)")
    print(f"  M3 denser + reverse less memory: {sum(~m1_denser & ~forward_less_memory)} ({sum(~m1_denser & ~forward_less_memory)/len(df)*100:.1f}%)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
