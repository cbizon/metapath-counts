#!/usr/bin/env python3
"""
Analyze 3-hop metapaths as predictors of 1-hop metapaths.

For each (3-hop, 1-hop) pair, we treat the 3-hop metapath as a binary predictor:
- If a 3-hop path exists between nodes A and B, we predict a 1-hop path should exist
- We calculate standard classification metrics (precision, recall, accuracy, F1, etc.)

Key challenge: We have path counts, not unique node pair counts.
Strategy: Use the "ANY" aggregated rows to estimate unique node pairs with 3-hop paths.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def calculate_metrics(tp, fp, fn, tn):
    """Calculate classification metrics from confusion matrix."""
    total = tp + fp + fn + tn

    # Handle edge cases
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative predictive value

    accuracy = (tp + tn) / total if total > 0 else 0.0

    # F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Matthews Correlation Coefficient
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denom_squared = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    mcc_denominator = np.sqrt(float(mcc_denom_squared)) if mcc_denom_squared > 0 else 0.0
    mcc = mcc_numerator / mcc_denominator if mcc_denominator > 0 else 0.0

    # True positive rate (sensitivity/recall)
    tpr = recall

    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # False negative rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Balanced accuracy
    balanced_accuracy = (tpr + specificity) / 2

    # Positive likelihood ratio
    plr = tpr / fpr if fpr > 0 else np.inf

    # Negative likelihood ratio
    nlr = fnr / specificity if specificity > 0 else np.inf

    return {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Total': total,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'NPV': npv,
        'Accuracy': accuracy,
        'Balanced_Accuracy': balanced_accuracy,
        'F1': f1,
        'MCC': mcc,
        'TPR': tpr,
        'FPR': fpr,
        'FNR': fnr,
        'PLR': plr,
        'NLR': nlr,
    }


def main():
    # Read the overlap data
    df = pd.read_csv('3hop_1hop_overlap.tsv', sep='\t')

    print(f"Loaded {len(df)} rows")
    print(f"Unique 3-hop metapaths: {df['3hop_metapath'].nunique()}")
    print()

    # The 3hop_count column already contains unique node pairs with the 3-hop path
    # We'll analyze ALL rows including the ANY rows (which predict "any 1-hop exists")

    print(f"Analyzing {len(df)} (3-hop, 1-hop) pairs (including ANY predictions)")
    print()

    # Calculate metrics for each (3-hop, 1-hop) pair
    results = []

    for _, row in df.iterrows():
        hop3 = row['3hop_metapath']
        hop1 = row['1hop_metapath']
        hop3_count = row['3hop_count']  # Unique node pairs with 3-hop
        hop1_count = row['1hop_count']  # Unique node pairs with this 1-hop
        overlap = row['overlap']        # Unique node pairs with both
        total_possible = row['total_possible']

        # Confusion matrix:
        # TP: Node pairs with both 3-hop and this 1-hop
        tp = overlap

        # FP: Node pairs with 3-hop but NOT this 1-hop
        fp = hop3_count - overlap

        # FN: Node pairs with this 1-hop but NOT the 3-hop
        fn = hop1_count - overlap

        # TN: Node pairs with neither
        # Total possible - pairs with 3-hop - pairs with 1-hop + overlap (inclusion-exclusion)
        tn = total_possible - hop3_count - hop1_count + overlap

        # Ensure non-negative values
        fp = max(0, fp)
        fn = max(0, fn)
        tn = max(0, tn)

        # Calculate metrics
        metrics = calculate_metrics(tp, fp, fn, tn)

        # Add identifying information
        result = {
            '3hop_metapath': hop3,
            '1hop_metapath': hop1,
            '3hop_unique_pairs': hop3_count,
            '1hop_unique_pairs': hop1_count,
            'overlap': overlap,
            'total_possible_pairs': total_possible,
            **metrics
        }

        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by F1 score descending
    results_df = results_df.sort_values('F1', ascending=False)

    # Save full results
    output_file = 'metapath_prediction_metrics.tsv'
    results_df.to_csv(output_file, sep='\t', index=False, float_format='%.6f')
    print(f"Saved detailed results to {output_file}")
    print()

    # Print summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print("Metric Distributions:")
    print("-" * 80)
    metric_cols = ['Precision', 'Recall', 'Specificity', 'Accuracy', 'Balanced_Accuracy',
                   'F1', 'MCC', 'FPR', 'FNR']
    summary_stats = results_df[metric_cols].describe()
    print(summary_stats.to_string())
    print()

    # Top 10 by F1 score
    print("=" * 80)
    print("TOP 10 PREDICTIONS BY F1 SCORE")
    print("=" * 80)
    print()
    top10 = results_df.head(10)
    display_cols = ['3hop_metapath', '1hop_metapath', 'Precision', 'Recall', 'F1', 'MCC']
    print(top10[display_cols].to_string(index=False))
    print()

    # Best precision
    print("=" * 80)
    print("TOP 10 BY PRECISION")
    print("=" * 80)
    print()
    top_precision = results_df.nlargest(10, 'Precision')
    print(top_precision[display_cols].to_string(index=False))
    print()

    # Best recall
    print("=" * 80)
    print("TOP 10 BY RECALL")
    print("=" * 80)
    print()
    top_recall = results_df.nlargest(10, 'Recall')
    print(top_recall[display_cols].to_string(index=False))
    print()

    # Best balanced accuracy
    print("=" * 80)
    print("TOP 10 BY BALANCED ACCURACY")
    print("=" * 80)
    print()
    top_balanced = results_df.nlargest(10, 'Balanced_Accuracy')
    print(top_balanced[display_cols].to_string(index=False))
    print()

    # Aggregate by 3-hop metapath
    print("=" * 80)
    print("AGGREGATE PERFORMANCE BY 3-HOP METAPATH")
    print("=" * 80)
    print()

    agg_by_3hop = results_df.groupby('3hop_metapath').agg({
        'Precision': 'mean',
        'Recall': 'mean',
        'F1': 'mean',
        'MCC': 'mean',
        'Balanced_Accuracy': 'mean',
        '1hop_metapath': 'count'
    }).rename(columns={'1hop_metapath': 'num_1hop_tested'})

    agg_by_3hop = agg_by_3hop.sort_values('F1', ascending=False)
    print(agg_by_3hop.head(20).to_string())
    print()

    # Save aggregated results
    agg_output = 'metapath_prediction_by_3hop.tsv'
    agg_by_3hop.to_csv(agg_output, sep='\t', float_format='%.6f')
    print(f"Saved aggregated results to {agg_output}")


if __name__ == '__main__':
    main()
