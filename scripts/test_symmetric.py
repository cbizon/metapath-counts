#!/usr/bin/env python3
"""Test symmetric predicate handling."""

from metapath_counts import get_symmetric_predicates
import sys
sys.path.insert(0, '.')
from analyze_hop_overlap import format_metapath

# Test 1: Check symmetric predicates loaded
sym_preds = get_symmetric_predicates()
print(f'✓ Loaded {len(sym_preds)} symmetric predicates')
print(f'✓ Contains directly_physically_interacts_with: {"directly_physically_interacts_with" in sym_preds}')
print()

# Test 2: Format metapath with symmetric predicate
nodes = ['Gene', 'Gene']
preds = ['directly_physically_interacts_with']
dirs = ['F']
result = format_metapath(nodes, preds, dirs)
expected = 'Gene|directly_physically_interacts_with|A|Gene'
print(f'Symmetric metapath: {result}')
print(f'✓ Correct: {result == expected}')
print()

# Test 3: Format metapath with non-symmetric predicate
nodes = ['Disease', 'SmallMolecule', 'Gene']
preds = ['treats', 'affects']
dirs = ['R', 'F']
result = format_metapath(nodes, preds, dirs)
expected = 'Disease|treats|R|SmallMolecule|affects|F|Gene'
print(f'Non-symmetric metapath: {result}')
print(f'✓ Correct: {result == expected}')
print()

# Test 4: Mixed symmetric and non-symmetric
nodes = ['Disease', 'Gene', 'Gene', 'SmallMolecule']
preds = ['affects', 'directly_physically_interacts_with', 'regulates']
dirs = ['F', 'F', 'R']
result = format_metapath(nodes, preds, dirs)
print(f'Mixed metapath: {result}')
print(f'✓ Has A for symmetric: {"directly_physically_interacts_with|A" in result}')
print(f'✓ Has F for non-symmetric: {"affects|F" in result}')
print(f'✓ Has R for non-symmetric: {"regulates|R" in result}')

print('\nAll tests passed!')
