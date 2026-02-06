"""Golden test graph for end-to-end integration tests.

This graph is carefully designed to exercise:
1. Hierarchical type aggregation (Gene -> BiologicalEntity -> NamedThing)
2. Predicate hierarchy (treats -> related_to)
3. Pseudo-type expansion (Gene+Protein contributes to both Gene and Protein)
4. Multiple paths contributing to same aggregated variant
5. Symmetric predicates (interacts_with uses direction 'A')

Graph Structure:
===============

Nodes (9 total):
  Gene_A          - pure Gene
  Gene_B          - pure Gene
  GeneProtein_Z   - PSEUDO-TYPE: Gene+Protein (has both Gene and Protein as leaf types)
  SmallMolecule_X - pure SmallMolecule
  SmallMolecule_Y - pure SmallMolecule
  Disease_P       - pure Disease
  Disease_Q       - pure Disease
  Protein_M       - pure Protein
  Protein_N       - pure Protein

Edges (12 total):
  affects (5 edges):
    Gene_A         --affects--> Disease_P
    Gene_A         --affects--> Disease_Q
    Gene_B         --affects--> Disease_P
    Protein_M      --affects--> Disease_P
    GeneProtein_Z  --affects--> Disease_Q   # Pseudo-type!

  treats (2 edges):
    SmallMolecule_X --treats--> Disease_P
    SmallMolecule_Y --treats--> Disease_Q

  interacts_with (3 edges, symmetric):
    Gene_A         --interacts_with--> Protein_M
    Gene_B         --interacts_with--> Protein_N
    GeneProtein_Z  --interacts_with--> Gene_B   # Pseudo-type!

  regulates (1 edge):
    Gene_A --regulates--> Gene_B

  associated_with (1 edge):
    Protein_N --associated_with--> Disease_Q
"""

import json
from pathlib import Path


# Node definitions
NODES = [
    {
        "id": "TEST:Gene_A",
        "name": "Gene A",
        "category": ["biolink:Gene", "biolink:BiologicalEntity", "biolink:NamedThing"],
    },
    {
        "id": "TEST:Gene_B",
        "name": "Gene B",
        "category": ["biolink:Gene", "biolink:BiologicalEntity", "biolink:NamedThing"],
    },
    {
        "id": "TEST:GeneProtein_Z",
        "name": "Gene/Protein Z (dual-type)",
        # This node has TWO leaf types - will become pseudo-type Gene+Protein
        "category": ["biolink:Gene", "biolink:Protein", "biolink:BiologicalEntity", "biolink:NamedThing"],
    },
    {
        "id": "TEST:SmallMolecule_X",
        "name": "Small Molecule X",
        "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"],
    },
    {
        "id": "TEST:SmallMolecule_Y",
        "name": "Small Molecule Y",
        "category": ["biolink:SmallMolecule", "biolink:ChemicalEntity", "biolink:NamedThing"],
    },
    {
        "id": "TEST:Disease_P",
        "name": "Disease P",
        "category": ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature", "biolink:NamedThing"],
    },
    {
        "id": "TEST:Disease_Q",
        "name": "Disease Q",
        "category": ["biolink:Disease", "biolink:DiseaseOrPhenotypicFeature", "biolink:NamedThing"],
    },
    {
        "id": "TEST:Protein_M",
        "name": "Protein M",
        "category": ["biolink:Protein", "biolink:BiologicalEntity", "biolink:NamedThing"],
    },
    {
        "id": "TEST:Protein_N",
        "name": "Protein N",
        "category": ["biolink:Protein", "biolink:BiologicalEntity", "biolink:NamedThing"],
    },
]

# Edge definitions
EDGES = [
    # affects edges (5)
    {"subject": "TEST:Gene_A", "predicate": "biolink:affects", "object": "TEST:Disease_P"},
    {"subject": "TEST:Gene_A", "predicate": "biolink:affects", "object": "TEST:Disease_Q"},
    {"subject": "TEST:Gene_B", "predicate": "biolink:affects", "object": "TEST:Disease_P"},
    {"subject": "TEST:Protein_M", "predicate": "biolink:affects", "object": "TEST:Disease_P"},
    {"subject": "TEST:GeneProtein_Z", "predicate": "biolink:affects", "object": "TEST:Disease_Q"},

    # treats edges (2)
    {"subject": "TEST:SmallMolecule_X", "predicate": "biolink:treats", "object": "TEST:Disease_P"},
    {"subject": "TEST:SmallMolecule_Y", "predicate": "biolink:treats", "object": "TEST:Disease_Q"},

    # interacts_with edges (3) - symmetric predicate
    {"subject": "TEST:Gene_A", "predicate": "biolink:interacts_with", "object": "TEST:Protein_M"},
    {"subject": "TEST:Gene_B", "predicate": "biolink:interacts_with", "object": "TEST:Protein_N"},
    {"subject": "TEST:GeneProtein_Z", "predicate": "biolink:interacts_with", "object": "TEST:Gene_B"},

    # regulates edge (1)
    {"subject": "TEST:Gene_A", "predicate": "biolink:regulates", "object": "TEST:Gene_B"},

    # associated_with edge (1)
    {"subject": "TEST:Protein_N", "predicate": "biolink:associated_with", "object": "TEST:Disease_Q"},
]


def write_golden_graph(workspace_path: Path):
    """Write the golden graph to nodes.jsonl and edges.jsonl in workspace."""
    workspace = Path(workspace_path)
    workspace.mkdir(parents=True, exist_ok=True)

    nodes_path = workspace / "nodes.jsonl"
    edges_path = workspace / "edges.jsonl"

    with open(nodes_path, 'w') as f:
        for node in NODES:
            f.write(json.dumps(node) + '\n')

    with open(edges_path, 'w') as f:
        for edge in EDGES:
            f.write(json.dumps(edge) + '\n')

    return nodes_path, edges_path


# Summary statistics for verification
GRAPH_STATS = {
    "num_nodes": len(NODES),
    "num_edges": len(EDGES),
    "num_pseudo_type_nodes": 1,  # GeneProtein_Z
    "edges_by_predicate": {
        "affects": 5,
        "treats": 2,
        "interacts_with": 3,
        "regulates": 1,
        "associated_with": 1,
    },
    "node_types": {
        "Gene": 2,           # Gene_A, Gene_B
        "Gene+Protein": 1,   # GeneProtein_Z (pseudo-type)
        "SmallMolecule": 2,  # SmallMolecule_X, SmallMolecule_Y
        "Disease": 2,        # Disease_P, Disease_Q
        "Protein": 2,        # Protein_M, Protein_N
    }
}
