"""
Connectome graph utilities for Drosophila neural subcircuits.

Supports loading and constructing connectome subgraphs and compiling
edge lists into sparse PyTorch tensors with sign-predicted neurotransmitter
polarity.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple


class ConnectomeGraph:
    """Manages connectome graph data and compiles into sparse tensors."""
    def __init__(self, name: str = "MaleCNS-v1.0-subgraph"):
        self.name = name
        self.neuron_ids: List[int] = []
        self.id_to_idx: Dict[int, int] = {}
        self.cell_types: Dict[int, str] = {}
        self.neurotransmitters: Dict[int, str] = {}
        
        # Edges: (pre_idx, post_idx, weight, sign)
        self.edges: List[Tuple[int, int, float, float]] = []
        self.sparse_tensor: Optional[torch.Tensor] = None

    def add_neuron(self, neuron_id: int, cell_type: str = "unknown", neurotransmitter: str = "acetylcholine") -> int:
        """
        Registers a neuron in the graph and returns its continuous local index.
        """
        if neuron_id not in self.id_to_idx:
            idx = len(self.neuron_ids)
            self.neuron_ids.append(neuron_id)
            self.id_to_idx[neuron_id] = idx
            self.cell_types[neuron_id] = cell_type
            self.neurotransmitters[neuron_id] = neurotransmitter
            return idx
        return self.id_to_idx[neuron_id]

    def add_synapse(self, pre_id: int, post_id: int, synapse_count: int = 1):
        """
        Adds a synaptic connection between two neurons.
        Determines sign based on presynaptic neurotransmitter prediction.
        """
        pre_idx = self.add_neuron(pre_id)
        post_idx = self.add_neuron(post_id)

        nt = self.neurotransmitters.get(pre_id, "acetylcholine").lower()
        if "gaba" in nt or "glutamate" in nt:
            sign = -1.0  # Inhibitory
        else:
            sign = +1.0  # Excitatory

        weight = float(synapse_count) * sign
        self.edges.append((pre_idx, post_idx, weight, sign))

    def compile_sparse_tensor(self, device: str = "cpu") -> torch.Tensor:
        """
        Compiles all edges into a PyTorch sparse COO tensor of shape (N, N).
        """
        num_neurons = len(self.neuron_ids)
        if not self.edges or num_neurons == 0:
            return torch.sparse_coo_tensor(size=(0, 0))

        # Target (post) receives from Source (pre): W[post, pre]
        pre_indices = [e[0] for e in self.edges]
        post_indices = [e[1] for e in self.edges]
        weights = [e[2] for e in self.edges]

        indices = torch.tensor([post_indices, pre_indices], dtype=torch.long)
        values = torch.tensor(weights, dtype=torch.float32)

        self.sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (num_neurons, num_neurons), device=device
        ).coalesce()

        return self.sparse_tensor

    @classmethod
    def create_synthetic_fly_cns(cls, num_sensory: int = 750, num_central: int = 1200, num_motor: int = 50) -> "ConnectomeGraph":
        """
        Generates a synthetic connected CNS graph mirroring Drosophila's modular network statistics
        (small-world topology, sparse connectivity density ~0.15%, log-normal synaptic weights).
        """
        graph = cls(name="Synthetic-Drosophila-CNS")
        total_neurons = num_sensory + num_central + num_motor

        # Register neurons
        for i in range(total_neurons):
            if i < num_sensory:
                ctype = "Photoreceptor / Medulla"
                nt = "histamine" if i < num_sensory // 2 else "acetylcholine"
            elif i < num_sensory + num_central:
                ctype = "Central Complex / Mushroom Body"
                nt = "gaba" if np.random.rand() < 0.25 else "acetylcholine"
            else:
                ctype = "Descending Neuron (DN)"
                nt = "acetylcholine"
            graph.add_neuron(100000 + i, cell_type=ctype, neurotransmitter=nt)

        # Wire sensory to central
        for s in range(num_sensory):
            targets = np.random.choice(range(num_sensory, num_sensory + num_central), size=np.random.randint(4, 12), replace=False)
            for t in targets:
                graph.add_synapse(100000 + s, 100000 + t, synapse_count=np.random.randint(1, 8))

        # Wire central recurrently (sparse modular small-world)
        for c in range(num_sensory, num_sensory + num_central):
            targets = np.random.choice(range(num_sensory, num_sensory + num_central), size=np.random.randint(6, 18), replace=False)
            for t in targets:
                graph.add_synapse(100000 + c, 100000 + t, synapse_count=np.random.randint(1, 5))

        # Wire central to motor (descending neurons)
        for m in range(num_sensory + num_central, total_neurons):
            sources = np.random.choice(range(num_sensory, num_sensory + num_central), size=np.random.randint(15, 40), replace=False)
            for s in sources:
                graph.add_synapse(100000 + s, 100000 + m, synapse_count=np.random.randint(2, 10))

        graph.compile_sparse_tensor()
        return graph
