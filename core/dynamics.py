"""
Neural dynamics engine simulating graded potentials and leaky integrate-and-fire (LIF)
units with sparse synaptic propagation.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union


class SparseNeuralEngine:
    """
    Simulates a population of biological neurons with sparse directed connectivity.
    Can run on CPU or CUDA via PyTorch sparse tensors or NumPy.
    """
    def __init__(self, 
                 num_neurons: int,
                 tau_m: float = 0.02,         # 20ms membrane time constant
                 v_rest: float = 0.0,
                 v_thresh: float = 1.0,
                 v_reset: float = 0.0,
                 is_spiking: Optional[np.ndarray] = None,
                 device: str = "cpu"):
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # True if neuron emits discrete spikes; False if graded continuous potential
        if is_spiking is None:
            # Default: graded potentials for local interneurons, spiking for last 10% (DNs)
            self.is_spiking = torch.zeros(num_neurons, dtype=torch.bool, device=self.device)
        else:
            self.is_spiking = torch.tensor(is_spiking, dtype=torch.bool, device=self.device)

        # State variables
        self.v = torch.full((num_neurons,), v_rest, dtype=torch.float32, device=self.device)
        self.spikes = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)
        self.refractory_timer = torch.zeros(num_neurons, dtype=torch.float32, device=self.device)
        self.refractory_period = 0.003  # 3ms refractory limit (~330 Hz max firing rate)

        # Synaptic weight matrix (Sparse PyTorch CSR or COO tensor)
        self.adj_indices: Optional[torch.Tensor] = None
        self.adj_values: Optional[torch.Tensor] = None
        self.sparse_weight_matrix: Optional[torch.Tensor] = None

    def set_connectivity(self, source_indices: np.ndarray, target_indices: np.ndarray, weights: np.ndarray):
        """
        Builds the sparse connectivity matrix W where W[i, j] is weight from j to i.
        """
        src = torch.tensor(source_indices, dtype=torch.long, device=self.device)
        tgt = torch.tensor(target_indices, dtype=torch.long, device=self.device)
        w = torch.tensor(weights, dtype=torch.float32, device=self.device)

        indices = torch.stack([tgt, src])  # (2, N_edges) - target receives from source
        self.sparse_weight_matrix = torch.sparse_coo_tensor(
            indices, w, (self.num_neurons, self.num_neurons), device=self.device
        ).coalesce()

    def update_weights(self, source_indices: np.ndarray, target_indices: np.ndarray, new_weights: np.ndarray):
        """
        Fast dynamic update of specific plastic synaptic connections (e.g. KC -> MBON).
        """
        self.set_connectivity(source_indices, target_indices, new_weights)

    def step(self, external_current: Union[np.ndarray, torch.Tensor], dt: float = 0.005) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates one timestep of neural dynamics:
        V(t+dt) = V(t) + (dt/tau) * [-(V(t) - V_rest) + I_syn + I_ext]
        """
        if isinstance(external_current, np.ndarray):
            i_ext = torch.tensor(external_current, dtype=torch.float32, device=self.device)
        else:
            i_ext = external_current

        # Compute synaptic current I_syn = W * S(t)
        # S(t) is discrete spikes for spiking neurons, or graded activation for non-spiking
        activity = torch.where(self.is_spiking, self.spikes, torch.relu(self.v - self.v_rest))

        if self.sparse_weight_matrix is not None:
            i_syn = torch.sparse.mm(self.sparse_weight_matrix, activity.unsqueeze(1)).squeeze(1)
        else:
            i_syn = torch.zeros_like(self.v)

        # Decay factor
        alpha = dt / self.tau_m

        # Update membrane potential
        not_refractory = self.refractory_timer <= 0.0
        dv = alpha * (-(self.v - self.v_rest) + i_syn + i_ext)
        self.v = torch.where(not_refractory, self.v + dv, self.v)

        # Decrement refractory timers
        self.refractory_timer = torch.clamp(self.refractory_timer - dt, min=0.0)

        # Spiking evaluation
        spiked = (self.v >= self.v_thresh) & self.is_spiking & not_refractory
        self.spikes = spiked.float()
        
        # Reset spiked neurons
        self.v = torch.where(spiked, torch.tensor(self.v_reset, device=self.device), self.v)
        self.refractory_timer = torch.where(spiked, torch.tensor(self.refractory_period, device=self.device), self.refractory_timer)

        # Return voltages and activities as numpy arrays for HUD/motor integration
        return self.v.cpu().numpy(), activity.cpu().numpy()
