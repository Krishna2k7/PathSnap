"""
PathSnap Hidden Markov Model (HMM) & Viterbi Sequential Route Decoder
=====================================================================
Modular implementation of HMM-based sequential route reconstruction for map-matching.

Mathematical Formulation:
--------------------------
1. Hidden States:
   For each GPS observation t in {0, ..., N-1}, the hidden states are the K candidate
   road segments C_t = {c_{t, 1}, ..., c_{t, K_t}}.

2. Emission Probabilities:
   P(z_t | c_{t, i}) combines:
   a) The supervised ML classifier's probability: p_model(y_{t,i}=1 | x_{t,i})
   b) The physical GPS measurement error: Gaussian N(d_perp; 0, sigma^2)
   In log-space:
     log E(c_{t, i}) = alpha * log(p_model + eps) + (1 - alpha) * [-0.5 * (d_perp / sigma)^2 - log(sqrt(2*pi)*sigma)]

3. Transition Probabilities:
   Between candidate c_{t-1, i} = (u_{prev}, v_{prev}) and candidate c_{t, j} = (u, v):
   a) Same Arc (c_{t-1, i} == c_{t, j}):
      Vehicle remains on segment: log T = self_transition_log_prob
   b) Directly Connected Junction (v_{prev} == u):
      Discrepancy: D = |seg_len - d_GPS|
      Newson-Krumm exponential model: log T = -log(beta) - (D / beta)
   c) Short Multi-Hop Path (2 to max_hops via local graph BFS):
      log T = -log(beta) - (|d_route - d_GPS| / beta) - k * hop_penalty
   d) Disconnected / Missing Candidate Fallback:
      log T = log_epsilon (bounded floor, e.g. -18.0) to prevent -inf/NaN during dropouts.

4. Viterbi Dynamic Programming:
   Forward Pass:
     V_t(j) = log E(c_{t, j}) + max_i [ V_{t-1}(i) + log T(c_{t-1, i} -> c_{t, j}) ]
     B_t(j) = argmax_i [ V_{t-1}(i) + log T(c_{t-1, i} -> c_{t, j}) ]
   Backward Trace:
     s^*_{N-1} = argmax_j V_{N-1}(j)
     s^*_t = B_{t+1}(s^*_{t+1})
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import numpy as np
import pandas as pd

from data_processing.spatial_utils import (
    get_transformer,
    project_coords,
)


class ViterbiDecoder:
    """
    HMM & Viterbi Sequential Route Decoder.
    """

    def __init__(
        self,
        sigma: float = 10.0,
        beta: float = 25.0,
        alpha: float = 0.65,
        self_transition_log_prob: float = 0.5,
        hop_penalty: float = 2.0,
        log_epsilon: float = -18.0,
        max_hops: int = 3,
    ):
        self.sigma = sigma
        self.beta = beta
        self.alpha = alpha
        self.self_transition_log_prob = self_transition_log_prob
        self.hop_penalty = hop_penalty
        self.log_epsilon = log_epsilon
        self.max_hops = max_hops

        # Precompute constants
        self._gauss_norm_log = -math.log(math.sqrt(2.0 * math.pi) * self.sigma)
        self._exp_norm_log = -math.log(self.beta)

    def load_road_graph(self, arcs_path: str, nodes_path: str) -> Dict[str, Any]:
        """
        Loads road network and precomputes metric lengths and directed adjacency.
        """
        arcs_df = pd.read_csv(arcs_path, sep=r"\s+", header=None, names=["u", "v"], dtype=np.int64)
        nodes_df = pd.read_csv(nodes_path, sep=r"\s+", header=None, names=["lon", "lat"], dtype=np.float64)

        u_arr = arcs_df["u"].values
        v_arr = arcs_df["v"].values
        lon_arr = nodes_df["lon"].values
        lat_arr = nodes_df["lat"].values

        # Local metric projection
        mean_lon = float(np.mean(lon_arr))
        mean_lat = float(np.mean(lat_arr))
        transformer, _ = get_transformer(mean_lon, mean_lat)
        nx, ny = project_coords(lon_arr, lat_arr, transformer)

        p1x, p1y = nx[u_arr], ny[u_arr]
        p2x, p2y = nx[v_arr], ny[v_arr]
        lengths = np.sqrt((p2x - p1x) ** 2 + (p2y - p1y) ** 2)

        # Build adjacency: node -> list of (neighbor_node, arc_id, length)
        adj = defaultdict(list)
        for arc_id, (u_i, v_i, l_i) in enumerate(zip(u_arr, v_arr, lengths)):
            adj[u_i].append((v_i, arc_id, float(l_i)))

        return {
            "u": u_arr,
            "v": v_arr,
            "lengths": lengths,
            "adj": adj,
            "nx": nx,
            "ny": ny,
            "n_arcs": len(arcs_df),
            "memo_hops": {},
        }

    def _find_shortest_path_hops(
        self, start_node: int, target_node: int, graph_data: Dict[str, Any]
    ) -> Optional[Tuple[int, float]]:
        """
        Fast BFS to find shortest hop distance and path length up to max_hops.
        Memoized for microsecond lookups.
        """
        if start_node == target_node:
            return 0, 0.0

        cache_key = (start_node, target_node)
        if cache_key in graph_data["memo_hops"]:
            return graph_data["memo_hops"][cache_key]

        adj = graph_data["adj"]
        queue = deque([(start_node, 0, 0.0)])
        visited = {start_node}

        while queue:
            curr, hops, dist_accum = queue.popleft()
            if hops >= self.max_hops:
                continue
            for neighbor, _, arc_len in adj.get(curr, []):
                new_dist = dist_accum + arc_len
                if neighbor == target_node:
                    res = (hops + 1, new_dist)
                    graph_data["memo_hops"][cache_key] = res
                    return res
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, hops + 1, new_dist))

        graph_data["memo_hops"][cache_key] = None
        return None

    def compute_emission_log_prob(self, p_model: float, d_perp: float) -> float:
        """
        Blended emission log-probability combining ML model score and Gaussian distance error.
        """
        p_safe = max(1e-6, min(1.0 - 1e-6, p_model))
        model_term = math.log(p_safe)

        d_perp_safe = max(0.0, float(d_perp))
        gauss_term = self._gauss_norm_log - 0.5 * (d_perp_safe / self.sigma) ** 2

        return self.alpha * model_term + (1.0 - self.alpha) * gauss_term

    def compute_transition_log_prob(
        self,
        prev_arc: int,
        curr_arc: int,
        d_gps: float,
        graph_data: Dict[str, Any]
    ) -> float:
        """
        Evaluates topological transition log-probability.
        """
        # Case 1: Same Arc
        if prev_arc == curr_arc:
            return self.self_transition_log_prob

        u_arr = graph_data["u"]
        v_arr = graph_data["v"]
        lengths = graph_data["lengths"]

        v_prev = u_arr[prev_arc] if prev_arc < 0 else v_arr[prev_arc]
        u_curr = u_arr[curr_arc]
        curr_len = lengths[curr_arc]

        # Case 2: Directly Connected Outgoing Arc (Junction)
        if v_prev == u_curr:
            discrepancy = abs(curr_len - d_gps)
            return self._exp_norm_log - (discrepancy / self.beta)

        # Case 3: Short Multi-Hop Reachable
        hop_res = self._find_shortest_path_hops(v_prev, u_curr, graph_data)
        if hop_res is not None:
            hops, path_dist = hop_res
            total_dist = path_dist + curr_len
            discrepancy = abs(total_dist - d_gps)
            return self._exp_norm_log - (discrepancy / self.beta) - (hops * self.hop_penalty)

        # Case 4: Disconnected / Missing Candidate Fallback Floor
        return self.log_epsilon

    def decode(
        self,
        candidates_df: pd.DataFrame,
        graph_data: Dict[str, Any]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Performs Viterbi dynamic programming decoding across candidate sequences.
        
        Parameters
        ----------
        candidates_df : pd.DataFrame
            Rows containing 'obs_idx', 'candidate_arc_id', 'perp_dist_raw', 'pred_prob', and optionally 't'.
        graph_data : Dict[str, Any]
            Precomputed road network graph structure.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame]
            Array of decoded arc_ids (length N_obs) and decoded observation DataFrame.
        """
        # Group candidates by observation index
        grouped = candidates_df.groupby("obs_idx")
        obs_indices = sorted(grouped.groups.keys())
        n_obs = len(obs_indices)

        if n_obs == 0:
            return np.array([], dtype=np.int64), pd.DataFrame()

        # Step 0: Initialize DP tables
        # List of candidate dictionaries per observation
        cand_list_per_t = []
        for obs_i in obs_indices:
            sub = grouped.get_group(obs_i)
            cand_list_per_t.append({
                "arc_ids": sub["candidate_arc_id"].values,
                "d_perps": sub["perp_dist_raw"].values,
                "pred_probs": sub["pred_prob"].values,
                "t": sub["t"].values[0] if "t" in sub.columns else float(obs_i),
            })

        # Calculate straight-line GPS displacements if coordinates are available
        # Default displacement ~ 10-15m for 1Hz driving if not present
        d_gps_arr = np.full(n_obs, 12.0)
        for t in range(1, n_obs):
            dt = cand_list_per_t[t]["t"] - cand_list_per_t[t - 1]["t"]
            d_gps_arr[t] = max(1.0, 12.0 * max(1.0, dt))

        # Forward DP: V_t[k] = max log-prob ending in candidate k at step t
        viterbi_scores = []
        backpointers = []

        # t = 0: Initialize with emission probabilities
        c0 = cand_list_per_t[0]
        v0 = np.array([
            self.compute_emission_log_prob(p, d)
            for p, d in zip(c0["pred_probs"], c0["d_perps"])
        ], dtype=np.float64)
        viterbi_scores.append(v0)
        backpointers.append(np.zeros(len(v0), dtype=int))

        # t = 1 ... N-1: Recurrence
        for t in range(1, n_obs):
            prev_cands = cand_list_per_t[t - 1]["arc_ids"]
            curr_cands = cand_list_per_t[t]["arc_ids"]
            curr_perps = cand_list_per_t[t]["d_perps"]
            curr_probs = cand_list_per_t[t]["pred_probs"]
            d_gps = d_gps_arr[t]
            v_prev = viterbi_scores[t - 1]

            n_curr = len(curr_cands)
            n_prev = len(prev_cands)

            v_curr = np.zeros(n_curr, dtype=np.float64)
            bp_curr = np.zeros(n_curr, dtype=int)

            for j in range(n_curr):
                c_j = curr_cands[j]
                em_j = self.compute_emission_log_prob(curr_probs[j], curr_perps[j])

                # Maximize over previous candidates i
                best_score = -np.inf
                best_bp = 0
                for i in range(n_prev):
                    c_i = prev_cands[i]
                    trans_ij = self.compute_transition_log_prob(c_i, c_j, d_gps, graph_data)
                    total_p = v_prev[i] + trans_ij
                    if total_p > best_score:
                        best_score = total_p
                        best_bp = i

                v_curr[j] = best_score + em_j
                bp_curr[j] = best_bp

            viterbi_scores.append(v_curr)
            backpointers.append(bp_curr)

        # Backward Trace: find optimal path
        best_end_idx = int(np.argmax(viterbi_scores[-1]))
        decoded_indices = [best_end_idx]

        for t in range(n_obs - 1, 0, -1):
            prev_best = backpointers[t][decoded_indices[-1]]
            decoded_indices.append(prev_best)

        decoded_indices.reverse()

        # Map to actual arc_ids
        decoded_arcs = np.array([
            cand_list_per_t[t]["arc_ids"][idx]
            for t, idx in enumerate(decoded_indices)
        ], dtype=np.int64)

        decoded_df = pd.DataFrame({
            "obs_idx": obs_indices,
            "predicted_arc_id": decoded_arcs,
            "viterbi_log_prob": [viterbi_scores[t][idx] for t, idx in enumerate(decoded_indices)],
        })

        return decoded_arcs, decoded_df


def decode_trip(
    trip_df: pd.DataFrame,
    nodes_path: str,
    arcs_path: str,
    params: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Standalone inference function ready for direct import by FastAPI backend.
    """
    decoder = ViterbiDecoder(**(params or {}))
    graph_data = decoder.load_road_graph(arcs_path, nodes_path)
    _, decoded_df = decoder.decode(trip_df, graph_data)
    return decoded_df
