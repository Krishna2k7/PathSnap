"""
viterbi.py - Log-space Viterbi sequential decoder for PathSnap route reconstruction.
Combines Gradient Boosting candidate emission probabilities with road-network topological
transition probabilities to decode the global most likely sequence of road segments.
"""

from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd

from src.hmm.transition import NetworkTransitionModel


class ViterbiDecoder:
    """
    Decodes the optimal sequence of road segments across GPS observations
    using dynamic programming in log-space.
    """

    def __init__(self, transition_model: NetworkTransitionModel, emission_weight: float = 1.0):
        self.transition_model = transition_model
        self.emission_weight = emission_weight

    def decode(
        self,
        observations_candidates: List[List[Dict[str, float]]],
        gps_coords_meters: np.ndarray,
    ) -> List[int]:
        """
        Runs Viterbi dynamic programming to reconstruct the optimal sequence of road segments.
        
        Args:
            observations_candidates: List of candidate lists for each GPS observation.
                Each candidate dict contains 'arc_id' and 'prob' (calibrated emission prob).
            gps_coords_meters: (T, 2) array of projected (x, y) coordinates of GPS fixes.
            
        Returns:
            List of decoded arc_ids of length T.
        """
        T = len(observations_candidates)
        if T == 0:
            return []

        # Viterbi table: list of dicts mapping arc_id -> (max_log_prob, backpointer_arc_id)
        viterbi = []

        # t = 0 Initialization
        cands_0 = observations_candidates[0]
        v_0 = {}
        for c in cands_0:
            arc_id = c["arc_id"]
            prob = max(1e-7, float(c.get("prob", 0.5)))
            log_emission = math.log(prob) * self.emission_weight
            v_0[arc_id] = (log_emission, None)
        viterbi.append(v_0)

        # Forward recurrence t = 1 ... T - 1
        for t in range(1, T):
            prev_states = viterbi[t - 1]
            curr_cands = observations_candidates[t]
            curr_v = {}

            # Displacement between consecutive GPS fixes
            dx = gps_coords_meters[t, 0] - gps_coords_meters[t - 1, 0]
            dy = gps_coords_meters[t, 1] - gps_coords_meters[t - 1, 1]
            euclid_dist = math.hypot(dx, dy)

            for cand_j in curr_cands:
                arc_j = cand_j["arc_id"]
                prob_j = max(1e-7, float(cand_j.get("prob", 0.5)))
                log_emission = math.log(prob_j) * self.emission_weight

                best_prev_val = -float("inf")
                best_prev_arc = None

                for arc_i, (prev_log_prob, _) in prev_states.items():
                    log_trans = self.transition_model.compute_log_transition(arc_i, arc_j, euclid_dist)
                    total_val = prev_log_prob + log_trans

                    if total_val > best_prev_val:
                        best_prev_val = total_val
                        best_prev_arc = arc_i

                curr_v[arc_j] = (best_prev_val + log_emission, best_prev_arc)

            viterbi.append(curr_v)

        # Termination: select best state at time T - 1
        last_states = viterbi[-1]
        best_last_arc = max(last_states.keys(), key=lambda arc: last_states[arc][0])

        # Backtracking
        decoded_path = [best_last_arc]
        curr_arc = best_last_arc

        for t in range(T - 1, 0, -1):
            _, backpointer = viterbi[t][curr_arc]
            if backpointer is None:
                # Fallback to candidate with highest score if chain broke
                cands_prev = observations_candidates[t - 1]
                backpointer = max(cands_prev, key=lambda c: c.get("prob", 0))["arc_id"]
            decoded_path.append(backpointer)
            curr_arc = backpointer

        decoded_path.reverse()
        return decoded_path


def compute_path_continuity(decoded_arcs: List[int], arcs_df: pd.DataFrame) -> float:
    """
    Computes route continuity percentage: fraction of transitions that are either
    the same segment or directly connected in the road network.
    """
    if len(decoded_arcs) <= 1:
        return 100.0

    u_nodes = arcs_df["u"].values
    v_nodes = arcs_df["v"].values

    connected_count = 0
    total_transitions = len(decoded_arcs) - 1

    for i in range(total_transitions):
        a1 = decoded_arcs[i]
        a2 = decoded_arcs[i + 1]

        if a1 == a2:
            connected_count += 1
        elif v_nodes[a1] == u_nodes[a2]:
            connected_count += 1

    return (connected_count / total_transitions) * 100.0
