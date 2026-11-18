"""
transition.py - Road-network graph transition modeling for HMM map matching.
Computes network topological shortest path distances and transition probabilities
between candidate road segments following the Newson & Krumm formulation.
"""

from typing import Dict, Tuple
import math
import numpy as np
import pandas as pd
import networkx as nx


class NetworkTransitionModel:
    """
    Builds a directed road graph and calculates topological transition probabilities
    between candidate road segments.
    """

    def __init__(
        self,
        node_x: np.ndarray,
        node_y: np.ndarray,
        arcs_df: pd.DataFrame,
        beta: float = 15.0,
        unconnected_penalty: float = -15.0,
    ):
        """
        Args:
            node_x: Node x-coordinates in meters.
            node_y: Node y-coordinates in meters.
            arcs_df: DataFrame with 'u' (from_node) and 'v' (to_node).
            beta: Scale parameter for difference between network and Euclidean distance.
            unconnected_penalty: Log-probability penalty when no topological path exists.
        """
        self.node_x = node_x
        self.node_y = node_y
        self.u = arcs_df["u"].values
        self.v = arcs_df["v"].values
        self.beta = beta
        self.unconnected_penalty = unconnected_penalty

        # Compute metric length of every arc
        dx = self.node_x[self.v] - self.node_x[self.u]
        dy = self.node_y[self.v] - self.node_y[self.u]
        self.arc_lengths = np.hypot(dx, dy)

        # Build directed graph
        self.graph = nx.DiGraph()
        for arc_id in range(len(arcs_df)):
            u_node = int(self.u[arc_id])
            v_node = int(self.v[arc_id])
            length = float(self.arc_lengths[arc_id])
            self.graph.add_edge(u_node, v_node, weight=length, arc_id=arc_id)

        # Distance lookup cache to optimize repeated shortest-path queries
        self._path_cache: Dict[Tuple[int, int], float] = {}

    def get_network_distance(self, u_from: int, v_to: int, cutoff_dist: float = 500.0) -> float:
        """Computes shortest path distance in road graph from node u_from to node v_to."""
        if u_from == v_to:
            return 0.0

        pair = (u_from, v_to)
        if pair in self._path_cache:
            return self._path_cache[pair]

        try:
            # Dijkstra shortest path with weight='weight'
            dist = nx.shortest_path_length(self.graph, source=u_from, target=v_to, weight="weight")
            self._path_cache[pair] = dist
            return dist
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            self._path_cache[pair] = float("inf")
            return float("inf")

    def compute_log_transition(
        self, arc_prev: int, arc_curr: int, euclid_dist: float
    ) -> float:
        """
        Computes log P(arc_curr | arc_prev) given observed Euclidean displacement.
        """
        # If the vehicle stayed on the exact same road segment
        if arc_prev == arc_curr:
            # High probability if vehicle is driving along the segment
            return -abs(self.arc_lengths[arc_prev] * 0.1 - euclid_dist) / self.beta

        # Vehicle transitioned from arc_prev to arc_curr
        exit_node = int(self.v[arc_prev])
        enter_node = int(self.u[arc_curr])

        if exit_node == enter_node:
            # Directly connected segments
            net_dist = float(self.arc_lengths[arc_curr])
        else:
            path_len = self.get_network_distance(exit_node, enter_node)
            if math.isinf(path_len):
                return self.unconnected_penalty
            net_dist = path_len + float(self.arc_lengths[arc_curr])

        # Newson-Krumm exponential transition penalty in log-space
        diff = abs(net_dist - euclid_dist)
        return -diff / self.beta
