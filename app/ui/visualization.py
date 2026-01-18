"""
Visualization for the Language Exchange Matchmaking System.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, TYPE_CHECKING

import networkx as nx
import matplotlib.pyplot as plt

try:
    from ..config import LANG_HE, LANG_EN
    from ..matching import compute_raw_score
    from ..matching.cooldown import pair_key
    from ..persistence import reconcile_state
except ImportError:
    from config import LANG_HE, LANG_EN
    from matching import compute_raw_score
    from matching.cooldown import pair_key
    from persistence import reconcile_state

if TYPE_CHECKING:
    from ..models.state import AppState


def show_bipartite_graph(state: "AppState") -> None:
    """
    Visualize only ACTIVE edges:
    - Proposals in state.proposals (both users should be has_offer)
    - Matches (mutual current_partner_id, both status matched)

    Edge label uses proposal.score_at_offer when available for proposals,
    and compute_raw_score for matches.
    """
    reconcile_state(state)

    hebrew_ids = [uid for uid, u in state.users.items() if u.native_language == LANG_HE]
    english_ids = [uid for uid, u in state.users.items() if u.native_language == LANG_EN]

    if not hebrew_ids and not english_ids:
        print("\n(No users to display.)")
        return

    G = nx.Graph()
    for uid in hebrew_ids:
        G.add_node(uid, bipartite=0)
    for uid in english_ids:
        G.add_node(uid, bipartite=1)

    edges: List[Tuple[str, str]] = []
    edge_labels: Dict[Tuple[str, str], str] = {}
    edge_colors: List[str] = []
    edge_widths: List[float] = []

    def add_edge(u_id: str, v_id: str, w: float, color: str, width_scale: float = 1.0) -> None:
        G.add_edge(u_id, v_id)
        edges.append((u_id, v_id))
        edge_labels[(u_id, v_id)] = f"{w:.2f}"
        edge_colors.append(color)
        width = max(0.8, min(6.0, 0.8 + width_scale * math.log1p(abs(w))))
        edge_widths.append(width)

    # Proposals (gray)
    for _, prop in state.proposals.items():
        u1 = state.users.get(prop.user1_id)
        u2 = state.users.get(prop.user2_id)
        if not u1 or not u2:
            continue
        if u1.status != "has_offer" or u2.status != "has_offer":
            continue
        add_edge(u1.user_id, u2.user_id, float(prop.score_at_offer), color="gray", width_scale=1.0)

    # Matches (red) - add once
    seen = set()
    for uid, u in state.users.items():
        if u.status != "matched" or not u.current_partner_id:
            continue
        pid = u.current_partner_id
        if pid not in state.users:
            continue
        v = state.users[pid]
        if v.status != "matched" or v.current_partner_id != u.user_id:
            continue
        key = pair_key(u.user_id, v.user_id)
        if key in seen:
            continue
        seen.add(key)
        w = compute_raw_score(state, u, v)
        add_edge(u.user_id, v.user_id, float(w), color="red", width_scale=1.4)

    if not edges:
        print("\nNo active proposals or matches to display.")
        return

    pos: Dict[str, Tuple[float, float]] = {}
    for i, uid in enumerate(sorted(hebrew_ids)):
        pos[uid] = (0.0, float(i))
    for j, uid in enumerate(sorted(english_ids)):
        pos[uid] = (1.0, float(j))

    plt.figure(figsize=(9, 6))
    nx.draw_networkx_nodes(G, pos, nodelist=hebrew_ids, node_color="lightblue", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=english_ids, node_color="lightgreen", node_size=900)
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color=edge_colors, width=edge_widths)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title("Active proposals (gray) and matches (red)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
