import pennylane.numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def make_graph(
    n_vertices=8,
    prob_aresta=0.5,
    com_peso=True,
    peso_min=1,
    peso_max=10,
    seed=42
):
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n_vertices))

    # gera arestas aleatórias
    for i in range(n_vertices):
        for j in range(i + 1, n_vertices):
            if rng.random() < prob_aresta:
                if com_peso:
                    w = int(rng.integers(peso_min, peso_max + 1))
                else:
                    w = 1
                G.add_edge(i, j, weight=w)

    # garante conectividade total do grafo
    componentes = list(nx.connected_components(G))

    while len(componentes) > 1:
        comp1 = list(componentes[0])
        comp2 = list(componentes[1])

        v1 = int(rng.choice(comp1))
        v2 = int(rng.choice(comp2))

        if com_peso:
            w = int(rng.integers(peso_min, peso_max + 1))
        else:
            w = 1

        G.add_edge(v1, v2, weight=w)

        componentes = list(nx.connected_components(G))

    return G
def index_to_bitstring(idx, nbits, reverse_bits=False):
    bits = [(idx >> (nbits - 1 - k)) & 1 for k in range(nbits)]
    if reverse_bits:
        bits = bits[::-1]
    return bits


def cut_value(bits, G):
    total = 0.0
    for u, v, data in G.edges(data=True):
        w = float(data.get("weight", 1.0))
        total += w * (bits[u] ^ bits[v])
    return total


def brute_force_maxcut(G, reverse_bits=False):
    n = G.number_of_nodes()
    best_value = -1.0
    best_bits = None
    best_idx = None

    for idx in range(2**n):
        bits = index_to_bitstring(idx, n, reverse_bits=reverse_bits)
        val = cut_value(bits, G)
        if val > best_value:
            best_value = val
            best_bits = bits
            best_idx = idx

    return best_value, best_bits, best_idx

def draw_graph(G, title="Grafo"):
    pos = nx.spring_layout(G, seed=123)
    weights = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(6, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=700,
        font_weight="bold"
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title(title)
    plt.tight_layout()
    #plt.show()


def draw_cut_solution(G, bits, title="Solução MaxCut"):
    pos = nx.spring_layout(G, seed=123)
    weights = nx.get_edge_attributes(G, "weight")

    colors = ["tab:blue" if b == 0 else "tab:orange" for b in bits]

    plt.figure(figsize=(6, 5))
    nx.draw(
        G, pos,
        with_labels=True,
        node_color=colors,
        node_size=700,
        font_weight="bold"
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)
    plt.title(title)
    plt.tight_layout()
    #plt.show()