from matplotlib import cm, colors
from matplotlib import pyplot as plt
import numpy as np
import networkx as nx
from chmm_actions import forwardE

## A Few utility functions
def generate_markov_seq(trans_matrix, state_t0, seq_length):
    states = range(trans_matrix.shape[0])
    seq = np.empty(seq_length, dtype=int)
    seq[0] = state_t0
    for i in range(1, seq_length):
        seq[i] = np.random.choice(
            states, 
            p=trans_matrix[seq[i-1]]
        )
    return seq

def plot_graph(
    chmm, x, a, output_file,
    cmap=cm.Spectral,
    label_nodes=True,
    pos=None,
    kwargs_fig=None,
    kwargs_nxdraw=None, 
    kwargs_nxdrawedgelabel=None,
):
    states = chmm.decode(x, a)[1]
    
    v = np.unique(states)
    T = chmm.C[:, v][:, :, v]
    A = T.sum(0)
    A /= A.sum(1, keepdims=True)

    # Create NetworkX graph from adjacency matrix
    G = nx.DiGraph()
    G.add_nodes_from(range(len(v)))
    for i in range(len(v)):
        for j in range(len(v)):
            if A[i, j] > 0:
                G.add_edge(i, j, weight=A[i, j])

    # Node labels and colors (same logic)
    node_labels = np.arange(x.max() + 1).repeat(chmm.n_clones)[v]
    colors = [cmap(nl/node_labels.max()) for nl in node_labels]

    #pos = {i: coord for i, coord in enumerate(layout)}

    # Create figure with controlled axes
    fig, ax = plt.subplots()

    if not label_nodes:
        node_labels = None
    # Draw network
    nx.draw(
        G, pos, ax=ax,
        node_color=colors,
        #labels={i: str(label) for i, label in enumerate(v)},
        labels=node_labels,
        **kwargs_nxdraw
    )   


    # Create edge labels dictionary from weights
    edge_labels = {
        (u, v): f"{data['weight']:.{2}f}"
        for u, v, data in G.edges(data=True)}
    
    # Draw edge labels with positioning
    nx.draw_networkx_edge_labels(
        G, 
        pos,
        edge_labels=edge_labels,
        ax=ax,
        **kwargs_nxdrawedgelabel
    )
    
    return fig, ax


def get_mess_fwd(chmm, x, pseudocount=0.0, pseudocount_E=0.0):
    n_clones = chmm.n_clones
    E = np.zeros((n_clones.sum(), len(n_clones)))
    last = 0
    for c in range(len(n_clones)):
        E[last : last + n_clones[c], c] = 1
        last += n_clones[c]
    E += pseudocount_E
    norm = E.sum(1, keepdims=True)
    norm[norm == 0] = 1
    E /= norm
    T = chmm.C + pseudocount
    norm = T.sum(2, keepdims=True)
    norm[norm == 0] = 1
    T /= norm
    T = T.mean(0, keepdims=True)
    log2_lik, mess_fwd = forwardE(
        T.transpose(0, 2, 1), E, chmm.Pi_x, chmm.n_clones, x, x*0 , store_messages=True
    )
    return mess_fwd, E, T


def place_field(mess_fwd, rc, clone):
    assert mess_fwd.shape[0] == rc.shape[0] and clone < mess_fwd.shape[1]
    field = np.zeros(rc.max(0) + 1)
    count = np.zeros(rc.max(0) + 1, int)
    for t in range(mess_fwd.shape[0]):
        r, c = rc[t]
        field[r, c] += mess_fwd[t, clone]
        count[r, c] += 1
    count[count == 0] = 1
    return field / count
