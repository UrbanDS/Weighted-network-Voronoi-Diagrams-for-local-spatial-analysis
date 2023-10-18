'''
A library for computing network-constrained directional statistics of dynamic LISAs.
This library relies on the pysal module spatial_dynamics.directional for computing
the directional statistics, and lincs for computing network-constrained spatial 
weight matrix.

The code style is intentionally made similar with the module lincs for consistency.

'''
import pysal.spatial_dynamics.directional as direc
import numpy as np
from lincs import node_weights, edgepoints_from_network, dist_weights

def net_rose(network, eventIdx_t0, eventIdx_t1, weight, dist=None, k=8, permutations=0):
    """
    Calculation of rose diagram for local indicators of spatial association in network space, 
    see pysal.spatial_dynamics.directional for further examples

    Parameters
    ----------
    network: a clean network where each edge has up to three attributes:
             Its length, an event variable, and a base variable
    eventIdx_t0: integer
               an index for the event variable at t0
    eventIdx_t1: integer
               an index for the event variable at t1               
    weight: string
            type of binary spatial weights
            two options are allowed: Node-based, Distance-based
    k: int
       number of circular sectors in rose diagram

    permutations: int
       number of random spatial permutations for calculation of pseudo
       p-values

    Returns
    -------

    results: dictionary (keys defined below)

    counts:  array (k,1)
        number of vectors with angular movement falling in each sector

    cuts: array (k,1)
        intervals defining circular sectors (in radians)

    random_counts: array (permutations,k)
        counts from random permutations

    pvalues: array (kx1)
        one sided (upper tail) pvalues for observed counts

    """
    w, edges, e, edges_geom = None, None, None, []
    if weight == 'Node-based':
        w, edges = node_weights(network, attribute=True)
        n = len(edges)
        e = np.zeros([n,2])
        for edge in edges:
            edges_geom.append(edges[edge][0])
            e[edge] = [edges[edge][eventIdx_t0], edges[edge][eventIdx_t1]]
        w.id_order = edges.keys()
    elif dist is not None:
        id2edgepoints, id2attr, edge2id = edgepoints_from_network(network, attribute=True)
        for n1 in network:
            for n2 in network[n1]:
                network[n1][n2] = network[n1][n2][0]
        w, edges = dist_weights(network, id2edgepoints, edge2id, dist)
        n = len(id2attr)
        e = np.zeros(n)
        for edge in id2attr:
            edges_geom.append(edges[edge])
            e[edge] = [id2attr[edge][eventIdx_t0 - 1], id2attr[edge][eventIdx_t1 - 1]]
        w.id_order = id2attr.keys()
        
    return direc.rose(e, w, k, permutations)