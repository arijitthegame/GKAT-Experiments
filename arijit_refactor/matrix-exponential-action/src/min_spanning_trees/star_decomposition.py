import networkx as nx
import math
from collections import deque   

'''
Add ball and cone cuts : important components for star decomposition. 
This code implements start decomposition for unweighted trees. The code can easily be generalized to weighted trees. m
Needs to add in the Dijstra's algorithm
'''


def get_ball(distances, radius):
    ball = [].
    d = 0
    while d <= radius:
        ball+=distances[d]
        d+=1
    return ball

'''
The ball cut function
'''

def ball_cut(G, dist_from_center, rho, delta, num_edges, source):
    radius = rho * delta
    c = math.log(num_edges+1, 2) / ((1- 2*delta)*rho)

    if radius >= 1:
        ball = get_ball(dist_from_center, radius)
        cut_size = nx.cut_size(G, ball)
        volume = len(G.edges(ball))
    else:
        ball = [source]
        cut_size = G.degree[source] #UNWEIGHTED
        volume = cut_size

    while cut_size > c*(volume+1):
        radius += 1
        ball+=dist_from_center[floor(radius)]
        cut_size = nx.cut_size(G, ball)
        volume = len(G.edges(ball))
    return radius, ball


def get_ideal(G, S, anchor, distances):
    ideal = [anchor]
    # BFS from anchor
    seen = {anchor : 0}                  # create dict level (number of hops) when seen in BFS
    level = 0                  # the current level
    nextlevel = set(G.adj[anchor])     # dict of nodes to check at next level
    temp=set()

    while nextlevel: #TODO: Test this while loop. Maybe buggy
        thislevel = nextlevel  # advance to next level
        nextlevel = set()         # and start a new list (fringe)
        for v in thislevel:
            if v not in seen:
                if v in S: continue
                seen[v] = level  # set the level of vertex v
                d = distances[v]
                if d >= level:
                    ideal.append(v)
                    nextlevel.update(G.adj[v])  # add neighbors of v
                else: temp.update(G.adj[v])
            for u in temp:
                if u not in seen:
                    seen[u] = level+1
            temp = set()
        level += 1
    return ideal

# grows the ideal by radius l
def get_cone(G, l, ideal):
    cone = ideal
    while l > 0:
        for item in list(nx.node_boundary(G, cone)):
            cone.append(item)
        l-=1
    return cone

def cone_properties(G, cone, num_edges):
    cone_subgraph_size = G.subgraph(cone).size()
    cone_cut_size = nx.cut_size(G,cone)
    volume = cone_subgraph_size + cone_cut_size

    if cone_subgraph_size == 0: mu = (volume+1)*log(num_edges+1, 2)
    else: mu = (volume)*log(num_edges/cone_subgraph_size)
    return mu, cone_cut_size


def cone_cut(G, x, l, L, S, num_edges, distances):
    r = l
    ideal = get_ideal(G, S, x, distances)
    cone = ideal if r == 0 else get_cone(G, r, ideal)
    mu, cone_cut_size = cone_properties(G, cone, num_edges)
    while cone_cut_size > mu/(L-l):
        for item in list(nx.node_boundary(G, cone)):
            cone.append(item)
        mu, cone_cut_size = cone_properties(G, cone, num_edges)
        r+=1
    return r, cone


'''
Putting them all together to get the star decomposition

'''


def distances_to_center(G, center):
    seen = set()
    level = 0
    dists = {}
    nextlevel = {center}

    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        vs = []
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(G.adj[v])
                vs.append(v)
        if vs: dists[level] = vs
        level += 1
    return dists, level-2

def boundary_neighbors(G, node_boundary):
    neighbors = set()
    for node in node_boundary:
        neighbors.update(G.adj[node])
    return neighbors

# returns dictionary of shortest path lengths to the node boundary. For weighted trees we will need Dijstra's algorithm.
def contracted_distances(G, node_boundary):
    neighs = boundary_neighbors(G, node_boundary)
    H = G.copy()
    H.remove_nodes_from(node_boundary)
    v = 's'
    H.add_node(v)
    for neigh in neighs:
        if neigh in node_boundary: 
            continue
        else: 
            H.add_edge(v, neigh)
    return nx.single_source_shortest_path_length(H, v)

# decomposes graph into cones
def cone_decomp(H, node_boundary, Delta, num_edges):
    node_boundary_distances = contracted_distances(H, node_boundary)
    cones, anchors = [],[]
    while node_boundary:
        for node in node_boundary: anchor = node; break
        r, cone = cone_cut(H, anchor, 0, Delta, node_boundary, num_edges, node_boundary_distances)
        for node in cone:
            H.remove_node(node)
            if node in node_boundary: 
                node_boundary.remove(node)
        cones += [(cone, anchor)]
        anchors.append(anchor)
    return cones, anchors

def get_bridges(G, center, anchors, cutoff):
    visited = {center}
    queue = deque([(center, 0, G.neighbors(center))])
    while queue:
        parent, depth_now, children = queue[0]     ## length of shortest path is depth_now
        try:
            child = next(children)
            if child in anchors:
                anchors.remove(child)
                yield (child, parent)
            if child not in visited:
                visited.add(child)
                if depth_now < cutoff:
                    queue.append((child, depth_now + 1, G.neighbors(child)))
        except StopIteration:
            queue.popleft()

def star_decomp(G, center, delta, eps, num_edges):
    H = G.copy()
    distances, radius = distances_to_center(G, center)
    ball_radius, ball = ball_cut(G, distances, radius, delta, num_edges, center)
    node_boundary = set(nx.node_boundary(G, ball))
    H.remove_nodes_from(ball)
    cones, anchors = cone_decomp(H, node_boundary, eps*radius/2, num_edges)
    bridges = list(get_bridges(G, center, anchors, math.floor(ball_radius)))
    partitions = [(ball, center)] + cones
    return partitions, bridges



#TODO: ADD UNIT TESTS AND FINISH DEBUGGING
