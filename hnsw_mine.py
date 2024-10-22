import numpy as np
from heapq import heappush, heappop, heapify, heapreplace, nlargest, nsmallest
from math import log2
from random import random
from operator import itemgetter


class HNSW_mine:
    def __init__(self, distance_type, m=5, ef=200, m0=None, heuristic=True, vectorized=False):
        self.data = []
        self._m = m  # Number of bi-directional links
        self._ef = ef  # Size of the dynamic candidate list
        self._m0 = 2 * m if m0 is None else m0  # Max connections at level 0
        self._level_mult = 1 / log2(m)
        self._graphs = []  # Hierarchical graph layers
        self._enter_point = None  # Entry point in the graph

        # Select distance function
        if distance_type == "l2":
            self.distance_func = self._l2_distance
        else:
            raise ValueError('Invalid distance type! Choose "l2".')

        # Vectorized distance functions
        if vectorized:
            self.distance = self._single_distance
            self.vectorized_distance = self.distance_func
        else:
            self.distance = self.distance_func
            self.vectorized_distance = self._vectorized_distance

    def _l2_distance(self, a, b):
        return np.linalg.norm(a - b)

    def _single_distance(self, x, y):
        return self.distance_func(x, [y])[0]

    def _vectorized_distance(self, x, ys):
        return [self.distance_func(x, y) for y in ys]

    def add(self, elem, ef=None):
        if ef is None:
            ef = self._ef

        idx = len(self.data)
        self.data.append(elem)

        # Determine level for the new element
        level = int(-log2(random()) * self._level_mult) + 1

        if self._enter_point is not None:
            current_point = self._enter_point
            current_dist = self.distance(elem, self.data[current_point])

            # Search for closest neighbor at higher levels
            for layer in reversed(self._graphs[level:]):
                current_point, current_dist = self._search_layer_ef1(elem, current_point, current_dist, layer)
        
            ep = [(-current_dist, current_point)]
            for layer_level, layer in enumerate(reversed(self._graphs[:level])):
                max_neighbors = self._m if layer_level != 0 else self._m0

                # Search and connect neighbors at the current layer
                ep = self._search_layer(elem, ep, layer, ef)
                # ep = [(-dist, idx) for idx, dist in ep]
                layer[idx] = {}
                self.heuristic(layer[idx], ep, max_neighbors, layer, heap=True)

                # Add backlinks
                for neighbor_idx, dist in layer[idx].items():
                    self.heuristic(layer[neighbor_idx], (idx, dist), max_neighbors, layer)
        else:
            # Initialize graphs if this is the first element
            self._graphs.append({idx: {}})
            self._enter_point = idx

        # Extend graphs if necessary
        for _ in range(len(self._graphs), level):
            self._graphs.append({idx: {}})
            self._enter_point = idx

    def _search_layer_ef1(self, q, entry_point, dist_to_entry, layer):
        visited = set()
        candidates = [(dist_to_entry, entry_point)]
        best_point = entry_point
        best_dist = dist_to_entry
        visited.add(entry_point)

        while candidates:
            dist, current = heappop(candidates)
            if dist > best_dist:
                break

            neighbors = [n for n in layer[current] if n not in visited]
            visited.update(neighbors)
            neighbor_dists = self.vectorized_distance(q, [self.data[n] for n in neighbors])

            for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
                if neighbor_dist < best_dist:
                    best_point = neighbor
                    best_dist = neighbor_dist
                    heappush(candidates, (neighbor_dist, neighbor))

        return best_point, best_dist
            
    def search(self, q, k=1, ef=10, level=0, return_observed=True, entry_points=None):
        graphs = self._graphs
        point = self._enter_point
        for layer in reversed(graphs[level:]):
            point, dist = self.beam_search(layer, q=q, k=1, eps=[point], ef=1)[0]

        return self.beam_search(graph=graphs[level], q=q, k=k, eps=[point], ef=ef, return_observed=return_observed)

    def beam_search(self, graph, q, k, eps, ef, ax=None, marker_size=20, return_observed=False):
        '''
        graph – the layer where the search is performed
        q - query
        k - number of closest neighbors to return
        eps – entry points [vertex_id, ..., vertex_id]
        ef – size of the beam
        observed – if True returns the full of elements for which the distance were calculated
        returns – a list of tuples [(vertex_id, distance), ... , ]
        '''
        # Priority queue: (negative distance, vertex_id)
        candidates = []
        visited = set()  # set of vertex used for extending the set of candidates
        observed = dict() # dict: vertex_id -> float – set of vertexes for which the distance were calculated

        if ax:
            ax.scatter(x=q[0], y=q[1], s=marker_size, color='red', marker='^')
            ax.annotate('query', (q[0], q[1]))

        # Initialize the queue with the entry points
        for ep in eps:
            dist = self.distance_func(q, self.data[ep])
            heappush(candidates, (dist, ep))
            observed[ep] = dist

        while candidates:
            # Get the closest vertex (furthest in the max-heap sense)
            dist, current_vertex = heappop(candidates)

            if ax:
                ax.scatter(x=self.data[current_vertex][0], y=self.data[current_vertex][1], s=marker_size, color='red')
                ax.annotate( len(visited), self.data[current_vertex] )

            # check stop conditions #####
            observed_sorted = sorted( observed.items(), key=lambda a: a[1] )
            # print(observed_sorted)
            ef_largets = observed_sorted[ min(len(observed)-1, ef-1 ) ]
            # print(ef_largets[0], '<->', -dist)
            if ef_largets[1] < dist:
                break
            #############################

            # Add current_vertex to visited set
            visited.add(current_vertex)

            # Check the neighbors of the current vertex
            for neighbor in graph[current_vertex]:
                if neighbor not in observed:
                    dist = self.distance_func(q, self.data[neighbor])
                    # if neighbor not in visited:
                    heappush(candidates, (dist, neighbor))
                    observed[neighbor] = dist
                    if ax:
                        ax.scatter(x=self.data[neighbor][0], y=self.data[neighbor][1], s=marker_size, color='yellow')
                        # ax.annotate(len(visited), (self.data[neighbor][0], self.data[neighbor][1]))
                        ax.annotate(len(visited), self.data[neighbor])

        # Sort the results by distance and return top-k
        observed_sorted =sorted( observed.items(), key=lambda a: a[1] )
        if return_observed:
            return observed_sorted
        return observed_sorted[:k]

    def _search_layer(self, q, ep, layer, ef):
        visited = set()
        candidates = [(-dist, idx) for dist, idx in ep]
        heapify(candidates)
        visited.update(idx for _, idx in ep)

        while candidates:
            dist, current = heappop(candidates)
            if dist > -ep[0][0]:
                break

            neighbors = [n for n in layer[current] if n not in visited]
            visited.update(neighbors)
            neighbor_dists = self.vectorized_distance(q, [self.data[n] for n in neighbors])

            for neighbor, neighbor_dist in zip(neighbors, neighbor_dists):
                mdist = -neighbor_dist
                if len(ep) < ef:
                    heappush(candidates, (neighbor_dist, neighbor))
                    heappush(ep, (mdist, neighbor))
                if mdist > ep[0][0]:
                    heappush(candidates, (neighbor_dist, neighbor))
                    heapreplace(ep, (mdist, neighbor))

        return ep

    def heuristic(self, neighbors, candidates, m, layer, heap=False):
        neighbor_dicts = [layer[idx] for idx in neighbors]

        def prioritize(idx, dist):
            proximity = any(nd.get(idx, float('inf')) < dist for nd in neighbor_dicts)
            return proximity, dist, idx

        if heap:
            candidates = nsmallest(m, (prioritize(idx, -mdist) for mdist, idx in candidates))
            unchecked = m - len(neighbors)
            candidates_to_add = candidates[:unchecked]
            candidates_to_check = candidates[unchecked:]

            if candidates_to_check:
                to_remove = nlargest(len(candidates_to_check), (prioritize(idx, dist) for idx, dist in neighbors.items()))
            else:
                to_remove = []

            for _, dist, idx in candidates_to_add:
                neighbors[idx] = dist

            for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zip(candidates_to_check, to_remove):
                if (p_old, d_old) <= (p_new, d_new):
                    break
                del neighbors[idx_old]
                neighbors[idx_new] = d_new
        else:
            idx, dist = candidates
            candidates = [prioritize(idx, dist)]
            if len(neighbors) < m:
                neighbors[idx] = dist
            else:
                max_idx, max_val = max(neighbors.items(), key=itemgetter(1))
                if dist < max_val:
                    del neighbors[max_idx]
                    neighbors[idx] = dist

    def __getitem__(self, idx):
        for layer in self._graphs:
            if idx in layer:
                yield from layer[idx].items()
            else:
                return

