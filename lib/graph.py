import numpy as np
from utility import BasicQueue
from utility import logmsg
from utility import SortedList
import math

class GridGraphWithItems:
    def __init__(self,m,n,items_num,items_prob_matrix):
        self.amount_vertices = n*m  # amount of vertices
        self.n = n
        self.m = m
        self.adj_mat = np.zeros([self.amount_vertices,self.amount_vertices])

        for item in range(0,items_num):
            assert items_prob_matrix[item].sum() == 1, "Prob matrix for item {} is wrong. Probabilites for item " \
                                                       "should some to 1"
        self.items_prob_mat = items_prob_matrix
        self.items_num = items_num
        self._init_grid()
        self.not_visited_vertices = np.ones([self.amount_vertices])

    def _get_row_col_from_vertix_num(self,vertix):
        return ((vertix % self.m), (vertix / self.m))

    def _get_vertix_num_from_row_col(self, row, col):
        return int(col * self.m + row)

    def _init_grid(self):
        # Build transition matrix
        for vertix in range(0, self.amount_vertices):
            if vertix >= self.m:
                self.adj_mat[vertix][vertix - self.m] = 1
            if vertix % self.m:
                self.adj_mat[vertix][vertix - 1] = 1
            if vertix < (self.n - 1) * self.m:
                self.adj_mat[vertix][vertix + self.m ] = 1
            if (vertix+1) % self.m:
                self.adj_mat[vertix][vertix + 1] = 1

    def _get_manhattan_distance(self,vertix1,vertix2):
        row1, col1 = self._get_row_col_from_vertix_num(vertix1)
        row2, col2 = self._get_row_col_from_vertix_num(vertix2)
        return abs(row2-row1) + abs(col2-col1)

        # adds a edge linking "src" in "dest" with a "cost"
    def add_edge(self, src, dest, cost):
        # checks if the edge already exists
        assert cost >= 0, "Cannot assign negative cost"
        assert self.adj_mat[src,dest] == 0 and self.adj_mat[dest,src] == 0, "Edge between '{}' and '{} already exist. " \
                                                                          "Can't add new edge".format(src,dest)
        self.adj_mat[src,dest] = cost
        self.adj_mat[dest,src] = cost

    # checks if exists a edge linking "src" in "dest"
    def exists_edge(self, src, dest):
        return bool(self.adj_mat[src, dest])

    def get_approximate_best_path(self,start_vertex):
        path = [[start_vertix, 0]]
        nn, steps = self.get_nearest_neighbor_aggregate_prob(start_vertex)
        while nn is not None:
            nn, steps = self.get_nearest_neighbor_aggregate_prob(nn)
            path.append((nn, steps))

        path_exp_cost = self.calc_path_cost(path)
        return (path, path_exp_cost)

    def get_nearest_neighbor_aggregate_prob(self,vertex):
        vertices = [vertex]
        self.items_prob_mat[0] += 1
        self.not_visited_vertices[vertex] = 0
        nn = self.get_nearest_neighbor_aggregate_prob_impl(vertices,0)
        self.items_prob_mat[0] -= 1
        return nn

    def get_nearest_neighbor_aggregate_prob_impl(self,vertices,steps):
        weighted_neighbors = np.zeros(self.amount_vertices)
        neighbor_vec = np.zeros(self.amount_vertices)
        if len(vertices) == 0 or np.all(self.not_visited_vertices == 0):
            return (None,None)
        steps += 1
        for vertex in vertices:
            neighbor_vec =  np.logical_or(neighbor_vec, self.adj_mat[vertex])

        nextLevelVertices = list(np.flatnonzero(neighbor_vec))

        for i in range(0,self.items_num):
            weighted_neighbors += self.items_prob_mat[i]*neighbor_vec*self.not_visited_vertices

        if weighted_neighbors.max() > 0:
            best_neighbors = np.argwhere(weighted_neighbors == weighted_neighbors.max()).flatten()
            man_distances = np.array([self._get_manhattan_distance(vertex,neighbor) for neighbor in best_neighbors])
            best_neighbor = best_neighbors[man_distances.argmin()]
            self.not_visited_vertices[best_neighbor] = 0
            return (int(best_neighbor),steps)
        else:
            return self.get_nearest_neighbor_aggregate_prob_impl(nextLevelVertices,steps)

    def calc_path_cost(self,path):
        exp_cost = 0
        prev_all_found_prob = 0
        if (len(path) < self.amount_vertices):
            return math.inf
        for k in range(self.items_num,self.amount_vertices):
            cost = sum([cost for v,cost in path[0:k]])
            items_found_probs = [sum([self.items_prob_mat[i][v] for v,cost in path[0:k]]) for i in range(0,self.items_num)]
            all_items_found_prob = (1-prev_all_found_prob) * np.prod(np.array(items_found_probs))
            exp_cost += cost * all_items_found_prob
            prev_all_found_prob += all_items_found_prob
        print(prev_all_found_prob)
        return exp_cost

    '''
    def getRandomPaths(self, max_size):

        random_paths, list_vertices = [], list(self.vertices)

        initial_vertice = random.choice(list_vertices)
        if initial_vertice not in list_vertices:
            print('Error: initial vertice %d not exists!' % initial_vertice)
            sys.exit(1)

        list_vertices.remove(initial_vertice)
        list_vertices.insert(0, initial_vertice)

        for i in range(max_size):
            list_temp = list_vertices[1:]
            random.shuffle(list_temp)
            list_temp.insert(0, initial_vertice)

            if list_temp not in random_paths:
                random_paths.append(list_temp)

        return random_paths
    '''