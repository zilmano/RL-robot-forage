import numpy as np
import math

class ValueFunctionWithApproximation(object):
    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

class TileCodingApproximation(ValueFunctionWithApproximation):
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array,
                 num_of_items:int,
                 grid_size:int,
                 calc_tile_width = True):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maximum value for each dimension in state
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        assert state_high.ndim == state_low.ndim, "dimensions of both state boundaries should be the same"
        dim = state_low.size
        if calc_tile_width:
            # If this flag this enabled, set the number of tiles per dimension to 'num_tilings', and calculate the
            # width accordingly.
            auto_tile_width = np.zeros([dim])
            for d in range(0, dim):
                auto_tile_width[d] = num_tilings*(state_high[d] - state_low[d]) / ((num_tilings - 1) * num_tilings + 1)
            self.tile_width = auto_tile_width
        else:
            self.tile_width = tile_width
        fundamental_unit_width = self.tile_width/num_tilings
        self.num_of_tiles = dim*[0]
        for d in range(0,dim):
            self.num_of_tiles[d] = math.ceil((state_high[d]-state_low[d])/self.tile_width[d])+1
        self.state_low = state_low
        self.state_high = state_high
        self.dim = dim

        self.X_vector_size = num_tilings
        for d in range(0, dim):
            self.X_vector_size *= self.num_of_tiles[d]
        self.X_vector_size += num_of_items # The last 6 weights are for each of the items (1 if the the item is found)
        self.X_vector_size += grid_size
        self.num_tilings = num_tilings
        self.num_of_items = num_of_items
        self.tiling_weights = np.zeros(self.X_vector_size)
        self.tiling_limits = np.zeros([num_tilings,dim,max(*self.num_of_tiles,num_tilings)+1],dtype=np.float32)
        #offset = 1.12
        offset = 1
        for tiling_index in range(0,num_tilings):
            for d in range(0,dim):
                lower_bound = state_low[d]
                lower_bound -= tiling_index * fundamental_unit_width[d]*offset
                self.tiling_limits[tiling_index][d][0] = lower_bound
                for tilenum in range(1,self.num_of_tiles[d]+1):
                    lower_bound +=self.tile_width[d]
                    self.tiling_limits[tiling_index][d][tilenum] = lower_bound

    '''
    # Debug function
    def _set_init_weights(self):
        for i in range(0,self.num_tilings):
            for j in  range(0,self.num_of_tiles[0]):
                for k in range(0,self.num_tiles[1]):
                    self.tiling_weights[self._feature_indices_to_row_index((i,j,k))] = 0.5
            self.tiling_weights[self._feature_indices_to_row_index((i,3,3))] = 3
            self.tiling_weights[self._feature_indices_to_row_index((i,2,3))] = 2
    '''

    def _get_features(self, s, items_list, already_visited, debug=False):
        for d in range(0,self.dim):
            assert (s[d] <= self.state_high[d] and s[d] >= self.state_low[d]), "state value in dimension {} is out of bounds!".format(d)
        features = []

        x = np.zeros(self.X_vector_size)
        x[self.X_vector_size-self.grid_size-self.num_of_items:] = items_list
        x[self.X_vector_size-self.grid_size:] = already_visited
        for tiling_index in range(0,self.num_tilings):
            feature_indices = [tiling_index,]
            for dim_index in range(0,self.dim):
                dim_feature_index = None
                for tile_index in range(0,self.num_of_tiles[dim_index]):
                    if (s[dim_index] >= self.tiling_limits[tiling_index][dim_index][tile_index] and s[dim_index] < self.tiling_limits[tiling_index][dim_index][tile_index+1] ):
                        dim_feature_index = tile_index
                        break
                if s[dim_index] == self.tiling_limits[tiling_index][dim_index][-1]:
                    dim_feature_index = self.num_of_tiles[dim_index]-1
                elif dim_feature_index is None:
                    # sanity check
                    sys.exit("State in dimension {} is not covered by the tiling (out of bounds)")
                feature_indices.append(dim_feature_index)
            row_index = self._feature_indices_to_row_index(feature_indices)
            if debug:
                features.append(tuple(feature_indices))
                print("feature {} weights {} ".format(feature_indices, self.tiling_weights[row_index]))
            x[row_index] = 1

        return x,features

    def _feature_indices_to_row_index(self,feature_indices):
        row_index = 0
        prev_dim_size = 1
        for d, i in enumerate(reversed(feature_indices), 1):
            d = self.dim - d
            row_index += i * prev_dim_size
            prev_dim_size *= self.num_of_tiles[d]
        return row_index

    def __call__(self,s,items_list,already_visited):

        x,_ = self._get_features(s,items_list,already_visited)
        v = np.dot(self.tiling_weights,x)
        return v


    def update(self,alpha,G,s,items_list,already_visited):
        """
                Implement the update rule;
                w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

                input:
                    alpha: learning rate
                    G: TD-target
                    s_tau: target state for updating (yet, update will affect the other states)
                ouptut:
                    None
                """
        x = np.array(self._get_features(s,items_list,already_visited)[0])
        Vhat = self(s,items_list)
        self.tiling_weights += alpha*(G - Vhat) * x

        return None
