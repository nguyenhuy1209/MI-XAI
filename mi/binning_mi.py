import numpy as np

from mi.base_mi import AbsMIMethod

class BinningMI(AbsMIMethod):
    def __init__(self, X, y, bins):
        self.X = X
        self.y = y
        self.bins = bins

        plane_params = dict(zip(['pys1', 'p_YgX', 'b1', 'b', 'len_unique_a', 'unique_inverse_x', 'unique_inverse_y', 'pxs'], 
                            self.extract_initial_probability(X, np.array(y).astype(np.float))))

        plane_params['bins'] = bins
        plane_params['label'] = y
        
        self.plane_params = plane_params


    def extract_initial_probability(self, x, label):
        """
        Calculate the probabilities of the given layer_output and labels p(x), p(y) and (y|x)
        Input:
            X: the training layer_output
            y: the training label
        Output:
            P(X), P(y)
        """

        # calculate pys (what for?)
        pys = np.sum(label, axis=0) / float(label.shape[0])

        #  calculate pxs
        b = np.ascontiguousarray(x).view(np.dtype((np.void, x.dtype.itemsize * x.shape[1])))
        # b = np.ascontiguousarray(x)
        unique_array, unique_indices, unique_inverse_x, unique_counts = \
            np.unique(b, return_index=True, return_inverse=True, return_counts=True)
        unique_a = x[unique_indices]
        b1 = np.ascontiguousarray(unique_a).view(np.dtype((np.void, unique_a.dtype.itemsize * unique_a.shape[1])))
        pxs = unique_counts / float(np.sum(unique_counts))

        # calculate p(y|x)
        p_y_given_x = []
        for i in range(0, len(unique_array)):
            indexs = unique_inverse_x == i
            py_x_current = np.mean(label[indexs, :], axis=0)
            p_y_given_x.append(py_x_current)
        p_y_given_x = np.array(p_y_given_x).T

        # calculate pys1
        b_y = np.ascontiguousarray(label).view(np.dtype((np.void, label.dtype.itemsize * label.shape[1])))
        unique_array_y, unique_indices_y, unique_inverse_y, unique_counts_y = \
            np.unique(b_y, return_index=True, return_inverse=True, return_counts=True)
        pys1 = unique_counts_y / float(np.sum(unique_counts_y))

        len_unique_a = len(unique_a)
        return pys1, p_y_given_x, b1, b, len_unique_a, unique_inverse_x, unique_inverse_y, pxs
    
    def mutual_information(self, layer_output):
        """ 
        Given the outputs T of one layer of an NN, calculate MI(X;T) and MI(T;Y)
        
        params:
            layer_output - a 3d numpy array, where 1st dimension is training objects, second - neurons
        
        returns:
            IXT, ITY - mutual information
        """
            
        information = self.calc_information_for_layer_with_other(layer_output)
        return information['local_IXT'], information['local_ITY']


    def calc_information_for_layer_with_other(self, layer_output):
        local_IXT, local_ITY = self.calc_information_sampling(layer_output)

        params = {}
        params['local_IXT'] = local_IXT
        params['local_ITY'] = local_ITY
        return params

    def calc_information_sampling(self, layer_output):
        bins = bins.astype(np.float32)
        # bins = stats.mstats.mquantiles(np.squeeze(layer_output.reshape(1, -1)), np.linspace(0,1, num=num_of_bins))
        # hist, bin_edges = np.histogram(np.squeeze(layer_output.reshape(1, -1)), normed=True)
        
        # discretized layer T -> transform continuous to discrete variable
        digitized = bins[np.digitize(np.squeeze(layer_output.reshape(1, -1)), bins) - 1].reshape(len(layer_output), -1)
        
        # dont know why
        b2 = np.ascontiguousarray(digitized).view(
            np.dtype((np.void, digitized.dtype.itemsize * digitized.shape[1])))

        # calculating P(t_i)
        unique_array, unique_inverse_t, unique_counts = \
            np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
        p_ts = unique_counts / float(sum(unique_counts))
        PXs, PYs = np.asarray(self.plane_params.pxs).T, np.asarray(self.plane_params.pys1).T
        
        # P(X), P(Y) and P(t_i) to calculate MI 
        local_IXT, local_ITY = self.calc_information_from_mat(PXs, PYs, p_ts, digitized, self.plane_params.unique_inverse_x, self.plane_params.unique_inverse_y,
                                                        unique_array)
        return local_IXT, local_ITY

    def calc_entropy_for_specipic_t(self, current_ts, px_i):
        """Calc entropy for specipic t"""
        b2 = np.ascontiguousarray(current_ts).view(
            np.dtype((np.void, current_ts.dtype.itemsize * current_ts.shape[1])))
        _, _, unique_counts = \
            np.unique(b2, return_index=False, return_inverse=True, return_counts=True)
        p_current_ts = unique_counts / float(sum(unique_counts))
        p_current_ts = np.asarray(p_current_ts, dtype=np.float64).T
        H2X = px_i * (-np.sum(p_current_ts * np.log2(p_current_ts)))
        return H2X

    def calc_condtion_entropy(self, px, t_data, unique_inverse_x):
        # Condition entropy of t given x
        # px: P(X) or P(Y)
        # t_data: discreted T
        # for i in range(px.shape[0]): # for every unique X?
        H2X_array = np.array([self.calc_entropy_for_specipic_t(t_data[unique_inverse_x == i, :], px[i]) for i in range(px.shape[0])])
        H2X = np.sum(H2X_array)
        return H2X

    def calc_information_from_mat(self, px, py, ps2, data, unique_inverse_x, unique_inverse_y):
        """Calculate the MI based on binning of the data"""
        H2 = -np.sum(ps2 * np.log2(ps2))
        H2X = self.calc_condtion_entropy(px, data, unique_inverse_x)
        H2Y = self.calc_condtion_entropy(py.T, data, unique_inverse_y)
        IY = H2 - H2Y
        IX = H2 - H2X
        return IX, IY