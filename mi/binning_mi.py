from mi.base_mi import AbsMIMethod

class BinningMI(AbsMIMethod):
    def __init__(self, X, y, bins):
        self.X = X
        self.y = y
        self.bins = bins

        self.px, self.py = self.extract_initial_probability(X, y)

    def extract_initial_probability(self, X, y):
        """
        Input:
            X: the training data
            y: the training label
        Output:
            P(X), P(y)
        """
        pass
    