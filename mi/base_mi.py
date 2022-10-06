from abc import ABC, abstractmethod

class AbsMIMethod(ABC):
    @abstractmethod
    def mutual_information(self, T):
        """
        Input: 
            T: the output of one layer
        Output:
            I(X|T), I(Y|T)
        """
        raise NotImplementedError