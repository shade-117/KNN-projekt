from abc import ABC, abstractmethod


class BaseDataLoader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    # def initialize(self):
    #     # self.opt = opt
    #     pass

    def load_data(self):
        return None
