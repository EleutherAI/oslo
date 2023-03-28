from abc import ABC

import torch


class ParamGenerator(ABC):
    def append(self, param: torch.nn.Parameter):
        pass

    def generate(self):
        pass

    def clear(self):
        pass


class OrderedParamGenerator(ParamGenerator):
    """OrderedParamGenerator

    Contain the order of parameters visited during runtime.
    """

    def __init__(self) -> None:
        self.param_visited_order = []

    def append(self, param: torch.nn.Parameter):
        """Append a parameter to the param_visited_order.

        Args:
            param (torch.nn.Parameter): A torch parameter.
        """
        self.param_visited_order.append(param)

    def generate(self):
        """Generate the parameter order.

        Yields:
            torch.nn.Parameter: A torch parameter.
        """
        visited_set = set()
        for p in self.param_visited_order:
            if p not in visited_set:
                yield p
            visited_set.add(p)
        del visited_set

    def is_empty(self) -> bool:
        """Check if the param_visited_order is empty.

        Returns:
            bool: True if the param_visited_order is empty.
        """
        return len(self.param_visited_order) == 0

    def clear(self):
        """Clear the param_visited_order."""
        self.param_visited_order = []
