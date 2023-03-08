from torch.optim import Optimizer


class BaseOptimizerWrapper(Optimizer):
    def __init__(self, optim: Optimizer):
        """
        Wrap an optimizer with the base optimizer.

        Args:
            optim (Optimizer): The optimizer to be wrapped.
        """
        self.optim = optim

    @property
    def param_groups(self):
        """
        Return the parameter groups.
        """
        return self.optim.param_groups

    @property
    def defaults(self):
        """
        Return the default values for the optimizer.
        """
        return self.optim.defaults

    def add_param_group(self, *args, **kwargs):
        """
        Add a parameter group to the optimizer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return self.optim.add_param_group(*args, **kwargs)

    def step(self, *args, **kwargs):
        """
        Take a step for the optimizer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        return self.optim.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """
        Zero the gradients of the optimizer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.optim.zero_grad(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        """
        Load the state dict for the optimizer.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.optim.load_state_dict(*args, **kwargs)

    def state_dict(self):
        """
        Return the state dict of the optimizer.
        """
        return self.optim.state_dict()
