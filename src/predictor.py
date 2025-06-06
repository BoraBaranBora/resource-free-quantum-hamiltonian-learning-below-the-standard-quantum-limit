import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
        activation_fn=nn.Tanh,
        ignore_input=False,
        randomize_input=False,
        device=None
    ):
        super(Predictor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.ignore_input = ignore_input
        self.randomize_input = randomize_input

        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_size, hidden_layers[0]))
        layers.append(activation_fn())
        # Hidden layers
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(activation_fn())
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_size))

        self.network = nn.Sequential(*layers)

        # Determine device
        if device is None:
            try:
                self.device = next(self.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.to(self.device)

    def forward(self, batch_size=None, x=None):
        """
        Forward pass for the Predictor model.

        If `ignore_input` or `randomize_input` is set, the model will override the input tensor `x`.
        If `batch_size` is provided, it will generate a batch of default input tensors.
        """
        device = next(self.parameters()).device

        if self.ignore_input:
            # Generate a default input tensor for the model
            dim = int(((-1 + (1 + 8 * self.input_size) ** 0.5) / 2))  # Solve n(n+1)/2 = input_size
            base_input = torch.ones(1, self.input_size, device=device) * (2**4 / dim) ** 2
            if batch_size is not None:
                x = base_input.repeat(batch_size, 1)
            else:
                x = base_input.repeat(1, 1)

        elif self.randomize_input:
            # Generate random noise as input
            if batch_size:
                x = torch.randn(batch_size, self.input_size, device=device)
            else:
                x = torch.randn(1, self.input_size, device=device)

        elif x is None:
            # Raise an error if no input is provided and no override mode is enabled
            raise ValueError("Input is required unless `ignore_input` or `randomize_input` is enabled.")
        else:
            # Ensure provided x is on correct device
            x = x.to(device)

        # Pass through the network
        output = self.network(x)
        return output

    def set_ignore_input(self, ignore: bool):
        self.ignore_input = ignore

    def set_randomize_input(self, randomize: bool):
        self.randomize_input = randomize
