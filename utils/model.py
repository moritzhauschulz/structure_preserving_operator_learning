import torch 
import torch.nn as nn
from .deeponet import DeepONet
from utils.utils import FullFourierStrategy, FullFourierNormStrategy, FullFourierAvgNormStrategy


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x * torch.sigmoid(x)
    

class DeepONetWithGrad(DeepONet):
    def forward_with_grad(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        
        # Ensure x_func requires gradients
        x_func.requires_grad_(True)
        
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        
        x = self.multi_output_strategy.call(x_func, x_loc)
        
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        
        # Compute the gradient of the output with respect to x_func
        grad_outputs = torch.ones_like(x)
        gradients = torch.autograd.grad(outputs=x, inputs=x_func, grad_outputs=grad_outputs, create_graph=True)[0]
        
        return x, gradients



class CustomDeepONet(nn.Module):
    
    def __init__(self, hidden_branch=[128, 128, 128, 128], hidden_trunk=[128, 128, 128, 128], branch_in=3, trunk_in=1, num_outputs=1, activation=Swish, init='glorot'):
        super(DeepONet, self).__init__()
        
        self.branch_in = branch_in
        self.trunk_in = trunk_in
        self.num_outputs = num_outputs

        #construct branch net
        layers_branch_list = []
        layers_branch = [branch_in] + hidden_branch
        for (i, layer) in enumerate(layers_branch[:-1]):
            layers_branch_list.append(nn.Linear(layer, layers_branch[i+1]))
            if not i == len(layers_branch):
                layers_branch_list.append(activation())
        # layers_branch_list.append(nn.Linear(layers_branch[-1], num_outputs))
        self.branch_net = nn.Sequential(
            *layers_branch_list
        )

        #construct trunk net
        layers_trunk_list = []
        layers_trunk = [trunk_in] + hidden_trunk
        for (i, layer) in enumerate(layers_trunk[:-1]):
            layers_trunk_list.append(nn.Linear(layer, layers_trunk[i+1]))
            layers_trunk_list.append(activation())
        # layers_trunk_list.append(nn.Linear(layers_trunk[-1], num_outputs))
        self.trunk_net = nn.Sequential(
            *layers_trunk_list
        )

        #bias
        self.b = nn.Parameter(torch.zeros(1))

        # apply initialization
        if init == "glorot":
            self.apply(self.glorot_initialization)

    def glorot_initialization(self, layer):
        """
        Applies Glorot Initialization to the weights of the given layer.
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # Apply Glorot initialization to weights
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # Initialize biases to zero


    def forward(self, input):
        u = input[0]
        y = input[1]

        b = self.branch_net(u)
        t = self.trunk_net(y)

        out = torch.sum(b * t, dim = 1, keepdim=True)
        out = out + self.b
        

        return out
    
class c(nn.Module):
    def __init__(self, args, hidden_layers, activation=Swish, num_outputs=1, num_inputs=1, init='glorot', strategy='Fourier'):
        super(FullFourier, self).__init__()
        self.hidden_layers = hidden_layers
        activation_dict = {'swish': Swish, 'relu': nn.ReLU(), 'tanh': nn.Tanh()}
        self.activation = activation_dict[activation]
        self.num_outputs = num_outputs
        self.init =init

        self.layers_list = []
        self.layers_list.append(nn.Linear(num_inputs, hidden_layers[0]))
        self.layers_list.append( self.activation())
        for (i, layer) in enumerate(hidden_layers[:-1]):
            self.layers_list.append(nn.Linear(layer, hidden_layers[i+1]))
            self.layers_list.append( self.activation())
        self.layers_list.append(nn.Linear(hidden_layers[-1], num_outputs))
        self.layers = nn.Sequential(
            *self.layers_list
        )

        if strategy == 'FullFourier':
            self.strategy = FullFourierStrategy(args, self)
        elif strategy == 'FullFourierNorm':
            self.strategy = FullFourierNormStrategy(args, self)
        elif strategy == 'FullFourierAvgNorm':
            self.strategy = FullFourierAvgNormStrategy(args, self)

        # apply initialization
        if init == "glorot":
            self.apply(self.glorot_initialization)

    def glorot_initialization(self, layer):
        """
        Applies Glorot Initialization to the weights of the given layer.
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)  # Apply Glorot initialization to weights
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)  # Initialize biases to zero
    
    def forward(self, input,x=None, y=None):
        input_fourier = torch.fft.fftn(input, dim=[-1,-2])
        input_fourier = torch.cat((input_fourier.real, input_fourier.imag), dim=-1)
        out = self.layers(input)
        return self.strategy.call(out, x, y)

        
    

    


    






    