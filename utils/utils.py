import matplotlib.pyplot as plt
from itertools import cycle
from typing import List, Union
import numpy as np
import torch
from utils.data import exact_soliton


from deepxde.nn.deeponet_strategy import DeepONetStrategy

class CustomStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of the branch net is not evenly divisible by {self.net.num_outputs}."
            )
        if layer_sizes_branch[-1] / self.net.num_outputs != layer_sizes_trunk[-1]:
            raise AssertionError(
                f"Output size of the trunk net does not equal to {layer_sizes_branch[-1] // self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )
    
class OrthonormalBranchNormalTrunkRegStrategy(CustomStrategy):
    def call(self, x_func, x_loc):
        branch_out = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            branch_out_ = branch_out[:, shift : shift + size]
            x = self.net.merge_branch_trunk(branch_out_, x_loc, i)
            xs.append(x)
            shift += size
        output = self.net.concatenate_outputs(xs)
        return output, (branch_out, x_loc)
    

class OrthonormalBranchNormalTrunkStrategy(CustomStrategy):
    def call(self, x_func, x_loc):
        torch.autograd.set_detect_anomaly(True)
        branch_out_in = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs)
        # #multiply first coordinate by omega
        # branch_out[:, 0, :] = branch_out[:, 0, :] * x_func[:, 2].unsqueeze(1)
        # perform QR decomposition on last two dimensions
        q, r = torch.linalg.qr(branch_out_in)
        if not torch.allclose(torch.eye(q.shape[2]), torch.bmm(q.permute(0, 2, 1), q), atol=1e-6):
            raise AssertionError("Q is not orthonormal")
        #invert implicit multiplication of first coordinate by omega
        branch_out = q.clone()
        branch_out[:, 0, :] *= q[:, 0, :] / x_func[:, 2].unsqueeze(1)
        #transform back to original shape
        branch_out = branch_out.reshape(-1, branch_out.shape[2] * self.net.num_outputs)
        norms = torch.norm(x_loc, p=2, dim=1, keepdim=True)
        #setnorms to 1 if zero
        zero_mask = (norms == 0)
        norms = norms + zero_mask.float()  #TODO: Find a better solution to this!
        x_loc = x_loc / norms
        if not torch.allclose(torch.ones(x_loc.shape[0])[~zero_mask.squeeze(1)], torch.norm(x_loc, p=2, dim=1, keepdim=True)[~zero_mask.squeeze(1)]):
            raise AssertionError("Trunk output is not normalized")
        if (zero_mask.squeeze(1)).sum() > branch_out.shape[0] * 0.01:
            raise AssertionError(f"Had to exclude more than 1% ({(zero_mask.squeeze(1)).sum()/branch_out.shape[0]}) of the batch at the normalizations stage")
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            branch_out_ = branch_out[:, shift : shift + size]
            x = self.net.merge_branch_trunk(branch_out_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs), (branch_out, x_loc) 
    
class QRStrategy(CustomStrategy):
    def call(self, x_func, x_loc):
        # print(x_func.shape)
        torch.autograd.set_detect_anomaly(True)
        branch_out_in = self.net.branch(x_func)
        # print(branch_out_in.shape)
        un_alpha = self.net.activation_trunk(self.net.trunk(x_loc))
        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs)
        # print(branch_out_in.shape)
        # compute W, W_half and W_inv_half
        W_inv_half = torch.eye(branch_out_in.shape[1]).repeat(x_func.shape[0], 1, 1)
        W_half = torch.eye(branch_out_in.shape[1]).repeat(x_func.shape[0], 1, 1)
        omega = x_func[:,-1]
        W_inv_half[:,0,0] = omega ** -1
        W_half[:,0,0] = omega
        W = W_half ** 2        
        # perform QR decomposition on last two dimensions
        Q, R = torch.linalg.qr(branch_out_in)
        if not torch.allclose(torch.eye(Q.shape[2]), torch.bmm(Q.permute(0, 2, 1), Q), atol=1e-6):
            raise AssertionError("Q is not orthonormal")
        
        #make B_hat
        B_hat = Q @ W_inv_half
        # B_hat = B_hat.reshape(-1, B_hat.shape[2] * self.net.num_outputs)
        
        #check that QR=B
        if not torch.allclose(Q @ R, branch_out_in, atol=1e-5):
            raise AssertionError("QR decomposition does not equal B")

        #make alpha_hat
        un_alpha = (W_half @ R @ un_alpha.unsqueeze(-1)).squeeze(-1)
        norms = torch.norm(un_alpha, p=2, dim=1, keepdim=True)
        zero_mask = (norms == 0)
        norms = norms + zero_mask.float()  #TODO: Find a better solution to this!
        alpha_hat = un_alpha / norms
        if not torch.allclose(torch.ones(alpha_hat.shape[0])[~zero_mask.squeeze(1)], torch.norm(alpha_hat, p=2, dim=1, keepdim=True)[~zero_mask.squeeze(1)]):
            raise AssertionError("Trunk output is not normalized")
        if (zero_mask.squeeze(1)).sum() > B_hat.shape[0] * 0.001 and self.net.epoch > 10:
            raise AssertionError(f"Had to exclude more than 0.1% ({(zero_mask.squeeze(1)).sum()/B_hat.shape[0]}) of the batch at the normalizations stage")
        true_nrg = ((x_func[:,2] * x_func[:,0]) ** 2 + x_func[:,1] ** 2) #* 0.5
        # print(x_func.shape)
        # print(x_loc[-1])
        nrg = self.net.nrg_net(torch.concat((x_func, x_loc), dim=1))
        # print(f'average squared nrg: {(nrg ** 2).mean()} true_nrg: {true_nrg.mean()}')
        # print(f'predicted nrg: {nrg}, true nrg: {true_nrg}')
        alpha_hat = alpha_hat * torch.sqrt(true_nrg)[:,None] #nrg[:,None]
        out = B_hat @ alpha_hat.unsqueeze(-1)
        out = out.reshape(-1, self.net.num_outputs)
        return out, (B_hat.reshape(-1, B_hat.shape[2] * self.net.num_outputs), alpha_hat)

        # #combine branch and trunk
        # shift = 0
        # size = alpha_hat.shape[1]
        # xs = []
        # for i in range(self.net.num_outputs):
        #     branch_out_ = B_hat[:, shift : shift + size]
        #     x = self.net.merge_branch_trunk(branch_out_, alpha_hat, i)
        #     xs.append(x)
        #     shift += size
        # # print(f'final output shape: {self.net.concatenate_outputs(xs).shape}')
        # return self.net.concatenate_outputs(xs), (B_hat, alpha_hat) 
    
    
class NormalTrunkStrategy(CustomStrategy):
    def call(self, x_func, x_loc):
        torch.autograd.set_detect_anomaly(True)
        branch_out = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        norms = torch.norm(x_loc, p=2, dim=1, keepdim=True)
        #setnorms to 1 if zero
        zero_mask = (norms == 0)
        norms = norms + zero_mask.float()  #TODO: Find a better solution to this!
        x_loc = x_loc / norms
        if not torch.allclose(torch.ones(x_loc.shape[0])[~zero_mask.squeeze(1)], torch.norm(x_loc, p=2, dim=1, keepdim=True)[~zero_mask.squeeze(1)]):
            raise AssertionError("Trunk output is not normalized")
        if (zero_mask.squeeze(1)).sum() > branch_out.shape[0] * 0.01:
            raise AssertionError(f"Had to exclude more than 1% ({(zero_mask.squeeze(1)).sum()/branch_out.shape[0]}) of the batch at the normalizations stage")
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            branch_out_ = branch_out[:, shift : shift + size]
            x = self.net.merge_branch_trunk(branch_out_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)
    
class OrthonormalBranchStrategy(CustomStrategy):
    def call(self, x_func, x_loc):
        torch.autograd.set_detect_anomaly(True)
        branch_out = self.net.branch(x_func)
        x_loc = self.net.activation_trunk(self.net.trunk(x_loc))
        branch_out = branch_out.view(-1, self.net.num_outputs, branch_out.shape[1] // self.net.num_outputs)
        # #multiply first coordinate by omega
        # branch_out[:, 0, :] = branch_out[:, 0, :] * x_func[:, 2].unsqueeze(1)
        # perform QR decomposition on last two dimensions
        branch_out, r = torch.linalg.qr(branch_out)
        if not torch.allclose(torch.eye(branch_out.shape[2]), torch.bmm(branch_out.permute(0, 2, 1), branch_out), atol=1e-6):
            raise AssertionError("Q is not orthonormal")
        #invert implicit multiplication of first coordinate by omega
        branch_out[:, 0, :] = branch_out[:, 0, :] / x_func[:, 2].unsqueeze(1)
        #transform back to original shape
        branch_out = branch_out.reshape(-1, branch_out.shape[2] * self.net.num_outputs)
        shift = 0
        size = x_loc.shape[1]
        xs = []
        for i in range(self.net.num_outputs):
            branch_out_ = branch_out[:, shift : shift + size]
            x = self.net.merge_branch_trunk(branch_out_, x_loc, i)
            xs.append(x)
            shift += size
        return self.net.concatenate_outputs(xs)
    
class FourierStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierStrategy, self).__init__(net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of branch is not evenly divisible by {self.net.num_outputs}."
            )
        # if layer_sizes_trunk[-1] != self.net.num_outputs:
        #     raise AssertionError(
        #         f"Output size of the trunk net ({layer_sizes_trunk[-1]}) does not equal num outputs ({self.net.num_outputs})."
        #     )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )

    def call(self, x_func, x_loc, L=2 * np.pi):
        branch_out_in = self.net.branch(x_func) 
        #transform to complex
        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
        B = to_complex_tensor(branch_out_in)
        alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
        
        #make alpha complex float
        alpha = torch.complex(alpha, torch.zeros_like(alpha))

        four_coef = B @ alpha.unsqueeze(-1) #N x  M+1 x 1

        out = fourier_series_half_modes(x_loc, four_coef, self.L) #N x 1
        
        out = out.reshape(-1, 1) #for now final output is dim 1

        return out, (B.reshape(-1, B.shape[2] * self.net.num_outputs), alpha)
    
class FourierQRStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierStrategy, self).__init__(net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of branch is not evenly divisible by {self.net.num_outputs}."
            )
        # if layer_sizes_trunk[-1] != self.net.num_outputs:
        #     raise AssertionError(
        #         f"Output size of the trunk net ({layer_sizes_trunk[-1]}) does not equal num outputs ({self.net.num_outputs})."
        #     )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )
    
    def exact_soliton(x, t, c, a):
        arg = np.clip(np.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return ((c / 2) / np.cosh(arg) ** 2)  # Stable sech^2 computation
    
    def call(self, x_func, x_loc, L=2 * np.pi):
        branch_out_in = self.net.branch(x_func) 
        #transform to complex
        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
        B = to_complex_tensor(branch_out_in)
        un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
        un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

        Q, R = torch.linalg.qr(B)
        if not torch.allclose(torch.eye(Q.shape[2]), torch.bmm(Q.permute(0, 2, 1), Q), atol=1e-6):
            raise AssertionError("Q is not orthonormal")

        #check that QR=B
        if not torch.allclose(Q @ R, branch_out_in, atol=1e-5):
            raise AssertionError("QR decomposition does not equal B")
        
        B_hat = Q

        un_alpha = (R @ un_alpha.unsqueeze(-1)).squeeze(-1)
        norms = torch.norm(un_alpha, p=2, dim=1, keepdim=True)
        zero_mask = (norms == 0)
        norms = norms + zero_mask.float()  #TODO: Find a better solution to this!
        alpha_hat = un_alpha / norms
        if not torch.allclose(torch.ones(alpha_hat.shape[0])[~zero_mask.squeeze(1)], torch.norm(alpha_hat, p=2, dim=1, keepdim=True)[~zero_mask.squeeze(1)]):
            raise AssertionError("Trunk output is not normalized")
        if (zero_mask.squeeze(1)).sum() > B_hat.shape[0] * 0.001 and self.net.epoch > 10:
            raise AssertionError(f"Had to exclude more than 0.1% ({(zero_mask.squeeze(1)).sum()/B_hat.shape[0]}) of the batch at the normalizations stage")
        if (x_func.shape[1]<3):
            N = 100
            dx = self.L / N
            c, a = x_func[:,0], x_func[:,1]
            x = np.linspace(-self.L/2, self.L/2, 100, endpoint=False)
            u0 = exact_soliton(x, 0, c, a)
            true_nrg = np.sum(np.abs(u0)**2) * dx
        else:
            raise NotImplementedError('Energy calculation for other problems than 1d KdV not implemented')
        alpha_hat = alpha_hat * torch.sqrt(true_nrg)[:,None] #nrg[:,None]

        four_coef = B @ alpha_hat.unsqueeze(-1) #N x  M+1 x 1

        out = fourier_series_half_modes(x_loc, four_coef, self.L) #N x 1
        
        out = out.reshape(-1, 1) #for now final output is dim 1

        return out, (B.reshape(-1, B.shape[2] * self.net.num_outputs), alpha_hat)

def to_complex_tensor(tensor):
    """
    Converts a tensor of shape (N, M, 2K) into a complex tensor of shape (N, M, K).
    The first half of the last dimension is the real part, and the second half is the imaginary part.
    """
    K = tensor.shape[-1] // 2  # Determine K
    real = tensor[..., :K]  # First half is the real part
    imag = tensor[..., K:]   # Second half is the imaginary part
    return torch.complex(real, imag)  # Convert to complex tensor

def fourier_series_half_modes(x_loc, u_hat_m, L):
    """
    Compute u(x,t) from the Fourier expansion using only non-negative Fourier coefficients.

    Parameters:
    x_loc : torch.Tensor, shape (N, 2)
        Batched (t, x) pairs.
    u_hat_m : torch.Tensor, shape (N, M+1, 1)
        Fourier coefficients \hat{u}_m(t) for m >= 0.
    L : float
        Domain length.

    Returns:
    torch.Tensor, shape (N,)
        Computed function u(x,t) values.
    """
    N, num_modes, _ = u_hat_m.shape  # num_modes = M + 1
    M = num_modes - 1  # Max mode index
    t, x = x_loc[:, 0], x_loc[:, 1]  # Extract t and x from x_loc

    # Fourier wave numbers: m * k_0 where k_0 = 2π / L
    m_vals = torch.arange(0, M + 1, dtype=torch.float32, device=x_loc.device)  # Shape (M+1,)
    k_0 = 2 * torch.pi / L
    k_m = m_vals * k_0  # Shape (M+1,)

    # Compute exponentials for positive modes: e^(i k_m x) -> Shape (N, M+1)
    exp_term = torch.exp(1j * torch.outer(x, k_m))

    # Compute negative modes' coefficients using complex conjugates
    u_hat_neg = torch.conj(u_hat_m[:, 1:, :])  # Skip m=0 for symmetry
    exp_neg = torch.exp(-1j * torch.outer(x, k_m[1:]))  # Negative k_m exponentials

    # Compute Fourier sum
    u_x_t = torch.sum(exp_term * u_hat_m.squeeze(-1), dim=-1)  # Positive modes
    u_x_t += torch.sum(exp_neg * u_hat_neg.squeeze(-1), dim=-1)  # Negative modes

    return u_x_t.real


def print_summary(data):
    # Compute summary statistics by columns
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # Print the summary statistics
    print("Summary Statistics by Column:")
    print(f"Mean: {mean}")
    print(f"Standard Deviation: {std}")
    print(f"Minimum: {min_val}")
    print(f"Maximum: {max_val}")

