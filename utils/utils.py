import matplotlib.pyplot as plt
from itertools import cycle
from typing import List, Union
import numpy as np
import torch
from utils.data import exact_soliton
from scipy.fft import fft, ifft

from deepxde.nn.deeponet_strategy import DeepONetStrategy

class CustomStrategy(DeepONetStrategy):
    """Split the branch net and share the trunk net."""
    def __init__(self,args,net):
        super(CustomStrategy, self).__init__(net)
        self.args = args

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
    
class VanillaStrategy(CustomStrategy):
    def call(self, x_func, x_loc,x,y):
        x_loc.requires_grad = True
        branch_out = self.net.branch(x_func)
        alpha = self.net.activation_trunk(self.net.trunk(x_loc))
        B = branch_out.view(-1, self.net.num_outputs, branch_out.shape[1] // self.net.num_outputs)
        out = (B @ alpha.unsqueeze(-1))
        true_energy = ((x_func[:,2] * x_func[:,0]) ** 2 + x_func[:,1] ** 2) #* 0.5
        learned_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + out[:,1] ** 2
        gradients = torch.autograd.grad(outputs=out[:, 0], inputs=x_loc, grad_outputs=torch.ones_like(out[:, 0]), create_graph=True)[0]
        current_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + gradients ** 2
        energy_components = None
        return (out.squeeze(-1), [true_energy, current_energy, learned_energy, energy_components])
    

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
    def call(self, x_func, x_loc, x, y):
        x_loc.requires_grad = True
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
        true_energy = ((x_func[:,2] * x_func[:,0]) ** 2 + x_func[:,1] ** 2) #* 0.5
        # nrg = self.net.nrg_net(torch.concat((x_func, x_loc), dim=1))
        alpha_hat = alpha_hat * torch.sqrt(true_energy)[:,None] #nrg[:,None]
        out = B_hat @ alpha_hat.unsqueeze(-1)
        out = out.reshape(-1, self.net.num_outputs)
        gradients = torch.autograd.grad(outputs=out[:, 0], inputs=x_loc, grad_outputs=torch.ones_like(out[:, 0]), create_graph=True)[0]
        current_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + gradients ** 2
        learned_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + out[:,1] ** 2
        energy_components = None
        #return out, (B_hat.reshape(-1, B_hat.shape[2] * self.net.num_outputs), alpha_hat)
        return (out.squeeze(-1), [true_energy, current_energy, learned_energy, energy_components])

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

class NormalStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(NormalStrategy, self).__init__(args, net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.Nx
        self.num_outputs = args.num_outputs
        self.problem = args.problem
        self.num_output_fn = args.num_output_fn
        self.IC = args.IC
        self.i = 0
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.use_implicit_nrg = args.use_implicit_nrg

    def call(self, x_func, x_loc, x, y):
        torch.autograd.set_detect_anomaly(True)
        x_loc.requires_grad = True
        branch_out = self.net.branch(x_func)
        alpha = self.net.activation_trunk(self.net.trunk(x_loc))
        B = branch_out.view(-1, self.net.num_outputs, branch_out.shape[1] // self.net.num_outputs)
        out = (B @ alpha.unsqueeze(-1))
        aux_out = out.clone() 
        true_energy = ((x_func[:,2] * x_func[:,0]) ** 2 + x_func[:,1] ** 2)[:,None]
        learned_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + out[:,1] ** 2
        gradients = torch.autograd.grad(outputs=out[:, 0], inputs=x_loc, grad_outputs=torch.ones_like(out[:, 0]), create_graph=True)[0]

        current_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + gradients ** 2
        if self.use_implicit_nrg:
            norming_nrg = current_energy
            zero_mask = (norming_nrg == 0)
            norming_nrg = norming_nrg + zero_mask.float() 
            normalized_ch0 = out[:,0,:] / torch.sqrt(norming_nrg) * torch.sqrt(true_energy)
            out = torch.cat([normalized_ch0.unsqueeze(1), out[:,1:,:]], dim=1)
        else:
            norming_nrg = learned_energy
            zero_mask = (norming_nrg == 0)
            norming_nrg = norming_nrg + zero_mask.float() 
            out = out / torch.sqrt(norming_nrg).unsqueeze(1) * torch.sqrt(true_energy).unsqueeze(1)
        learned_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + out[:,1] ** 2
        gradients = torch.autograd.grad(outputs=out[:, 0], inputs=x_loc, grad_outputs=torch.ones_like(out[:, 0]), create_graph=True)[0]
        current_energy = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + gradients ** 2
        energy_components = None

        # else:
        #     for i in range(self.args.num_norm_refinements):
        #         gradients = torch.autograd.grad(outputs=out[:, 0], inputs=x_loc, grad_outputs=torch.ones_like(out[:, 0]), create_graph=True)[0]
        #         if self.args.detach:
        #             detached_gradients = gradients.detach()
        #             norms = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + detached_gradients ** 2
        #             undetached_norms = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + gradients ** 2
        #             zero_mask = (norms == 0)
        #             norms = norms + zero_mask.float()
        #             undetached_norms = undetached_norms + zero_mask.float()
        #             normalized_ch0 = out[:,0,:] / torch.sqrt(norms) * torch.sqrt(true_nrg)
        #             out = torch.cat([normalized_ch0.unsqueeze(1), out[:,1:,:]], dim=1)
        #             undetached_normalized_ch0 = out[:,0,:] / torch.sqrt(undetached_norms) * torch.sqrt(true_nrg)
        #             aux_out = torch.cat([undetached_normalized_ch0.unsqueeze(1), undetached_normalized_ch0[:,1:,:]], dim=1)
        #         else:
        #             norms = x_func[:,2].unsqueeze(-1) * out[:,0] ** 2 + gradients ** 2
        #             zero_mask = (norms == 0)
        #             norms = norms + zero_mask.float()
        #             normalized_ch0 = out[:,0,:] / torch.sqrt(norms) * torch.sqrt(true_nrg)
        #             out = torch.cat([normalized_ch0.unsqueeze(1), out[:,1:,:]], dim=1)
        #             aux_out = None
        #         # print(gradients.shape)
        return (out.squeeze(-1), [true_energy, current_energy, learned_energy, energy_components])
    
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


def zero_pad_and_build_symmetric(four_coef, K):
    """
    Pads one-sided Fourier coefficients to length K by inserting zeros
    and builds a Hermitian-symmetric spectrum for real-valued ifft.

    Args:
        four_coef: (B, N), where N is odd (DC + positive freqs)
        K: target spatial resolution (must be odd and >= 2*N - 1)

    Returns:
        full_spectrum: (B, K), Hermitian-symmetric Fourier spectrum
    """
    B, N = four_coef.shape
    assert N % 2 == 0, "Expected even N (number of positive frequencies including DC)"
    assert K % 2 == 1, "K must be odd to preserve symmetry with odd N"
    assert K >= 2 * N - 1, f"K must be at least 2N-1 to preserve structure (got K={K}, N={N})"

    # Number of positive freqs (excluding DC)
    num_pos = N - 1
    num_neg = num_pos

    # Amount of zero padding to insert between DC and mirrored neg freqs
    pad_len = K - (1 + 2 * num_pos)
    pad_mid = torch.zeros(B, pad_len, dtype=four_coef.dtype, device=four_coef.device)

    # Build the full symmetric spectrum
    pos = four_coef[:, 1:]         # Positive freqs
    neg = torch.conj(torch.flip(pos, dims=[1]))  # Negative freqs

    full_spectrum = torch.cat([
        four_coef[:, :1],  # DC
        pos,
        pad_mid,
        neg
    ], dim=1)

    return full_spectrum


class FourierStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierStrategy, self).__init__(args, net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.Nx
        self.num_outputs = args.num_outputs
        self.problem = args.problem
        self.num_output_fn = args.num_output_fn
        self.IC = args.IC
        self.i = 0
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.use_implicit_nrg = args.use_implicit_nrg


    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of branch is not evenly divisible by {self.net.num_outputs}."
            )
        
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )
    
    def exact_soliton(x, t, c, a):
        arg = np.clip(np.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return ((c / 2) / np.cosh(arg) ** 2)  # Stable sech^2 computation
    
    def call(self, x_func, x_loc, x=None, y=None):
        branch_out_in = self.net.branch(x_func)

        x_loc.requires_grad = True

        #transform to complex
        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
        B = to_complex_tensor(branch_out_in)
        un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
        un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

        four_coef = (B @ un_alpha.unsqueeze(-1)).squeeze(-1) #N x  M+1

        four_coef_list = [four_coef[:, i*int(self.num_outputs//2):(i+1)*int(self.num_outputs//2)] for i in range(self.num_output_fn)]
        out_list = []
        new_four_coef_list = []
        for four_coef in four_coef_list:
            four_coef[:, 0].imag = 0
            four_coef_hat = zero_pad_and_build_symmetric(four_coef, self.K)
            out = torch.fft.ifft(four_coef_hat.squeeze(-1))
            assert out.imag.abs().max() < 1e-5, f'Imaginary part of fourier coefficients is too large: {four_coef_hat.imag.abs().max()}'
            out_list.append(out)
            new_four_coef_list.append(four_coef_hat)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies(self, out, four_coef, x_func, x_loc, x, y)

        self.i += 1

        return out.real, list(energies)
    
class FourierNormStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierNormStrategy, self).__init__(args, net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.Nx
        self.num_outputs = args.num_outputs
        self.problem = args.problem
        self.IC = args.IC
        self.num_output_fn = args.num_output_fn
        self.i = 0
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.use_implicit_nrg = args.use_implicit_nrg


    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of branch is not evenly divisible by {self.net.num_outputs}."
            )
        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )
    
    
    
    def exact_soliton(x, t, c, a):
        arg = np.clip(np.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return ((c / 2) / np.cosh(arg) ** 2)  # Stable sech^2 computation
    
    def call(self, x_func, x_loc, x=None, y=None):

        if self.problem == '1d_wave' or self.problem == '1d_KdV_Soliton':
            
            if self.problem == '1d_wave':
                x_loc.requires_grad = True

            branch_out_in = self.net.branch(x_func) 
            
            branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
            B = to_complex_tensor(branch_out_in)
            un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
            un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

            four_coef = (B @ un_alpha.unsqueeze(-1)).squeeze(-1) #N x  M+1 
            four_coef_list = [four_coef[:, i*int(self.num_outputs//2):(i+1)*int(self.num_outputs//2)] for i in range(self.num_output_fn)]
            prelim_out_list = []
            prelim_four_coef_list = []
            for i, four_coef in enumerate(four_coef_list):
                four_coef[:, 0].imag = 0
                prelim_four_coef_hat = zero_pad_and_build_symmetric(four_coef, self.K)
                prelim_out = torch.fft.ifft(prelim_four_coef_hat.squeeze(-1))
                assert prelim_out.imag.abs().max() < 1e-3, f'Imaginary part of fourier coefficients is too large: {prelim_out.imag.abs().max()}'
                prelim_out_list.append(prelim_out)
                prelim_four_coef_list.append(prelim_four_coef_hat)
            prelim_out = torch.cat(prelim_out_list, dim=1)
            prelim_four_coef = torch.cat(prelim_four_coef_list, dim=1)

            assert prelim_out.imag.abs().max() < 1e-4, f'Imaginary part of fourier coefficients is too large: {prelim_out.imag.abs().max()}'
            energies = compute_energies(self, prelim_out, prelim_four_coef, x_func, x_loc, x, y)

            scaling_factor = compute_scaling_factor(self, prelim_out, prelim_four_coef, energies)
            
            four_coef_final = prelim_four_coef * scaling_factor

            four_coef_list_final = [four_coef_final[:, i*int(four_coef_final.shape[1]//self.num_output_fn):(i+1)*int(four_coef_final.shape[1]//self.num_output_fn)] for i in range(self.num_output_fn)]
            out_list = []
            for four_coef in four_coef_list_final:
                out = torch.fft.ifft(four_coef, dim=1) #* self.K
                assert out.imag.abs().max() < 1e-3, f'Imaginary part of fourier coefficients is too large: {out.imag.abs().max()}'
                out_list.append(out)
            out = torch.cat(out_list, dim=1)

            out = out.squeeze(-1)

            energies = compute_energies(self, out, four_coef_final, x_func, x_loc, x, y)

            # fourier_energy = torch.sum(torch.abs(four_coef_hat) ** 2, dim=1, keepdim=True) * torch.tensor(self.L)

            self.i += 1

            return out.real, list(energies)
        
        # elif self.problem == '1d_KdV_Soliton':
        #     branch_out_in = self.net.branch(x_func) 
        
        #     branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
        #     B = to_complex_tensor(branch_out_in)
        #     un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
        #     un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

        #     four_coef = (B @ un_alpha.unsqueeze(-1)).squeeze(-1) #N x  M+1 
        #     neg_four_coef = torch.conj(torch.flip(four_coef[:, 1:], dims=[1]))  # Flip to maintain Hermitian symmetry
        #     four_coef[:, 0].imag = 0  # Ensure DC is real
        #     four_coef = torch.cat((four_coef, neg_four_coef), dim=1)

        #     if (x_func.shape[1]<3):
        #         N = 500
        #         dx = self.L / N
        #         a, c = x_func[:,0], x_func[:,1]
        #         x = torch.linspace(-self.L/2, self.L/2, N).unsqueeze(1)
        #         u0 = exact_soliton(x, 0, c, a)
        #         true_nrg = torch.sum(torch.abs(u0)**2, dim=0) * dx
        #     else:
        #         raise NotImplementedError('Energy calculation for other problems than 1d KdV not implemented')

        #     four_coef_hat = old_project_fourier_coefficients(four_coef, true_nrg.unsqueeze(-1), self.L)

        #     out = torch.fft.ifft(four_coef_hat.squeeze(-1)) * self.K
        #     assert out.imag.abs().max() < 1e-5, f'Imaginary part of fourier coefficients is too large: {four_coef_hat.imag.abs().max()}'

        #     fourier_energy = torch.sum(torch.abs(four_coef_hat) ** 2, dim=1, keepdim=True) * torch.tensor(self.L)

        #     return out.real, (B.reshape(-1, B.shape[2] * self.net.num_outputs), un_alpha, fourier_energy)


def weighted_qr_ignore_zero_rows(A, W):
    """
    Perform a 'weighted QR' on batched matrix A w.r.t. diag(W), ignoring rows where W=0.
    Returns Q, R such that:
      1) A = Q R   (reconstructs only the rows where W>0 exactly) 
      2) Q^T diag(W) Q = I in the subspace of nonzero W.
         Rows with W=0 get filled with zeros in Q by default.
    
    Args:
        A: tensor of shape (batch_size, m, n)
        W: tensor of shape (batch_size, m) diagonal entries, may have zeros
    """
    assert A.shape[:-1] == W.shape, "Mismatched shapes between A and W."

    # 1) Identify nonzero-weight rows
    mask = (W > 0)
    
    # 2) Extract submatrix and subweights
    batch_size = A.shape[0]
    Q_list = []
    R_list = []
    
    # Process each batch element
    for i in range(batch_size):
        A_sub = A[i][mask[i], :]            # shape (m_sub, n)
        W_sub = W[i][mask[i]]               # shape (m_sub,)

        # If no zero weights, just do a normal weighted QR
        if A_sub.shape[0] == A[i].shape[0]:
            Q_batch, R_batch = weighted_qr(A[i].unsqueeze(0), W[i].unsqueeze(0))
            Q_list.append(Q_batch.squeeze(0))
            R_list.append(R_batch.squeeze(0))
            continue

        # 3) Compute Weighted QR on submatrix
        Q_sub, R = weighted_qr(A_sub.unsqueeze(0), W_sub.unsqueeze(0))
        Q_sub = Q_sub.squeeze(0)
        R = R.squeeze(0)

        # 4) Reassemble full-size Q 
        m, n = A[i].shape
        Q = A[i].new_zeros((m, n))  # same dtype/device
        Q[mask[i], :] = Q_sub       # put back the valid rows
        
        Q_list.append(Q)
        R_list.append(R)

    return torch.stack(Q_list), torch.stack(R_list)

def weighted_qr(A, W):
    """
    Standard Weighted QR: A = Q R, with Q^T diag(W) Q = I.
    (Requires that W > 0 on all diagonal entries.)
    
    Args:
        A: tensor of shape (batch_size, m, n)
        W: tensor of shape (batch_size, m) diagonal entries
    """
    W_sqrt = W.sqrt()      # shape (batch_size, m)
    W_inv_sqrt = 1.0 / W_sqrt

    W = torch.complex(W, torch.zeros_like(W_sqrt))
    W_sqrt = torch.complex(W_sqrt, torch.zeros_like(W_sqrt)) 
    W_inv_sqrt = torch.complex(W_inv_sqrt, torch.zeros_like(W_inv_sqrt))

    # Multiply each row i by sqrt(W[i])
    B = W_sqrt.unsqueeze(-1) * A

    # Batched QR
    Q_std, R = torch.linalg.qr(B, mode='reduced')
    
    # Map Q_std back
    Q = W_inv_sqrt.unsqueeze(-1) * Q_std
    
    return Q, R

class FourierQRStrategy(CustomStrategy):

    def __init__(self, args, net):
        super(FourierQRStrategy, self).__init__(args, net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.Nx
        self.problem = args.problem
        self.num_outputs = args.num_outputs
        self.num_output_fn = args.num_output_fn
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.IC = args.IC
        self.use_implicit_nrg = args.use_implicit_nrg

    def build(self, layer_sizes_branch, layer_sizes_trunk):
        if layer_sizes_branch[-1] % self.net.num_outputs != 0:
            raise AssertionError(
                f"Output size of branch is not evenly divisible by {self.net.num_outputs}."
            )

        return self.net.build_branch_net(layer_sizes_branch), self.net.build_trunk_net(
            layer_sizes_trunk
        )
    
    def exact_soliton(x, t, c, a):
        arg = np.clip(np.sqrt(c) * (x - c * t - a) / 2, -50, 50)  # Prevent extreme values
        return ((c / 2) / np.cosh(arg) ** 2)  # Stable sech^2 computation
    
    def call(self, x_func, x_loc, x=None, y=None):

        x_loc.requires_grad = True
        branch_out_in = self.net.branch(x_func) 

        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
        B = to_complex_tensor(branch_out_in)
        # assert B.shape[1] >= B.shape[2], "B should be at least as wide as it is tall"
        un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
        un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

        # Create diagonal matrix with ones and wave numbers
        diag_size = self.num_outputs #(self.num_outputs // 2 + 1) * self.num_output_fn
        half_size = diag_size // 2
        k = torch.fft.fftfreq(self.K, d=(self.L/self.K))[0:half_size] * 2 * torch.pi
        W = torch.cat([
                self.IC['c']*k,
                torch.ones(half_size)
            ])
        #W[0]=0 #for test
        W = W * torch.sqrt(torch.tensor(self.L / (self.K ** 2)))

        mask = (W > 0)
        
        Q = torch.zeros(B.shape[0], B.shape[1], min(torch.sum(mask).item(), B.shape[2]), dtype=B.dtype)
        B_sub = B[:, mask,:]
        W_sub = W[mask]
        W_sub_invs = 1.0 / W_sub

        W_sub_sqrt_mat = torch.diag(W_sub).unsqueeze(0).expand(B.shape[0],-1,-1)  # (n_nonzero, n_nonzero)
        W_sub_invsqrt = torch.diag(W_sub_invs).unsqueeze(0).expand(B.shape[0],-1,-1)  # (n_nonzero, n_nonzero)

        W_sub_sqrt_mat = torch.complex(W_sub_sqrt_mat, torch.zeros_like(W_sub_sqrt_mat))
        W_sub_invsqrt = torch.complex(W_sub_invsqrt, torch.zeros_like(W_sub_invsqrt))

        B_std = W_sub_sqrt_mat @ B_sub
        Q_std, R = torch.linalg.qr(B_std, mode='reduced')

        Q_sub = W_sub_invsqrt @ Q_std


        Q[:, mask, :] = Q_sub



        #check that QR=B
        if not torch.allclose((Q_std @ R), B_std, atol=1e-4):
            max_diff = torch.norm((Q_std @ R) - B_std, dim=(-2,-1)).max().item()
            print(f"Max reconstruction error ||B - Q R|| = {max_diff}")
            raise AssertionError("QR decomposition does not equal B")

        B_hat = Q
        un_alpha = (R @ un_alpha.unsqueeze(-1)).squeeze(-1)
        norms = torch.norm(un_alpha, p=2, dim=1, keepdim=True)
        zero_mask = (norms == 0)
        norms = norms + zero_mask.float()  #TODO: Find a better solution to this!
        alpha_hat = un_alpha / norms

        #compute energy
        gt_u = x[0][:,0,:]
        gt_ut = x[0][:,1,:]

        k = torch.tensor(np.fft.fftfreq(self.K, d=(self.L/self.K)) * 2* np.pi).float()  # to device?

        gt_u_hat = torch.fft.fft(gt_u, n=self.K, dim=1)  # dim=1 since shape is (time, space)
        gt_ut_hat = torch.fft.fft(gt_ut, n=self.K, dim=1)

        true_energy_ut_component = torch.sum(torch.abs(gt_ut_hat)**2, dim=1) * self.L / (self.K ** 2)
        true_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(gt_u_hat)**2, dim=1) * self.L / (self.K ** 2)
        true_energy = true_energy_ut_component + true_energy_u_component

        alpha_hat = alpha_hat * torch.sqrt(true_energy)[:,None] / torch.sqrt(torch.tensor(2))

        four_coef = (B_hat @ alpha_hat.unsqueeze(-1)) #N x  M+1 


        four_coef = four_coef.squeeze(-1)

        # W_mat = torch.complex(torch.diag_embed(W ** 2), torch.zeros_like(torch.diag_embed(W)))

        # u_weighted = four_coef.unsqueeze(-1)  # (batch_size, 4, 1)
        # energy_weighted = torch.diagonal(
        #     torch.bmm(
        #         torch.bmm(u_weighted.transpose(-2,-1).conj(), W_mat.unsqueeze(0).expand(B.shape[0], -1, -1)),
        #         u_weighted
        #     ),
        #     dim1=-2, dim2=-1
        # ).real

        # print(f'computed energy {energy_weighted[50]}')
        # print(f'true energy {true_energy[50]/2}')
        # print(four_coef.shape)
        # print(self.num_outputs)

        four_coef_list = [four_coef[:, i*int(self.num_outputs//2):(i+1)*int(self.num_outputs//2)] for i in range(self.num_output_fn)]
        
        prelim_out_list = []
        prelim_four_coef_list = []
        torch.autograd.set_detect_anomaly(True)
        for i, four_coef in enumerate(four_coef_list):
            new_real = torch.sqrt(torch.clamp(torch.abs(four_coef[:, 0]), min=1e-12))
            new_imag = torch.zeros_like(new_real)
            new_col0 = torch.complex(new_real, new_imag)

            # Clone four_coef and replace column 0 non-inplace
            four_coef = four_coef.clone()
            four_coef[:, 0] = new_col0
            # Construct new complex number
            four_coef[:, 0] = torch.complex(new_real, new_imag)
            # four_coef[:, 0].real = torch.sqrt(torch.abs(four_coef[:, 0]))
            # four_coef[:, 0].imag = 0
            # print(four_coef.shape)
            prelim_four_coef_hat = zero_pad_and_build_symmetric(four_coef, self.K)
            prelim_out = torch.fft.ifft(prelim_four_coef_hat.squeeze(-1))
            #assert prelim_out.imag.abs().max() < 1e-3, f'Imaginary part of fourier coefficients is too large: {prelim_four_coef_hat.imag.abs().max()}'
            prelim_out_list.append(prelim_out)
            prelim_four_coef_list.append(prelim_four_coef_hat)
        out = torch.cat(prelim_out_list, dim=1)
        four_coef = torch.cat(prelim_four_coef_list, dim=1)
        
        energies = compute_energies(self, out, four_coef, x_func, x_loc, x, y)

        #assert out.imag.abs().max() < 1e-5, f'Imaginary part of fourier coefficients is too large: {four_coef.imag.abs().max()}'

        return out.real, list(energies)

def compute_energies(self, prelim_out, four_coef, x_func, x_loc, x, y):
    if self.problem == '1d_KdV_Soliton':
        N = 500
        dx = self.L / N
        a, c = x_func[:,0], x_func[:,1]
        x = torch.linspace(-self.L/2, self.L/2, N, endpoint=False).unsqueeze(1)
        u0 = exact_soliton(x, 0, c, a)
        true_energy = torch.sum(torch.abs(u0)**2, dim=0) * dx
        # Compute the current L2 norm squared per batch
        current_energy = (torch.sum(torch.abs(four_coef) ** 2, dim=1, keepdim=True) * self.L / (self.K ** 2)).squeeze(-1).squeeze(-1)
        learned_energy = None
    elif self.problem == '1d_wave':

        if self.num_output_fn == 1:

            #generate wave numbers
            k = torch.tensor(np.fft.fftfreq(self.K, d=(self.L/self.K)) * 2* np.pi).float()  # to device?

            #compute current energy
            ut = torch.autograd.grad(outputs=prelim_out, inputs=x_loc, grad_outputs=torch.ones_like(prelim_out), create_graph=True, allow_unused=True)[0] #allow_unused?
            ut_hat = torch.fft.fft(ut, n=self.K, dim=1)
            u_hat = four_coef.squeeze(-1)

            current_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) * self.L / (self.K ** 2)
            current_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) * self.L / (self.K ** 2)
            # print(f'cur energy ut component: {current_energy_ut_component.mean().item()}')
            # print(f'cur energy u component: {current_energy_u_component.mean().item()}') 
            
            current_energy = current_energy_ut_component + current_energy_u_component

            learned_energy_ut_component = None
            learned_energy_u_component = None
            learned_energy = None

        elif self.num_output_fn == 2:
            #generate wave numbers
            k = torch.tensor(np.fft.fftfreq(self.K, d=(self.L/self.K)) * 2* np.pi).float()  # to device?
            
            output_list = [prelim_out[:, i*int(prelim_out.shape[1]//self.num_output_fn):(i+1)*int(prelim_out.shape[1]//self.num_output_fn)] for i in range(self.num_output_fn)]
            prelim_out = output_list[0]
            prelim_outt = output_list[1]

            coef_list = [four_coef[:, i*int(four_coef.shape[1]//self.num_output_fn):(i+1)*int(four_coef.shape[1]//self.num_output_fn)] for i in range(self.num_output_fn)]
            u_hat = coef_list[0].squeeze(-1)
            ut_hat = coef_list[1].squeeze(-1)

            #compute learned energy            
            learned_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) * self.L / (self.K ** 2)
            learned_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) * self.L / (self.K ** 2)
            # print(f'learned energy ut component: {learned_energy_ut_component.mean().item()}')
            # print(f'learned energy u component: {learned_energy_u_component.mean().item()}')

            learned_energy = learned_energy_ut_component + learned_energy_u_component


            #compute current energy
            ut = torch.autograd.grad(outputs=prelim_out, inputs=x_loc[:,0], grad_outputs=torch.ones_like(prelim_out), create_graph=True, allow_unused=True)[0] #allow_unused?
            #ut_hat = torch.fft.fft(ut, n=self.K, dim=1)
            ut_hat = torch.fft.fft(ut, n=self.K, dim=1)

            current_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) * self.L / (self.K ** 2)
            current_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) * self.L / (self.K ** 2)
            # print(f'current energy ut component: {current_energy_ut_component.mean().item()}')
            # print(f'current energy u component: {current_energy_u_component.mean().item()}') 

            print(f'learned energy ut component: {learned_energy_ut_component.mean().item()}')
            print(f'current energy ut component: {current_energy_ut_component.mean().item()}')

            current_energy = current_energy_ut_component + current_energy_u_component

        if x is not None:
            gt_u = x[0][:,0,:]
            gt_ut = x[0][:,1,:]

            k = torch.tensor(np.fft.fftfreq(self.K, d=(self.L/self.K)) * 2* np.pi).float()  # to device?

            gt_u_hat = torch.fft.fft(gt_u, n=self.K, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=self.K, dim=1)

            true_energy_ut_component = torch.sum(torch.abs(gt_ut_hat)**2, dim=1) * self.L / (self.K ** 2)
            true_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(gt_u_hat)**2, dim=1) * self.L / (self.K ** 2)
            true_energy = true_energy_ut_component + true_energy_u_component

        if y is not None:
            gt_u = y[:,0,:]
            gt_ut = y[:,1,:]

            gt_u_hat = torch.fft.fft(gt_u, n=self.K, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=self.K, dim=1)
            target_energy_ut_component = torch.sum(torch.abs(gt_ut_hat)**2, dim=1) * self.L / (self.K ** 2)
            target_energy_ux_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(gt_u_hat)**2, dim=1) * self.L / (self.K ** 2)
            target_energy = target_energy_ut_component + target_energy_ux_component

        #assert (true_energy-target_energy).abs().max() < 1, f'max difference between true energy and target energy is {(true_energy-target_energy).abs().max().item()}'
    else:
        raise NotImplementedError('Energy calculation for other problems than 1d wave not implemented')
    
    energy_components = {'true_energy_u_component': true_energy_u_component.unsqueeze(1), 'true_energy_ut_component': true_energy_ut_component.unsqueeze(1), 
                         'current_energy_u_component': current_energy_u_component.unsqueeze(0), 'current_energy_ut_component': current_energy_ut_component.unsqueeze(0),
                        'target_energy': target_energy, 'target_energy_ux_component': target_energy_ux_component, 'target_energy_ut_component': target_energy_ut_component,
                         'learned_energy_u_component': learned_energy_u_component.unsqueeze(0), 'learned_energy_ut_component': learned_energy_ut_component.unsqueeze(0)}
        

    return true_energy, current_energy, learned_energy, energy_components

def compute_energies_full_fourier(self, prelim_out, four_coef, og_x, og_y, in_physical_space=False):
    if in_physical_space:
        # prelim_out = prelim_out.detach().cpu().numpy()
        dx = self.Lx/self.Nx  # space step
        dt = self.Lt/self.Nt
        # du_dx, du_dt = np.gradient(prelim_out, dx, dt, axis=(-2, -1))
        prelim_out = prelim_out.detach()
        # dx = self.Lx/self.Nx  # space step
        # dt = self.Lt/self.Nt
        du_dx = torch.gradient(prelim_out, dim=-2)[0]/dx 
        du_dt = torch.gradient(prelim_out, dim=-1)[0]/dt
        phys_current_energy_ux_component = torch.sum(self.IC['c']**2 * du_dx**2, dim=(-2)) * dx
        phys_current_energy_ut_component = torch.sum(du_dt**2, dim=(-2)) * dx 
        phys_current_energy = phys_current_energy_ut_component + phys_current_energy_ux_component
        # phys_current_energy_ux_component = np.sum(self.IC['c']**2 * du_dx**2, axis=(-2)) * dx
        # phys_current_energy_ut_component = np.sum(du_dt**2, axis=(-2)) * dx
        # phys_current_energy = phys_current_energy_ut_component + phys_current_energy_ux_component

        
    if self.problem == '1d_wave':

        y = og_y[:,0,:,:]
        yt = og_y[:,1,:,:]

        wave_k = torch.fft.fftfreq(self.Nx, d=(self.Lx/self.Nx)) * 2 * np.pi  # Wave numbers

        u_hat = torch.fft.fft(y, dim=1)
        ut_hat = torch.fft.fft(yt, dim=1)

        og_target_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)
        og_target_energy_u_component = torch.sum(self.IC['c']**2 * (wave_k.unsqueeze(0).unsqueeze(-1).expand(u_hat.shape) ** 2) * torch.abs(u_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)
        og_target_energy = og_target_energy_ut_component + og_target_energy_u_component

        u_hat = torch.fft.fftn(y, dim=(1,2))
        u = torch.fft.ifftn(u_hat, dim=(1,2))

        ux_hat = torch.fft.fft(u, dim=1)

        target_energy_ux_component = torch.sum(self.IC['c']**2 * (wave_k.unsqueeze(0).unsqueeze(-1).expand(ux_hat.shape) ** 2) * torch.abs(ux_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)

        wave_kt = 2 * torch.pi * torch.fft.fftfreq(self.Nt, d=self.Lt / self.Nt)

        wave_kt_expanded = wave_kt.view(1, 1, -1)

        ut_hat = torch.fft.fft(y, dim=-1)

        dut_hat = wave_kt_expanded * ut_hat

        dut = torch.fft.ifftn(dut_hat, dim=-1)

        fourc_coef_in_space = torch.fft.fft(dut, dim=1)

        target_energy_ut_component = torch.sum(torch.abs(fourc_coef_in_space) ** 2, dim=1) * self.Lx  / (self.Nx**2)

        target_energy = target_energy_ut_component + target_energy_ux_component
        
        if self.num_output_fn == 1:

            u = torch.fft.ifftn(four_coef, dim=(1,2))
            
            ux_hat = torch.fft.fft(u, dim=1)

            current_energy_ux_component = torch.sum(self.IC['c']**2 * (wave_k.unsqueeze(0).unsqueeze(-1).expand(ux_hat.shape) ** 2) * torch.abs(ux_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)

            wave_kt = 2 * torch.pi * torch.fft.fftfreq(self.Nt, d=self.Lt / self.Nt)

            wave_kt_expanded = wave_kt.view(1, 1, -1)

            ut_hat = torch.fft.fft(u, dim=-1)

            dut_hat = wave_kt_expanded * ut_hat

            dut = torch.fft.ifft(dut_hat, dim=-1)

            fourc_coef_in_space = torch.fft.fft(dut, dim=1)

            current_energy_ut_component = torch.sum(torch.abs(fourc_coef_in_space) ** 2, dim=1) * self.Lx  / (self.Nx**2)

            current_energy = current_energy_ut_component + current_energy_ux_component

            learned_energy_ut_component = None
            learned_energy_ux_component = None
            learned_energy = None

            gt_u = og_x[:,0,:]
            gt_ut = og_x[:,1,:]

            gt_u_hat = torch.fft.fft(gt_u, n=self.Nx, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=self.Nx, dim=1)

            true_energy_ut_component = torch.sum(torch.abs(gt_ut_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)
            true_energy_ux_component = torch.sum(self.IC['c']**2 * (wave_k ** 2) * torch.abs(gt_u_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)
            true_energy = true_energy_ut_component + true_energy_ux_component

        if self.num_output_fn == 2:
            raise NotImplementedError('Energy calculation for two output functions not implemented')
        
        energy_components = {'true_energy_u_component': true_energy_ux_component, 'true_energy_ut_component': true_energy_ut_component, 
                         'current_energy_u_component': current_energy_ux_component, 'current_energy_ut_component': current_energy_ut_component,
                         'learned_energy_u_component': learned_energy_ux_component, 'learned_energy_ut_component': learned_energy_ut_component,
                         'target_energy': target_energy.T, 'target_energy_ux_component': target_energy_ux_component, 'target_energy_ut_component': target_energy_ut_component,
                            'og_target_energy': og_target_energy, 'og_target_energy_ux_component': og_target_energy_u_component, 'og_target_energy_ut_component': og_target_energy_ut_component}
        if in_physical_space:
            energy_components['phys_current_energy_ux_component']= phys_current_energy_ux_component
            energy_components['phys_current_energy_ut_component']= phys_current_energy_ut_component
            energy_components['phys_current_energy']= phys_current_energy

        # print(f'true energy {true_energy.mean()}')
        # print(f'current energy {current_energy.mean()}')
        # print(f'true energy ut component {true_energy_ut_component.mean()}')
        # print(f'current energy ut component {current_energy_ut_component.mean()}')
        # print(f'true energy ux component {true_energy_ux_component.mean()}')
        # print(f'current energy ux component {current_energy_ux_component.mean()}')
        # print(f'og target energy {og_target_energy.mean()}')

    return true_energy, current_energy, learned_energy, energy_components





def low_pass_filter_fft(data_fft, cutoff_ratio=0.8):
    """
    Zero out high frequency components beyond a certain ratio of the Nyquist frequency.
    """
    Nt = data_fft.shape[-1]
    cutoff = int(cutoff_ratio * (Nt // 2))
    filtered_fft = data_fft.clone()
    filtered_fft[..., cutoff:Nt - cutoff] = 0
    return filtered_fft


    
def compute_scaling_factor(self, prelim_out, four_coef, energies) -> torch.Tensor:
    """
    Projects a batch of Fourier coefficients onto the manifold ∑|c_i|² = C.

    Args:
        four_coef (torch.Tensor): Tensor of shape [batch_size, num_fourier_modes]
                                  representing Fourier coefficients.
        C (float): Desired L2 norm squared (energy level).

    Returns:
        torch.Tensor: Projected Fourier coefficients of the same shape.
    """
    true_energy, current_energy, learned_energy, energy_components = energies

    if self.use_implicit_nrg:
        norming_energy = current_energy
    else:
        norming_energy = learned_energy

    mask = (norming_energy == 0)
    scaling_factor = torch.zeros_like(true_energy)
    scaling_factor[~mask] = torch.sqrt(true_energy[~mask] / norming_energy[~mask])
    scaling_factor = scaling_factor[:,None]

    # Check if scaling factor is complex
    assert not torch.is_complex(scaling_factor), "Scaling factor is complex"

    return scaling_factor

def scaling_factor_full_fourier(self, prelim_out, four_coef, energies) -> torch.Tensor:
    """
    Projects a batch of Fourier coefficients onto the manifold ∑|c_i|² = C.

    Args:
        four_coef (torch.Tensor): Tensor of shape [batch_size, num_fourier_modes]
                                  representing Fourier coefficients.
        C (float): Desired L2 norm squared (energy level).

    Returns:
        torch.Tensor: Projected Fourier coefficients of the same shape.
    """
    true_energy, current_energy, learned_energy, energy_components = energies

    if self.num_output_fn == 1:
        norming_energy = current_energy
    elif self.num_output_fn == 2:
        norming_energy = learned_energy

    #broadcast true_energy to match shape of norming_energy
    true_energy = true_energy.unsqueeze(-1).expand(-1, norming_energy.shape[1])
    mask = (norming_energy == 0)
    scaling_factor = torch.zeros_like(true_energy)
    scaling_factor[~mask] = torch.sqrt(true_energy[~mask] / norming_energy[~mask])
    scaling_factor = scaling_factor[:,None].expand(four_coef.shape)
   
    assert not torch.is_complex(scaling_factor), "Scaling factor is complex"

    return scaling_factor
    

def padded_ifft_hermitian(four_coef, K):
    """
    Handles real signal IFFT from one-sided coeffs with odd N, padded to spatial resolution K.
    """
    B, M = four_coef.shape

    assert K >= M, f"Target resolution K={K} must be >= {M} (2N - 1)"
    pad_left = (K - M) // 2
    pad_right = K - M - pad_left

    padded_spec = torch.nn.functional.pad(four_coef, (pad_left, pad_right), mode='constant', value=0)

    # Inverse FFT to real signal
    out = torch.fft.ifft(padded_spec, n=K)
    return out


def to_complex_tensor(tensor):
    """
    Converts a tensor of shape (N, M, 2K) into a complex tensor of shape (N, M, K).
    The first half of the last dimension is the real part, and the second half is the imaginary part.
    """
    K = tensor.shape[-1] // 2  # Determine K
    real = tensor[..., :K]  # First half is the real part
    imag = tensor[..., K:]   # Second half is the imaginary part
    return torch.complex(real, imag)  # Convert to complex tensor

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

class FullFourierStrategy():
    def __init__(self, args, net):
        self.net = net
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.Lx = args.xmax - args.xmin
        self.Lt = args.tmax - args.tmin
        self.Nx = args.Nx
        self.Nt = args.Nt
        self.problem = args.problem
        self.num_output_fn = args.num_output_fn
        self.IC = args.IC
        self.i = 0
        self.t_filter_cutoff_ratio = args.t_filter_cutoff_ratio
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.inference_projection = args.inference_projection
        self.num_norm_refinements = args.num_norm_refinements

    def call(self, out, x=None, y=None):
        four_coef = out
        # Combine real and imaginary parts into complex tensor
        four_coef = to_complex_tensor(four_coef)
        four_coef = four_coef.view(-1, self.Nx * self.num_output_fn, (self.Nt // 2) + 1)

        four_coef.view(-1, self.Nx * self.num_output_fn, self.Nt // 2 + 1)

        four_coef_list = [four_coef[:,i*int(self.Nx):(i+1)*int(self.Nx),:] for i in range(self.num_output_fn)]
        out_list = []
        new_four_coef_list = []
        for four_coef in four_coef_list:
            # use constructor to complete matrix
            F_full = construct_full_fourier_matrix(self.Nx, self.Nt, four_coef)
            # Apply low-pass filtering in time dimension only
            t_cutoff = int(self.t_filter_cutoff_ratio * (self.Nt // 2))
            x_cutoff = int(self.x_filter_cutoff_ratio * (self.Nx // 2))
            F_full = F_full.clone()
            F_full[:,:,t_cutoff:-t_cutoff] = 0
            F_full[:,x_cutoff:-x_cutoff,:] = 0
            f_full = torch.fft.ifftn(F_full, dim=(-2, -1))
            check_ifft_real(f_full)
            out_list.append(f_full)
            new_four_coef_list.append(F_full)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies_full_fourier(self, out, four_coef, x, y)

        # print(f'inference projection is {self.inference_projection}')
        if self.inference_projection:
            for i in range(self.num_norm_refinements):
                scaling_factor  = scaling_factor_full_fourier(self, out, four_coef, energies)#is it okay to just apply this to the transformed output?
                out = out * scaling_factor
                four_coef = torch.fft.fftn(out, dim=(-2, -1))
                energies = compute_energies_full_fourier(self, out, four_coef, x, y, in_physical_space=True)

        self.i += 1

        return out.real, list(energies)


class FullFourierNormStrategy():
    def __init__(self, args, net):
        self.net = net
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.Lx = args.xmax - args.xmin
        self.Lt = args.tmax - args.tmin
        self.Nx = args.Nx
        self.Nt = args.Nt
        self.problem = args.problem
        self.num_output_fn = args.num_output_fn
        self.IC = args.IC
        self.i = 0
        self.t_filter_cutoff_ratio = args.t_filter_cutoff_ratio
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.inference_norm = args.inference_norm
        self.num_norm_refinements = args.num_norm_refinements


    def call(self, out, x=None, y=None):
        four_coef = out
        # Combine real and imaginary parts into complex tensor
        four_coef = to_complex_tensor(four_coef)
        four_coef = four_coef.view(-1, self.Nx * self.num_output_fn, (self.Nt // 2) + 1)

        four_coef.view(-1, self.Nx * self.num_output_fn, self.Nt // 2 + 1)

        four_coef_list = [four_coef[:,i*int(self.Nx):(i+1)*int(self.Nx),:] for i in range(self.num_output_fn)]
        out_list = []
        new_four_coef_list = []
        for four_coef in four_coef_list:

            # use constructor to complete matrix
            F_full = construct_full_fourier_matrix(self.Nx, self.Nt, four_coef)
            # Apply low-pass filtering in time dimension only
            t_cutoff = int(self.t_filter_cutoff_ratio * (self.Nt // 2))
            x_cutoff = int(self.x_filter_cutoff_ratio * (self.Nx // 2))
            F_full = F_full.clone()
            F_full[:,:,t_cutoff:-t_cutoff] = 0
            F_full[:,x_cutoff:-x_cutoff,:] = 0
            f_full = torch.fft.ifftn(F_full, dim=(-2, -1))
            check_ifft_real(f_full)
            out_list.append(f_full)
            new_four_coef_list.append(F_full)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies_full_fourier(self, out, four_coef, x, y)

        for i in range(self.num_norm_refinements):
            scaling_factor  = scaling_factor_full_fourier(self, out, four_coef, energies)#is it okay to just apply this to the transformed output?
            out = out * scaling_factor
            four_coef = torch.fft.fftn(out, dim=(-2, -1))
            energies = compute_energies_full_fourier(self, out, four_coef, x, y, in_physical_space=True)

        self.i += 1
        return out.real, list(energies)

class FullFourierAvgNormStrategy():
    def __init__(self, args, net):
        self.net = net
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.Lx = args.xmax - args.xmin
        self.Lt = args.tmax - args.tmin
        self.Nx = args.Nx
        self.Nt = args.Nt
        self.problem = args.problem
        self.num_output_fn = args.num_output_fn
        self.IC = args.IC
        self.i = 0
        self.t_filter_cutoff_ratio = args.t_filter_cutoff_ratio
        self.x_filter_cutoff_ratio = args.x_filter_cutoff_ratio
        self.inference_norm = args.inference_norm

    def call(self, out, x=None, y=None):
        four_coef = out
        # Combine real and imaginary parts into complex tensor
        four_coef = to_complex_tensor(four_coef)
        four_coef = four_coef.view(-1, self.Nx * self.num_output_fn, (self.Nt // 2) + 1)

        four_coef.view(-1, self.Nx * self.num_output_fn, self.Nt // 2 + 1)

        four_coef_list = [four_coef[:,i*int(self.Nx):(i+1)*int(self.Nx),:] for i in range(self.num_output_fn)]
        out_list = []
        new_four_coef_list = []
        for four_coef in four_coef_list:

            # use constructor to complete matrix
            F_full = construct_full_fourier_matrix(self.Nx, self.Nt, four_coef)
            # Apply low-pass filtering in time dimension only
            t_cutoff = int(self.t_filter_cutoff_ratio * (self.Nt // 2))
            x_cutoff = int(self.x_filter_cutoff_ratio * (self.Nx // 2))
            F_full = F_full.clone()
            F_full[:,:,t_cutoff:-t_cutoff] = 0
            F_full[:,x_cutoff:-x_cutoff,:] = 0
            f_full = torch.fft.ifftn(F_full, dim=(-2, -1))
            check_ifft_real(f_full)
            out_list.append(f_full)
            new_four_coef_list.append(F_full)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies_full_fourier(self, out, four_coef, x, y)

        if not self.inference_norm or not self.net.training:
            scaling_factor  = scaling_factor_full_fourier(self, out, four_coef, energies) #is it okay to just apply this to the transformed output?

            out = out * scaling_factor.mean(dim=-1, keepdim=True).expand_as(out)
            four_coef = four_coef * scaling_factor.mean(dim=-1, keepdim=True).expand_as(four_coef)
            energies = compute_energies_full_fourier(self, out, four_coef, x, y)


        # updated_four_coef = torch.fft.fftn(out, dim=(-2, -1))

        # four_coef_list = [updated_four_coef[:,i*int(self.Nx):(i+1)*int(self.Nx),:] for i in range(self.num_output_fn)]
        # out_list = []
        # new_four_coef_list = []
        # for four_coef in four_coef_list:
        #     f_full = torch.fft.ifftn(four_coef, dim=(-2, -1))
        #     check_ifft_real(f_full)
        #     out_list.append(f_full)
        #     new_four_coef_list.append(four_coef)
        # out = torch.cat(out_list, dim=1)  #is this valid
        # four_coef = torch.cat(new_four_coef_list, dim=1) 
        # energies = compute_energies_full_fourier(self, out, updated_four_coef, x, y)
        self.i += 1
        return out.real, list(energies)

def construct_full_fourier_matrix(Nx, Nt, F_half):
    """
    Construct a Hermitian-symmetric matrix efficiently without redundancy.
    
    Args:
        Nx (int): Number of spatial frequencies.
        Nt (int): Number of time frequencies.
    
    Returns:
        F_full (torch.Tensor): Full Hermitian-symmetric matrix of shape (Nx, Nt).
    """

    Nt_half = Nt // 2 + 1
    assert F_half.shape[-1] == Nt // 2 + 1, 'time dimension is off in the half fourier matrix'
    assert F_half.shape[-2] == Nx, 'time dimension is off in the half fourier matrix'

    if Nx % 2 == 0:
        F_half[:,Nx//2+1:,0] = torch.conj(torch.flip(F_half[:,1:Nx//2,0], dims=[1]))
        F_half[:,Nx//2,0] = F_half[:,Nx//2,0].real.clone()  # Nyquist row
        F_half[:,Nx//2+1:, -1] = torch.conj(torch.flip(F_half[:,1:Nx//2,-1], dims=[1]))
    else:
        F_half[:,Nx//2+1:,0] = torch.conj(torch.flip(F_half[:,1:Nx//2+1 ,0], dims=[1]))
        F_half[:,Nx//2+1:, -1] = torch.conj(torch.flip(F_half[:,1:Nx//2+1,-1], dims=[1]))
    
    if Nt % 2 == 0:
        F_half[:,0, -1] = F_half[:,0, -1].real.clone()  # Clone before modifying Nyquist column
        

    F_full = torch.empty(F_half.shape[0],Nx, Nt, dtype=torch.complex64)

    # Copy non-negative frequencies
    F_full[:,:, :Nt_half] = F_half

    if Nt % 2 == 0:
        F_full[:,0,Nt//2+1:] = torch.conj(torch.flip(F_full[:,0,1:Nt//2], dims=[1]))
        F_full[:,1:,Nt//2+1:] = torch.conj(torch.flip(F_half[1:,1:-1],dims=[-2,-1]))
    else:
        F_full[:,0,Nt//2+1:] = torch.conj(torch.flip(F_full[:,0,1:Nt//2 + 1], dims=[0]))
        F_full[:,1:,Nt//2+1:] = torch.conj(torch.flip(F_half[:,1:,1:],dims=[-2,-1]))

    # Ensure real values at (0,0) and Nyquist frequency intersections
    F_full[:,0, 0] = F_full[:,0, 0].real.clone()  # DC component
    if Nx % 2 == 0 and Nt % 2 == 0:
        F_full[:,Nx//2, Nt//2] = F_full[:,Nx//2, Nt//2].real.clone()

    return F_full

def check_ifft_real(f_full):
    """
    Check if the inverse Fourier transform of a Hermitian-symmetric matrix is real.
    
    Args:
        F_full (torch.Tensor): Full Hermitian-symmetric matrix of shape (Nx, Nt).
    
    Returns:
        is_real (bool): True if the inverse Fourier transform is real, False otherwise.
    """
    # Check if the real part is close to zero
    is_real = torch.allclose(f_full.imag, torch.zeros_like(f_full.real), atol=1e-5)
    
    return is_real
