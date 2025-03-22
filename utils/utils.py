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
    def call(self, x_func, x_loc,x,y):
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
    def call(self, x_func, x_loc, x, y):
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
        self.K = args.Nx
        self.problem = args.problem
        self.num_output_fn = args.num_output_fn
        self.IC = args.IC
        self.i = 0

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

        four_coef_list = [four_coef[:, i*int(self.K//2 + 1):(i+1)*int(self.K//2+1)] for i in range(self.num_output_fn)]
        out_list = []
        new_four_coef_list = []
        for four_coef in four_coef_list:
            neg_four_coef = torch.conj(torch.flip(four_coef[:, 1:], dims=[1]))
            four_coef[:, 0].imag = 0
            four_coef = torch.cat((four_coef, neg_four_coef), dim=1)
            four_coef_hat = four_coef.unsqueeze(-1) #* self.K
            out = torch.fft.ifft(four_coef_hat.squeeze(-1)) #* self.K
            assert out.imag.abs().max() < 1e-5, f'Imaginary part of fourier coefficients is too large: {four_coef_hat.imag.abs().max()}'
            out_list.append(out)
            new_four_coef_list.append(four_coef_hat)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies(self, out, four_coef, x_func, x_loc, x, y)

        self.i += 1

        return out.real, (B.reshape(-1, B.shape[2] * self.net.num_outputs), un_alpha, energies)
    
class FourierNormStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierNormStrategy, self).__init__(net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.Nx
        self.problem = args.problem
        self.IC = args.IC
        self.num_output_fn = args.num_output_fn
        self.i = 0

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
            four_coef_list = [four_coef[:, i*int(self.K//2 + 1):(i+1)*int(self.K//2+1)] for i in range(self.num_output_fn)]
            prelim_out_list = []
            prelim_four_coef_list = []
            for i, four_coef in enumerate(four_coef_list):
                neg_four_coef = torch.conj(torch.flip(four_coef[:, 1:], dims=[1]))
                four_coef[:, 0].imag = 0
                four_coef = torch.cat((four_coef, neg_four_coef), dim=1)
                four_coef_hat = four_coef.unsqueeze(-1)
                prelim_out = torch.fft.ifft(four_coef_hat.squeeze(-1)) #* self.K
                assert prelim_out.imag.abs().max() < 1e-3, f'Imaginary part of fourier coefficients is too large: {four_coef_hat.imag.abs().max()}'
                prelim_out_list.append(prelim_out)
                prelim_four_coef_list.append(four_coef_hat)
            prelim_out = torch.cat(prelim_out_list, dim=1)
            prelim_four_coef = torch.cat(prelim_four_coef_list, dim=1)

            assert prelim_out.imag.abs().max() < 1e-4, f'Imaginary part of fourier coefficients is too large: {prelim_out.imag.abs().max()}'
            four_coef_final, energies = project_fourier_coefficients(self, prelim_out, prelim_four_coef, x_func, x_loc, x, y)

            four_coef_list_final = [four_coef_final[:, i*int(four_coef_final.shape[1]//self.num_output_fn):(i+1)*int(four_coef_final.shape[1]//self.num_output_fn)] for i in range(self.num_output_fn)]
            out_list = []
            for four_coef in four_coef_list_final:
                out = torch.fft.ifft(four_coef, dim=1) #* self.K
                assert out.imag.abs().max() < 1e-3, f'Imaginary part of fourier coefficients is too large: {out.imag.abs().max()}'
                out_list.append(out)
            out = torch.cat(out_list, dim=1)

            energies = compute_energies(self, out, four_coef_final, x_func, x_loc, x, y)

            # fourier_energy = torch.sum(torch.abs(four_coef_hat) ** 2, dim=1, keepdim=True) * torch.tensor(self.L)

            self.i += 1

            return out.real, (B.reshape(-1, B.shape[2] * self.net.num_outputs), un_alpha, energies)
        
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


class FourierQRStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierQRStrategy, self).__init__(net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.Nx
        self.problem = args.problem

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
    
    def call(self, x_func, x_loc, L=2 * np.pi):

        branch_out_in = self.net.branch(x_func) 

        branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
        B = to_complex_tensor(branch_out_in)
        un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
        un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

        Q, R = torch.linalg.qr(B)
        
        # For complex unitary matrices, we need Q^H Q = I where Q^H is the conjugate transpose
        if not torch.allclose(torch.complex(torch.eye(Q.shape[2]), torch.zeros(Q.shape[2])), torch.bmm(Q.conj().transpose(1,2), Q), atol=1e-6):
            raise AssertionError("Q is not unitary")

        #check that QR=B
        if not torch.allclose(Q @ R, B, atol=1e-5):
            raise AssertionError("QR decomposition does not equal B")
        
        B_hat = Q

        un_alpha = (R @ un_alpha.unsqueeze(-1)).squeeze(-1)

        if (x_func.shape[1]<3):
            N = 500
            dx = self.L / N
            a, c = x_func[:,0], x_func[:,1]
            x = torch.linspace(-self.L/2, self.L/2, N).unsqueeze(1)
            u0 = exact_soliton(x, 0, c, a)
            true_nrg = torch.sum(torch.abs(u0)**2, dim=0) * dx
            # print(true_nrg.shape)
        else:
            raise NotImplementedError('Energy calculation for other problems than 1d KdV not implemented')

        
        alpha_hat = project_fourier_coefficients(self.problem, un_alpha, true_nrg.unsqueeze(-1), self.L)

        four_coef = B_hat @ alpha_hat.unsqueeze(-1) #N x  M+1 x 1
        neg_four_coef = torch.conj(torch.flip(four_coef[:, 1:], dims=[1]))  # Flip to maintain Hermitian symmetry
        four_coef[:, 0].imag = 0  # Ensure DC is real
        four_coef = torch.cat((four_coef, neg_four_coef), dim=1)

        out = torch.fft.ifft(four_coef.squeeze(-1)) * self.K
        assert out.imag.abs().max() < 1e-5, f'Imaginary part of fourier coefficients is too large: {four_coef.imag.abs().max()}'

        fourier_energy = torch.sum(torch.abs(four_coef) ** 2, dim=1, keepdim=True) * torch.tensor(self.L)

        return out.real, (B.reshape(-1, B.shape[2] * self.net.num_outputs), un_alpha, fourier_energy)

def compute_energies(self, prelim_out, four_coef, x_func, x_loc, x, y):
    if self.problem == '1d_KdV_Soliton':
        N = 500
        dx = self.L / N
        a, c = x_func[:,0], x_func[:,1]
        x = torch.linspace(-self.L/2, self.L/2, N).unsqueeze(1)
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
            ut = torch.autograd.grad(outputs=prelim_out, inputs=x_loc, grad_outputs=torch.ones_like(prelim_out), create_graph=True, allow_unused=True)[0] #allow_unused?
            ut_hat = torch.fft.fft(ut, n=self.K, dim=1)

            current_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) * self.L / (self.K ** 2)
            current_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) * self.L / (self.K ** 2)
            # print(f'current energy ut component: {current_energy_ut_component.mean().item()}')
            # print(f'current energy u component: {current_energy_u_component.mean().item()}') 

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
            target_energy = torch.sum(torch.abs(gt_ut_hat)**2 + self.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) * self.L / (self.K ** 2)

        assert (true_energy-target_energy).abs().max() < 1, f'max difference between true energy and target energy is {(true_energy-target_energy).abs().max().item()}'
    else:
        raise NotImplementedError('Energy calculation for other problems than 1d wave not implemented')
    
    energy_components = {'true_energy_u_component': true_energy_u_component, 'true_energy_ut_component': true_energy_ut_component, 
                         'current_energy_u_component': current_energy_u_component, 'current_energy_ut_component': current_energy_ut_component,
                         'learned_energy_u_component': learned_energy_u_component, 'learned_energy_ut_component': learned_energy_ut_component}
        
    return true_energy, current_energy, learned_energy, energy_components

def compute_energies_full_fourier(self, prelim_out, four_coef, og_x, og_y):
    if self.problem == '1d_wave':
        #compute 1d target energy:
        print(og_x[0,0,:].shape)
        print(og_y[0,0,:,0].shape)


        # plt.plot(og_x[0,0,:].detach().numpy())
        # plt.plot(og_y[0,0,:,0].detach().numpy())
        # plt.show()




        y = og_y[:,0,:,:]
        yt = og_y[:,1,:,:]

        print(self.Nx)

        wave_k = torch.fft.fftfreq(self.Nx, d=(self.Lx/self.Nx)) * 2 * np.pi  # Wave numbers

        print(wave_k.shape)

        print(y.shape)

        u_hat = torch.fft.fft(y, dim=1)
        ut_hat = torch.fft.fft(yt, dim=1)

        print(u_hat.shape)

        og_target_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)
        og_target_energy_u_component = torch.sum(self.IC['c']**2 * (wave_k.unsqueeze(0).unsqueeze(-1).expand(u_hat.shape) ** 2) * torch.abs(u_hat)**2, dim=1) * self.Lx / (self.Nx ** 2)
        og_target_energy = og_target_energy_ut_component + og_target_energy_u_component

        # kx = torch.tensor(np.fft.fftfreq(self.Nx, d=(self.Lx/self.Nx)) * 2* np.pi).float() 
        # kt = torch.tensor(np.fft.fftfreq(self.Nt, d=(self.Lt/self.Nt)) * 2* np.pi).float()
        
        target_four_coef = torch.fft.fftn(y, dim=(1,2))
        print(target_four_coef.shape)

        # --- User-provided inputs ---
        # four_coef: tensor of shape (N, Nx, Nt) containing the Fourier coefficients
        # L_x, L_t: the spatial and temporal domain lengths (floats)
        # For example, they might be defined as:
        # L_x, L_t = 2 * torch.pi, 1.0

        N, Nx, Nt = target_four_coef.shape

        # Create the time grid. For a periodic domain we use Nt equispaced points in [0, L_t).
        # Note: torch.linspace includes the endpoint, so we use arange to avoid it.
        t = torch.arange(Nt, device=target_four_coef.device) * (self.Lt / Nt)  # shape: (Nt,)

        # Create an index tensor for the time Fourier modes.
        m = torch.arange(Nt, device=target_four_coef.device)  # shape: (Nt,)

        # Construct the exponential matrix for the inverse transform in time.
        # Its (n, m)-th entry is exp(2π i * m * t_n / L_t).
        # This matrix has shape (Nt, Nt).
        exp_matrix = torch.exp(2 * torch.pi * 1j * t.unsqueeze(1) * m.unsqueeze(0) / self.Lt)

        # --- Inverse transform in time (for each batch and spatial mode) ---
        # For each batch sample and spatial mode, we compute:
        #   f_k(t_n) = (1/Nt) * sum_{m=0}^{Nt-1} four_coef[n, k, m] * exp(2π i * m * t_n / L_t)
        # We can perform this for all batches at once by using torch.matmul.
        # The multiplication is along the time dimension.
        inv_time = (1.0 / Nt) * torch.matmul(target_four_coef, exp_matrix.T)  # shape: (N, Nx, Nt)

        # --- Compute the Fourier coefficients for the time derivative u_t ---
        # In Fourier space, differentiation in time multiplies each mode by 2π i m / L_t.
        # Create the multiplier for time modes.
        multiplier_t = 2 * torch.pi * 1j * (m / self.Lt)  # shape: (Nt,)
        # Multiply the Fourier coefficients along the time (last) dimension.
        four_coef_t_mult = target_four_coef * multiplier_t[None, None, :]  # shape: (N, Nx, Nt)
        # Now compute the inverse transform in time with the differentiated coefficients.
        u_t_fourier_coeff = (1.0 / Nt) * torch.matmul(four_coef_t_mult, exp_matrix.T)  # shape: (N, Nx, Nt)

        # --- Compute the Fourier coefficients for the spatial derivative u_x ---
        # The inverse time transform (without differentiation) has been computed in inv_time.
        # Differentiation in space multiplies the k-th mode by 2π i k / L_x.
        # Create the multiplier for spatial modes.
        k = torch.arange(Nx, device=four_coef.device)  # shape: (Nx,)
        multiplier_x = 2 * torch.pi * 1j * (k / self.Lx)  # shape: (Nx,)
        # Multiply each spatial mode of inv_time by its corresponding multiplier.
        u_x_fourier_coeff = inv_time * multiplier_x[None, :, None]  # shape: (N, Nx, Nt)

        target_energy_ut_component = torch.sum(torch.abs(u_t_fourier_coeff)**2, dim=1) * self.Lx / (self.Nx ** 2)
        target_energy_ux_component = torch.sum(self.IC['c']**2 * torch.abs(u_x_fourier_coeff)**2, dim=1) * self.Lx / (self.Nx ** 2)

        target_energy = target_energy_ut_component + target_energy_ux_component

        
        if self.num_output_fn == 1:
            
            # --- User-provided inputs ---
            # four_coef: tensor of shape (N, Nx, Nt) containing the Fourier coefficients
            # L_x, L_t: the spatial and temporal domain lengths (floats)
            # For example, they might be defined as:
            # L_x, L_t = 2 * torch.pi, 1.0

            N, Nx, Nt = four_coef.shape

            # Create the time grid. For a periodic domain we use Nt equispaced points in [0, L_t).
            # Note: torch.linspace includes the endpoint, so we use arange to avoid it.
            t = torch.arange(Nt, device=four_coef.device) * (self.Lt / Nt)  # shape: (Nt,)

            # Create an index tensor for the time Fourier modes.
            m = torch.arange(Nt, device=four_coef.device)  # shape: (Nt,)

            # Construct the exponential matrix for the inverse transform in time.
            # Its (n, m)-th entry is exp(2π i * m * t_n / L_t).
            # This matrix has shape (Nt, Nt).
            exp_matrix = torch.exp(2 * torch.pi * 1j * t.unsqueeze(1) * m.unsqueeze(0) / self.Lt)

            # --- Inverse transform in time (for each batch and spatial mode) ---
            # For each batch sample and spatial mode, we compute:
            #   f_k(t_n) = (1/Nt) * sum_{m=0}^{Nt-1} four_coef[n, k, m] * exp(2π i * m * t_n / L_t)
            # We can perform this for all batches at once by using torch.matmul.
            # The multiplication is along the time dimension.
            inv_time = (1.0 / Nt) * torch.matmul(four_coef, exp_matrix.T)  # shape: (N, Nx, Nt)

            # --- Compute the Fourier coefficients for the time derivative u_t ---
            # In Fourier space, differentiation in time multiplies each mode by 2π i m / L_t.
            # Create the multiplier for time modes.
            multiplier_t = 2 * torch.pi * 1j * (m / self.Lt)  # shape: (Nt,)
            # Multiply the Fourier coefficients along the time (last) dimension.
            four_coef_t_mult = four_coef * multiplier_t[None, None, :]  # shape: (N, Nx, Nt)
            # Now compute the inverse transform in time with the differentiated coefficients.
            u_t_fourier_coeff = (1.0 / Nt) * torch.matmul(four_coef_t_mult, exp_matrix.T)  # shape: (N, Nx, Nt)

            # --- Compute the Fourier coefficients for the spatial derivative u_x ---
            # The inverse time transform (without differentiation) has been computed in inv_time.
            # Differentiation in space multiplies the k-th mode by 2π i k / L_x.
            # Create the multiplier for spatial modes.
            k = torch.arange(Nx, device=four_coef.device)  # shape: (Nx,)
            multiplier_x = 2 * torch.pi * 1j * (k / self.Lx)  # shape: (Nx,)
            # Multiply each spatial mode of inv_time by its corresponding multiplier.
            u_x_fourier_coeff = inv_time * multiplier_x[None, :, None]  # shape: (N, Nx, Nt)

            print(u_x_fourier_coeff.shape)

            current_energy_ut_component = torch.sum(torch.abs(u_t_fourier_coeff)**2, dim=1) * self.Lx / (self.Nx ** 2)
            current_energy_ux_component = torch.sum(self.IC['c']**2 * torch.abs(u_x_fourier_coeff)**2, dim=1) * self.Lx / (self.Nx ** 2)

            current_energy = current_energy_ut_component + current_energy_ux_component

            learned_energy_ut_component = None
            learned_energy_ux_component = None
            learned_energy = None

            # k = torch.tensor(np.fft.fftfreq(Nx, d=(self.Lx/self.Nx)) * 2* np.pi).float()  # to device?

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
                         'target_energy': target_energy, 'target_energy_ux_component': target_energy_ux_component, 'target_energy_ut_component': target_energy_ut_component,
                            'og_target_energy': og_target_energy, 'og_target_energy_ux_component': og_target_energy_u_component, 'og_target_energy_ut_component': og_target_energy_ut_component}

        print(f'true energy {true_energy.mean()}')
        print(f'current energy {current_energy.mean()}')
        print(f'true energy ut component {true_energy_ut_component.mean()}')
        print(f'current energy ut component {current_energy_ut_component.mean()}')
        print(f'true energy ux component {true_energy_ux_component.mean()}')
        print(f'current energy ux component {current_energy_ux_component.mean()}')
        print(f'og target energy {og_target_energy.mean()}')

    return true_energy, current_energy, learned_energy, energy_components





        



def project_fourier_coefficients(self, prelim_out, four_coef, x_func, x_loc, x, y) -> torch.Tensor:
    """
    Projects a batch of Fourier coefficients onto the manifold ∑|c_i|² = C.

    Args:
        four_coef (torch.Tensor): Tensor of shape [batch_size, num_fourier_modes]
                                  representing Fourier coefficients.
        C (float): Desired L2 norm squared (energy level).

    Returns:
        torch.Tensor: Projected Fourier coefficients of the same shape.
    """
    true_energy, current_energy, learned_energy, energy_components = compute_energies(self, prelim_out, four_coef, x_func, x_loc, x, y)

    if self.num_output_fn == 1:
        norming_energy = current_energy
    elif self.num_output_fn == 2:
        norming_energy = learned_energy
    # norming_energy = current_energy if learned_energy is None else learned_energy

    # print(norming_energy.shape)
    # print(true_energy.shape)
    mask = (norming_energy == 0)
    scaling_factor = torch.zeros_like(true_energy)
    scaling_factor[~mask] = torch.sqrt(true_energy[~mask] / norming_energy[~mask])
    scaling_factor = scaling_factor[:,None]
    # print(scaling_factor)

    # Check if scaling factor is complex
    assert not torch.is_complex(scaling_factor), "Scaling factor is complex"

    # print(scaling_factor.shape)

    four_coef_proj = four_coef.squeeze(-1) * scaling_factor.expand(-1, four_coef.shape[1]) 

    # _, current_energy, _ = compute_energies(self, prelim_out, four_coef_proj, x_func, x_loc, x, y)

    # if self.i > 1:
    #     assert (learned_energy - true_energy).abs().max() < 1, f'Energy is not conserved after projection – max difference was: {(learned_energy - true_energy).abs().max().item()}'

    return four_coef_proj, (true_energy, current_energy, learned_energy, energy_components)

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
    

def padded_ifft(four_coef_hat, K):
    """
    Compute the inverse Fourier transform with zero-padding to match a target resolution K.
    Handles batched input tensors.
    
    Parameters:
        four_coef_hat (torch.Tensor): Tensor of shape (batch_size, N) containing Fourier coefficients
        K (int): The target number of points for the output resolution
    
    Returns:
        torch.Tensor: The reconstructed function sampled at K points, shape (batch_size, K)
    """
    batch_size, N = four_coef_hat.shape  # Original number of Fourier modes

    # Zero-padding in Fourier space
    if K > N:
        pad_left = (K - N) // 2
        pad_right = K - N - pad_left
        four_coef_hat_padded = torch.nn.functional.pad(four_coef_hat, (pad_left, pad_right), mode='constant', value=0)
    else:
        # If K <= N, truncate the higher frequencies
        four_coef_hat_padded = four_coef_hat[:, :K]

    # Compute the inverse FFT
    out = torch.fft.ifft(four_coef_hat_padded) * (K / N)  # Adjust scaling factor

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

            f_full = torch.fft.ifftn(F_full, dim=(-2, -1))
            check_ifft_real(f_full)
            out_list.append(f_full)
            new_four_coef_list.append(F_full)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies_full_fourier(self, out, four_coef, x, y)

        self.i += 1

        return out.real, (energies)


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

            f_full = torch.fft.ifftn(F_full, dim=(-2, -1))
            check_ifft_real(f_full)
            out_list.append(f_full)
            new_four_coef_list.append(F_full)
        out = torch.cat(out_list, dim=1)
        four_coef = torch.cat(new_four_coef_list, dim=1)
        energies = compute_energies_full_fourier(self, out, four_coef, x, y)

        scaling_factor  = scaling_factor_full_fourier(self, out, four_coef, energies) #is it okay to just apply this to the transformed output?

        # four_coef_list = [updated_four_coef[:,i*int(self.Nx):(i+1)*int(self.Nx),:] for i in range(self.num_output_fn)]
        # out_list = []
        # for four_coef in four_coef_list:
        #     f_full = torch.fft.ifftn(four_coef, dim=(-2, -1))
        #     check_ifft_real(f_full)
        #     out_list.append(f_full)
        out = torch.cat(out_list, dim=1) * scaling_factor #is this valid
        # energies = compute_energies_full_fourier(self, out, four_coef, x, y)
        self.i += 1
        return out.real, energies

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
