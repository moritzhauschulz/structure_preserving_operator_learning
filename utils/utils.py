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
        self.K = args.col_N
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
    
def compute_energies(self, prelim_out, four_coef, x_func, x_loc, x, y):
    if self.problem == '1d_KdV_Soliton':
        N = 500
        dx = self.L / N
        a, c = x_func[:,0], x_func[:,1]
        x = torch.linspace(-self.L/2, self.L/2, N).unsqueeze(1)
        u0 = exact_soliton(x, 0, c, a)
        true_energy = torch.sum(torch.abs(u0)**2, dim=0) * dx
        # Compute the current L2 norm squared per batch
        current_energy = (torch.sum(torch.abs(four_coef) ** 2, dim=1, keepdim=True) * torch.tensor(self.L)).squeeze(-1).squeeze(-1)
        # print(current_energy.shape)
        learned_energy = None
    elif self.problem == '1d_wave':

        if self.num_output_fn == 1:

            #generate wave numbers
            k = torch.tensor(np.fft.fftfreq(self.K, d=(self.L/self.K)) * 2* np.pi).float()  # to device?

            #compute current energy
            ut = torch.autograd.grad(outputs=prelim_out, inputs=x_loc, grad_outputs=torch.ones_like(prelim_out), create_graph=True, allow_unused=True)[0] #allow_unused?
            ut_hat = torch.fft.fft(ut, n=self.K, dim=1)
            u_hat = four_coef.squeeze(-1)

            # print(u_hat.shape)
            # print(ut_hat.shape)

            current_energy = torch.sum(torch.abs(ut_hat)**2 + self.IC['c']**2 * torch.abs(k * u_hat)**2, dim=1) / self.K
            if self.i % 100 == 0:
                current_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) / self.K
                current_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) / self.K
                print(f'cur energy ut component: {current_energy_ut_component.mean().item()}')
                print(f'cur energy u component: {current_energy_u_component.mean().item()}') 
            
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
            learned_energy = torch.sum(torch.abs(ut_hat)**2 + self.IC['c']**2 * torch.abs(k * u_hat)**2, dim=1) / self.K 
            if self.i % 100 == 0:
                learned_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) / self.K
                learned_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) / self.K
                print(f'learned energy ut component: {learned_energy_ut_component.mean().item()}')
                print(f'learned energy u component: {learned_energy_u_component.mean().item()}')

            #compute current energy
            ut = torch.autograd.grad(outputs=prelim_out, inputs=x_loc, grad_outputs=torch.ones_like(prelim_out), create_graph=True, allow_unused=True)[0] #allow_unused?
            ut_hat = torch.fft.fft(ut, n=self.K, dim=1)

            current_energy = torch.sum(torch.abs(ut_hat)**2 + self.IC['c']**2 * torch.abs(k * u_hat)**2, dim=1) / self.K 
            if self.i % 100 == 0:
                current_energy_ut_component = torch.sum(torch.abs(ut_hat)**2, dim=1) / self.K
                current_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(u_hat)**2, dim=1) / self.K
                print(f'cur energy ut component: {current_energy_ut_component.mean().item()}')
                print(f'cur energy u component: {current_energy_u_component.mean().item()}') 
            

        if x is not None:
            gt_u = x[0][:,0,:]
            gt_ut = x[0][:,1,:]

            k = torch.tensor(np.fft.fftfreq(self.K, d=(self.L/self.K)) * 2* np.pi).float()  # to device?


            gt_u_hat = torch.fft.fft(gt_u, n=self.K, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=self.K, dim=1)
            true_energy = torch.sum(torch.abs(gt_ut_hat)**2 + self.IC['c']**2 * (k ** 2) * torch.abs(gt_u_hat)**2, dim=1) / self.K

            # print(f'init energy is {true_energy[0]}')
        
            # #compute via traezoid from function values directly
            # gt_u = x[0][:,0,:]
            # gt_ut = x[0][:,1,:]
            # print(gt_u.shape)
            # direct_true_energy = torch.sum(torch.abs(gt_ut)**2 + self.IC['c']**2 * torch.abs(gt_u)**2, dim=1) * self.L / self.K

            # assert (true_energy-direct_true_energy).abs().max() < 1, f'max difference between fourier true energy and direct true energy is {(true_energy-direct_true_energy).abs().max().item()}'


        if y is not None:
            gt_u = y[:,0,:]
            gt_ut = y[:,1,:]

            if self.i % 100 == 0:
                true_energy_ut_component = torch.sum(torch.abs(gt_ut_hat)**2, dim=1) / self.K
                true_energy_u_component = torch.sum(self.IC['c']**2 * (k ** 2) * torch.abs(gt_u_hat)**2, dim=1) / self.K
                print(f'true energy ut component: {true_energy_ut_component.mean().item()}')
                print(f'true energy u component: {true_energy_u_component.mean().item()}') 

            gt_u_hat = torch.fft.fft(gt_u, n=self.K, dim=1)  # dim=1 since shape is (time, space)
            gt_ut_hat = torch.fft.fft(gt_ut, n=self.K, dim=1)
            target_energy = torch.sum(torch.abs(gt_ut_hat)**2 + self.IC['c']**2 * torch.abs(k * gt_u_hat)**2, dim=1) / self.K
            # print(f'target energy is {target_energy[0]}')

        assert (true_energy-target_energy).abs().max() < 1, f'max difference between true energy and target energy is {(true_energy-target).abs().max().item()}'

        
    else:
        raise NotImplementedError('Energy calculation for other problems than 1d wave not implemented')
    
    return true_energy, current_energy, learned_energy

    
class FourierNormStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierNormStrategy, self).__init__(net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.col_N
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

            # if self.problem == '1d_KdV_Soliton':
            #     fourier_energy = torch.sum(torch.abs(four_coef_hat)**2, dim=1) * torch.tensor(self.L)
            # elif self.problem == '1d_wave':
            #     fourier_energy = None

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
        elif self.problem == '1d_KdV_Soliton':
            branch_out_in = self.net.branch(x_func) 
        
            branch_out_in = branch_out_in.view(-1, self.net.num_outputs, branch_out_in.shape[1] // self.net.num_outputs) #N x (M + 1) x 2K ; K should be twice the number of outputs because we have real and imaginary part
            B = to_complex_tensor(branch_out_in)
            un_alpha = self.net.activation_trunk(self.net.trunk(x_loc[:,0].unsqueeze(-1))) #only apply to time dimension
            un_alpha = torch.complex(un_alpha, torch.zeros_like(un_alpha))

            four_coef = (B @ un_alpha.unsqueeze(-1)).squeeze(-1) #N x  M+1 
            neg_four_coef = torch.conj(torch.flip(four_coef[:, 1:], dims=[1]))  # Flip to maintain Hermitian symmetry
            four_coef[:, 0].imag = 0  # Ensure DC is real
            four_coef = torch.cat((four_coef, neg_four_coef), dim=1)

            if (x_func.shape[1]<3):
                N = 500
                dx = self.L / N
                a, c = x_func[:,0], x_func[:,1]
                x = torch.linspace(-self.L/2, self.L/2, N).unsqueeze(1)
                u0 = exact_soliton(x, 0, c, a)
                true_nrg = torch.sum(torch.abs(u0)**2, dim=0) * dx
            else:
                raise NotImplementedError('Energy calculation for other problems than 1d KdV not implemented')

            four_coef_hat = old_project_fourier_coefficients(four_coef, true_nrg.unsqueeze(-1), self.L)

            out = torch.fft.ifft(four_coef_hat.squeeze(-1)) * self.K
            assert out.imag.abs().max() < 1e-5, f'Imaginary part of fourier coefficients is too large: {four_coef_hat.imag.abs().max()}'

            fourier_energy = torch.sum(torch.abs(four_coef_hat) ** 2, dim=1, keepdim=True) * torch.tensor(self.L)

            return out.real, (B.reshape(-1, B.shape[2] * self.net.num_outputs), un_alpha, fourier_energy)


class FourierQRStrategy(CustomStrategy):
    def __init__(self, args, net):
        super(FourierQRStrategy, self).__init__(net)
        assert args.xmin == -args.xmax, 'x_min and x_max must be symmetric'
        self.L = args.xmax - args.xmin
        self.K = args.col_N
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
        # norms = torch.norm(un_alpha, p=2, dim=1, keepdim=True)
        # zero_mask = (norms == 0)
        # norms = norms + zero_mask.float()  #TODO: Find a better solution to this!

        
        # alpha_hat = un_alpha / norms

        # if not torch.allclose(torch.ones(alpha_hat.shape[0])[~zero_mask.squeeze(1)], torch.norm(alpha_hat, p=2, dim=1, keepdim=True)[~zero_mask.squeeze(1)]):
        #     raise AssertionError("Trunk output is not normalized")
        # if (zero_mask.squeeze(1)).sum() > B_hat.shape[0] * 0.001 and self.net.epoch > 10:
        #     raise AssertionError(f"Had to exclude more than 0.1% ({(zero_mask.squeeze(1)).sum()/B_hat.shape[0]}) of the batch at the normalizations stage")
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

def old_project_fourier_coefficients(four_coef: torch.Tensor, C: float, L: float = 2 * np.pi) -> torch.Tensor:
    """
    Projects a batch of Fourier coefficients onto the manifold ∑|c_i|² = C.

    Args:
        four_coef (torch.Tensor): Tensor of shape [batch_size, num_fourier_modes]
                                  representing Fourier coefficients.
        C (float): Desired L2 norm squared (energy level).

    Returns:
        torch.Tensor: Projected Fourier coefficients of the same shape.
    """
    # Compute the current L2 norm squared per batch
    current_energy = torch.sum(torch.abs(four_coef) ** 2, dim=1, keepdim=True) * torch.tensor(L)

    # Avoid division by zero (replace zero-norm cases with ones)
    zero_mask = (current_energy == 0)
    current_energy = torch.where(zero_mask, torch.ones_like(current_energy), current_energy)

    # Compute the scaling factor
    scaling_factor = torch.sqrt(C / current_energy)

    # Apply projection
    four_coef_proj = four_coef * scaling_factor

    return four_coef_proj

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
    true_energy, current_energy, learned_energy = compute_energies(self, prelim_out, four_coef, x_func, x_loc, x, y)

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

    return four_coef_proj, (true_energy, current_energy, learned_energy)
    

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

