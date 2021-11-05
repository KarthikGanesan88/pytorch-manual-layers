import torch
import torch.nn as nn
import torch.nn.grad
import torch.nn.functional as F
import numpy as np
import cuda_layers
import cpp_layers
import pdb

class convAppx(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, weight, bias, padding=(1, 1), stride=(1, 1)):
        confs = torch.from_numpy(np.array([stride[0], padding[0]]))
        ctx.save_for_backward(X, weight, bias, confs)

        (b, n_C_prev, n_H_prev, n_W_prev) = X.shape
        (n_oC, n_iC, f, f) = weight.shape

        n_H = ((n_H_prev - f + (2 * padding[0])) // stride[0]) + 1
        n_W = ((n_W_prev - f + (2 * padding[0])) // stride[0]) + 1

        X_pad = F.pad(X, (padding[0], padding[0], padding[0], padding[0]))

        if X.is_cuda:
            # This conv layer makes heavy use of torch functions such as unfold below
            # This way a conv layer, just like an FC layer, just becomes a big matrix mult
            # which is really efficient to run on GPU.

            inp_unf = torch.nn.functional.unfold(X_pad, (f, f))
            inp_unf = inp_unf.transpose(1, 2)
            weight = weight.view(weight.size(0), -1).t()

            # The weights must be flattened before calling the CUDA kernel, otherwise
            # the kernel will see them in their original shape.
            inp_unf_flat = inp_unf.flatten()
            weight_flat = weight.flatten()

            out_unf = cuda_layers.conv_forward(inp_unf_flat, weight_flat, bias,
                                               inp_unf.size(1), inp_unf.size(2),
                                               weight.size(1), b)

            out_unf = out_unf.transpose(1, 2)
            out_unf = out_unf.view(b, n_oC, n_H, n_W)
            return out_unf
        else:

            # This does a more 'traditional' stacked for-loop approach, written in C++
            # in case to use sans GPU or to make debug easier.
            output = cpp_layers.conv_forward(X_pad, weight, bias, padding[0], stride[0])
            return output

            # Lastly, this does the entire conv layer directly in python for easier debug.
            # Z = torch.empty(b, n_oC, n_H, n_W, device='cpu')
            # for i in range(b):
            #     for c in range(n_oC):
            #         for h in range(n_H):
            #             for w in range(n_W):
            #                 accumulation = 0.0
            #                 for l in range(n_iC):
            #                     for j in range(f):
            #                         for k in range(f):
            #                             A = X_pad[i, l, (j + h), (k + w)]
            #                             B = weight[c, l, j, k]
            #                             accumulation += A * B
            #                 Z[i, c, h, w] = accumulation + bias[c]
            # return Z

    @staticmethod
    def backward(ctx, grad_output):
        X, weight, bias, confs = ctx.saved_tensors
        confs = confs.numpy()
        stride, padding = confs[0], confs[1]
        grad_input = grad_weight = grad_bias = None

        # IMPORTANT NOTE: This code does the 'standard' backward pass for CONV.
        # This does not take into account the changes made to the forward pass.

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(X.shape, weight, grad_output, stride, padding)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(X, weight.shape, grad_output, stride, padding)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0, 2, 3)).squeeze(0)

        if bias is not None:
            return grad_input, grad_weight, grad_bias, None, None, None, None
        else:
            return grad_input, grad_weight, None, None, None, None, None


class MyConv2d(nn.Module):
    def __init__(self, n_channels, out_channels, kernel_size, padding, stride, dilation=1):
        super(MyConv2d, self).__init__()

        self.kernel_size = (kernel_size, kernel_size)
        self.kernal_size_number = kernel_size * kernel_size
        self.out_channels = out_channels
        self.dilation = (dilation, dilation)
        self.padding = (padding, padding)
        self.stride = (stride, stride)
        self.n_channels = n_channels
        self.weight = nn.Parameter(
            torch.rand(self.out_channels, self.n_channels, self.kernel_size[0], self.kernel_size[1]))
        self.bias = nn.Parameter(torch.rand(self.out_channels))

    def forward(self, x):
        res = convAppx.apply(x, self.weight, self.bias, self.padding, self.stride)

        return res

