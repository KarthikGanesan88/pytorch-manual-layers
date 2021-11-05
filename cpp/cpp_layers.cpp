#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <iostream>
#include <iomanip>
#include <cmath>

namespace F = torch::nn::functional;
namespace py = pybind11;

torch::Tensor conv_forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    int padding,
    int stride
){

    int m = input.size(0);          // Number of inputs in batch
    int n_H_prev = input.size(2);   // Height of input image
    int n_W_prev = input.size(3);   // Width of input image
    int n_oC = weight.size(0);       // Number of output channels
    int n_iC = weight.size(1);       // Number of input channels
    int f = weight.size(2);         // Size of fxf filter.

    int n_H = ((n_H_prev - f + 2 * padding) / stride) + 1;
    int n_W = ((n_W_prev - f + 2 * padding) / stride) + 1;

    torch::Tensor X_pad = F::pad(input, F::PadFuncOptions({padding,padding,padding,padding}));

    torch::Tensor output = torch::empty({m, n_oC, n_H, n_W} /*, torch::kDouble*/);

    for (int i=0; i<m; i++){                      //Input batches
      for (int c=0; c<n_oC; c++){                 //Output Channels
        for (int h=0; h<n_H; h++){                //Height
          for (int w=0; w<n_W; w++){              //Width
            float accumulation = 0;
            for (int l=0; l<n_iC; l++){           //Input channels
              for (int j=0; j<f; j++){
                for (int k=0; k<f; k++){
                  float A = X_pad.index({i,l,j+h,k+w}).item<float>();
                  float B = weight.index({c,l,j,k}).item<float>();
                  accumulation += A*B;
                }
              }
            }
            //Set output which is the sum over the window.
            output.index({i,c,h,w}) = accumulation + bias.index({c}).item<float>();
          }
        }
      }
    }
    return output;
}


torch::Tensor linear_forward(
  torch::Tensor input,
  torch::Tensor weight,
  torch::Tensor bias
){

  // Create a new transposed weight tensor.
  auto weight_transposed = torch::transpose(weight, 0, 1);
  auto output = torch::empty({input.size(0),weight.size(0)} /*, torch::kDouble*/);

  float accumulation = 0;
  for (int k=0; k<input.size(0); k++){
    for (int l=0; l<weight_transposed.size(1); l++){
      accumulation = 0;
      for (int j=0; j<input.size(1); j++){
        float A = input.index({k,j}).item<float>();
        float B = weight_transposed.index({j,l}).item<float>();
        accumulation += (A*B);
      }
      output.index({k,l}) = accumulation + bias.index({l}).item<float>();
    }
  }
  return (output);

}

PYBIND11_MODULE(cpp_layers, m) {
    m.doc() = "Implementation of forward pass of conv and linear layers in C++";
    m.def("linear_forward", &linear_forward, "linear_forward");
    m.def("conv_forward", &conv_forward, "conv_forward");
}
