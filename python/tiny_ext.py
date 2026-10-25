import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r'''
#include <torch/extension.h>
__global__ void add_one_kernel(const float* x,float* y,int n){
int i = blockIdx.x * blockDim.x + threadIdx.x;
if(i < n){
y[i] = x[i] + 1.0f;
}
}

torch::Tensor add_one(torch::Tensor x){
auto y = torch::empty_like(x);
int n = x.numel();
add_one_kernel<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
return y;
}

'''

cpp_src = "torch::Tensor add_one(torch::Tensor x);"

ext = load_inline(name="tiny_ext", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["add_one"])
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
print(ext.add_one(x))
