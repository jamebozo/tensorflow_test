import torch
from torch.autograd import Variable

""" This tutorial teaches:
1) Variable is like a node in pyTorch, 
   Tensors are put in Variables, we could use
   variable.data to retrieve them

2) Variabels have ".grad", 
   by `requires_grad`, we could enable/disable them

3) Tensor / Variable arithmatic is the same,
   only Variable has ".grad"  

4) Optional:
   mean(t*t) = (t*t)/4,  because tensor has 4 elems
   d out /d t = t / 2 = [0.5, 1] [ 1.5, 2]
"""
tensor = torch.FloatTensor([[1,2],[3,4]])
variable = Variable(tensor, requires_grad=True)

t_out = torch.mean(tensor*tensor)
v_out = torch.mean(variable*variable)

print(t_out)
print(v_out)

v_out.backward()
print(variable.grad)
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4*2*variable = variable / 2
# [0.5, 1] [1.5, 2]

print(variable)
print(variable.data)
# print(variable.numpy())  #xxxx
print(variable.data.numpy())
