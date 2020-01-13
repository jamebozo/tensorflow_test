import torch
import numpy as np

np_data = np.arange(6).reshape((2,3))
tt = torch.from_numpy(np_data)
	
print(np_data)
print(tt)
print(tt.numpy())

#=======================
data = [-1, -2, 1, 2]

tt1 = torch.FloatTensor(data)  # 32bit float

print('\nnumpy: ', np.mean(data),
	  '\ntorch: ', torch.mean(tt1))

#=======================
import torch
import numpy as np

data2 = [[1,2], [3,4]]
tt2 = torch.FloatTensor(data2) # 32-bit floating point

# multiply
print('\nnumpy', np.matmul(data2, data2),
	  '\ntorch', torch.mm(tt2, tt2)
	)