import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5,5,200)
x = Variable(x)

x_np = x.data.numpy()

y_relu    = F.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh    = torch.tanh(x).data.numpy()
y_softplus = F.softplus(x).data.numpy()  # percentage, not graph

plt.subplot(221)
plt.plot(x_np, y_relu, c='red', label='relu')
#plt.ylim(-1,5)
plt.legend(loc='best')

plt.subplot(222)
plt.plot(x_np, y_sigmoid, c='red', label='sigmoid')
#plt.ylim(-1,5)
plt.legend(loc='best')


plt.subplot(223)
plt.plot(x_np, y_tanh, c='red', label='tanh')
plt.ylim(-1,5)
plt.legend(loc='best')


plt.subplot(224)
plt.plot(x_np, y_softplus, c='red', label='softplus')
plt.ylim(-1,5)
plt.legend(loc='best')

plt.show()