import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

print(torch.linspace(-1,1,100).shape)
x = torch.unsqueeze(torch.linspace(-1,1,100), dim=1)
print(x.shape, x[:10])

y = x.pow(2) + 0.2* torch.rand(x.size())

x, y = Variable(x) , Variable(y)
#plt.scatter(x.data.numpy(), y.data.numpy())
#plt.show()

class Net(torch.nn.Module): # always include when create model
	def __init__(self, n_in, n_hidden, n_out):
		super(Net, self).__init__() # Always do
		self.hidden  = torch.nn.Linear(n_in, n_hidden)
		self.predict = torch.nn.Linear(n_hidden, n_out)
		
	def forward(self, x):
		x = F.relu(self.hidden(x))
		x = self.predict(x)
		# out no activation, 
		# because activation turns result to 0-1, not what we want
		return x

net = Net(1, 10, 1)
print(net)  # show summary
		#pass

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # use sochastc gradient descent
loss_func = torch.nn.MSELoss()  
# Use Mean Square error, good for `Regression Problems

for t in range(100):
	# 1) input 
	prediction = net(x)
	loss = loss_func(prediction, y) # front/back? pred/gt

	# 2) zero
	# backprop accumulates grad every time(RNN), 
	# we don't want that in CNN 
	optimizer.zero_grad()

	# 3) final step 
	loss.backward()
	optimizer.step()

	# optimize gradient.
	optimizer.step()

	# print(loss.data.numpy())
	if t % 5 == 0:
		# plot and show learning progress
		plt.cla() # clear active axes in cur figure, leaves other axes untouch
		plt.scatter(x.data.numpy(), y.data.numpy()) 
		plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
		plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size':20, 'color': 'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()
