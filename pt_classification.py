import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

#========================
n_data = torch.ones(100,2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data,1)
y1 = torch.ones(100)

# Default x train = 32bit float, y label = 64bit int
x = torch.cat((x0,x1), 0).type(torch.FloatTensor) # 32bit float
y = torch.cat((y0,y1), ).type(torch.LongTensor)   # 64bit int

# x, y = Variable(x) , Variable(y)

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

net = Net(2, 10, 2)
print(net)  # show summary
		#pass

plt.ion()
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # use sochastc gradient descent
loss_func = torch.nn.CrossEntropyLoss()  
# Use Mean Square error, good for `Regression Problems

for t in range(100):
	# 1) input 
	out = net(x) # F.softmax(out)
	loss = loss_func(out, y) # front/back? pred/gt

	# 2) zero
	# backprop accumulates grad every time(RNN), 
	# we don't want that in CNN 
	optimizer.zero_grad()

	# 3) final step 
	loss.backward()
	optimizer.step()

	
	# print(loss.data.numpy())
	if t % 2 == 0:
		#out = net(x)
		# plot and show learning progress
		plt.cla() # clear active axes in cur figure, leaves other axes untouch
		#out = F.softmax(out)
		prediction = torch.max(out,1)[1] # np argmax, axis = 1
		prd_y = prediction.data.numpy()
		trg_y = y.data.numpy()
		plt.scatter(x.data.numpy()[:,0], x.data.numpy()[:,1], c=prd_y, s=100, lw=0, cmap='RdYlGn') 
		accuracy = float((prd_y == trg_y).astype(int).sum()) / float(trg_y.size)
		
		plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color': 'red'})
		plt.pause(0.1)

plt.ioff()
plt.show()
