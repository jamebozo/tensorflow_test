import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)  # -1 ~ 1, evenly create 50 points

new_ticks = np.linspace(-1, 2, 5)
plt.xticks(new_ticks)
plt.yticks([-2, -1.8, -1, 1.22, 3])


plt.xlim((-1,2))
plt.ylim((-2,3))

y2 = x**2
y1 = 2*x+1 # place in fig1

# anything below this will be in figure 1
#plt.figure()  
l1, = plt.plot(x, y2, label='up')
l2, = plt.plot(x, y1, label='down', color='red')
#plt.legend()
plt.legend(handles=[l1, l2,], loc='best', labels=['aaa', 'bbb'])

plt.show()