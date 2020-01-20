import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-1, 1, 50)  # -1 ~ 1, evenly create 50 points

y2 = x**2
y1 = 2*x+1 # place in fig1

# anything below this will be in figure 1
plt.figure()  
plt.plot(x, y2)


plt.xlim((-1, 2))
plt.ylim((-2, 3))

plt.xlabel('I am x')
plt.ylabel('I am y')

new_ticks = np.linspace(-1,2,5)  # average place x ticks
plt.xticks(new_ticks)   # x-axis

#plt.yticks([-2, -1, 1], ['bad', 'norm', 'good'])  # [pos, str]

#plt.yticks([-1, -1.8, -1], [r'$really\ bad\\alpha', r'$norm', r'$good'])
# anything below this will be in figure 2
#plt.figure(num=3, fig_size=(8,5))   # set fig 3, size is 8,5
#plt.plot(x, y1, color='red', linewidth=1, linestyle='--')

# gca = 'get current axis'
ax = plt.gca()
ax.spines['right'].set_color('none')  # hide right side
ax.spines['top'].set_color('none')    
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

ax.spines['bottom'].set_position(('data', -1 ))  # set x axis to y=-1
ax.spines['left'].set_position(('data', 1))      # set y axis to x=1

plt.show()