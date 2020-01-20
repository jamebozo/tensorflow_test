import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0,10,0.1)
y1 = 0.05 * x**2 # green
y2 = -1 * y1     # blue 

fix, ax1 = plt.subplots()
ax2 = ax1.twinx()  # mirror x 
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b--')

ax1.set_xlabel('X data')
ax1.set_ylabel('Y1', color = 'g')
ax2.set_ylabel('Y2', color = 'b')

#y = [1,2,3,4]
#print(y,)
plt.show()