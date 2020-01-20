import matplotlib.pyplot as plt

fig = plt.figure()
x = [1,2,3,4,5,6,7]
y = [1,3,4,2,5,8,6]

left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax1 = fig.add_axes([left, bottom, width, height])

ax1.plot(x,y,'r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('title')

#=======================================
# small left up (way 1)
left, bottom, width, height = 0.2, 0.6, 0.25, 0.25
ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot(x,y,'r')  # draw red line
ax2.set_xlabel('x') # set x labels
ax2.set_ylabel('y') # set y labels
ax2.set_title('inside1')


#=======================================
# small right bottom (way 2)
left, bottom, width, height = 0.6, 0.2, 0.25, 0.25

plt.axes([left, bottom, width, height])

# no ax.
plt.plot(x,y,'g')  # draw green line
plt.xlabel('x') # set x labels
plt.ylabel('y') # set y labels
plt.title('inside2')

plt.show()