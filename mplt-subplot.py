import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# plt.figure()
# gs = gridspec.GridSpec(3,3)
# ax1= plt.subplot(gs[0,:])
# ax2= plt.subplot(gs[1,:2])
# ax3= plt.subplot(gs[1:,2])
# ax4= plt.subplot(gs[-1,0])
# ax5= plt.subplot(gs[-1,-2])

# plt.tight_layout()
# plt.show()

f, ((ax11, ax12),(ax21,ax22)) = plt.subplots(2,2,sharex=True, sharey=True)
ax11.scatter([1,2],[1,2])
plt.tight_layout()
plt.show()