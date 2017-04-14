import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

N = 5
D = 2

#cmap = plt.cm.get_cmap('jet')
cmap = LinearSegmentedColormap.from_list('mycmap', [(0, 'red'), 
                                                    (1, 'blue')]
                                        )
x = 1 - 2 * np.random.rand(N, D)
c_gt = np.random.randint(0, 2, N)


##### Define w
w = np.random.randn(D)

w = w / np.linalg.norm(w)

t = 0

##### Classify
w_t = np.append(w, t)

t_append = -1 * np.ones((N, 1))
x_t = np.append(x, t_append, 1)

c_eval = np.dot(x_t, np.transpose(w_t))
c_out = np.zeros(np.size(c_eval))

c_out[np.where(c_eval > 0)] = 1


print "x: \n%s" % np.array_str(x)
print "w: \n%s" % np.array_str(w)
print "c_eval: \n%s" % np.array_str(c_eval)
print "c_out: \n%s" % np.array_str(c_out)
print "c_gt: \n%s" % np.array_str(c_gt)

##### Visualize
#fig, (axScatter, axClass) = plt.subplots(1, 2, figsize=(13, 6))
plt.figure(figsize=(15,8))
axScatter = plt.subplot(121, autoscale_on=False, aspect='equal', adjustable='box-forced')
axClass = plt.subplot(122, autoscale_on=False, aspect='equal', adjustable='box-forced', sharex=axScatter, sharey=axScatter)

# Samples
axScatter.scatter(x[:, 0], x[:, 1], c=c_gt, cmap=cmap, linewidths=0)
axClass.scatter(x[:, 0], x[:, 1], c=c_out, cmap=cmap, linewidths=0)

axScatter.title.set_text('Ground truth')
axClass.title.set_text('Classification')
#axClass.title('Classification', fontsize=10)

#viewport
v = 1.5
axScatter.axis([-v, v, -v, v])
axClass.axis([-v, v, -v, v])

# Class line (TODO: Add threshold)
axClass.plot([0, w[0]], [0, w[1]])


#plt.axis('equal')
#axScatter.axis('equal')
#axClass.axis('equal')
plt.draw()
plt.show()

#print "result: \n%s" % np.array_str(result)
