import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib import cm
from MPC_control_pdf import mpc_control
from stuff import read_prob_map, get_pdf
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
from sklearn import preprocessing

# choose between two different maps
map_nr = 2
if map_nr==1:
  n_components = 32
  map_width, map_height = 600, 600
  box_size_w,box_size_h = 25, 25
  path = "p_map_2_11_5000_600x600_1.txt"
elif map_nr == 2:
  n_components = 29
  map_width, map_height = 800, 800
  box_size_w,box_size_h = 25, 25
  path = "p_map_2_11_5000_800x800_1.txt"
count = 0

# initial values for mpc
steps = 300  # steps to be calculated
dt = 1/5  # time per steps
T = 15  # horizon

# inital x and y values
init_x = 9
init_y = 9

boxes_number_width = map_width/box_size_w
boxes_number_height = map_height/box_size_h
arr = read_prob_map(path, boxes_number_width, boxes_number_height )

# calculate Gaussian Mixture
gm = GaussianMixture(n_components=n_components, random_state=0).fit(arr)
labels = gm.predict(arr)
means = gm.means_
covariances = gm.covariances_

_x, _y = np.mgrid[0:(boxes_number_width-1), 0:(boxes_number_height-1)]
xy = np.column_stack([_x.flat, _y.flat])
zg = get_pdf(n_components, xy, means, covariances)

# reshape discrete pdf
zg = zg.reshape(_x.shape)
#zg = preprocessing.normalize(zg)

# interpolate discrete pdf
interp_func = LinearNDInterpolator(list(zip(xy[:,0],xy[:,1])), zg.flatten())

# init empty arrays
t = np.linspace(0, steps*dt, steps)
ol_t = np.linspace(0, T*dt, T)

# Closed loop
x = np.zeros(steps)
y = np.zeros(steps)
z = np.zeros(steps)

# Open loop
ol_x = np.zeros(T)
ol_y = np.zeros(T)
ol_z = np.zeros(T)

u_x = np.zeros(steps)
u_y = np.zeros(steps)

new_x = init_x
new_y = init_y
new_ol_x = init_x
new_ol_y = init_y
x[0] = init_x
y[0] = init_y
ol_x[0] = init_x
ol_y[0] = init_y
z[0] = interp_func(init_x,init_y)

# calculate the optimal input
inputs = mpc_control(init_x,init_y,interp_func,0,0,T,5*np.identity(2),5*np.identity(2),dt)
# first open loop inputs
first_ol_input = inputs

# calculate x and y for open loop input
for i in range(T-1):
    new_ol_x += first_ol_input[0][i]*dt**2
    new_ol_y += first_ol_input[1][i]*dt**2
    ol_x[i+1] = new_ol_x
    ol_y[i+1] = new_ol_y
    ol_z[i+1] = interp_func(new_ol_x,new_ol_y)

# closed loop for defined steps
for i in tqdm(range(steps-1), desc ="Progress: "):
    new_x += inputs[0][0]*dt**2
    new_y += inputs[1][0]*dt**2
    x[i+1] = new_x
    y[i+1] = new_y
    z[i+1] = interp_func(new_x,new_y)
    u_x[i+1] = inputs[0][0]
    u_y[i+1] = inputs[1][0]
    inputs = mpc_control(new_x,new_y,interp_func,inputs[0][0],inputs[1][0],T,np.identity(2),np.identity(2),dt)

# plot solution
xnew = np.linspace(0, boxes_number_width-1, 800)
ynew = np.linspace(0, boxes_number_height-1, 800)
Xnew, Ynew = np.meshgrid(xnew, ynew)

fig = plt.figure()
ax3 = fig.add_subplot(1, 2, 1)
ax3.scatter(init_x,init_y, color="green", label="Start")
ax3.plot(ol_x, ol_y, linewidth=3, color='r',alpha=1)
ax3.set_title('Open Loop')

ax5 = fig.add_subplot(1, 2, 2)
ax5.plot(ol_t, interp_func(ol_x,ol_y), linewidth=3, color='r',alpha=1)
ax5.set_title('Z-Axis Open Loop')

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(t, u_x, label="Acceleration x")
ax1.plot(t, u_y, label='Acceleration y')
ax1.set_xlabel("Time")
ax1.set_ylabel("Acceleration")
ax1.legend()
ax1.set_title('Inputs')

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(x, y)
ax2.scatter(init_x,init_y, color="green", label="Start")
ax2.scatter(means[:,0], means[:,1],means[:,2]*100, color='r', alpha=1)
ax2.grid(True)
ax2.set_title('Drone trajectory')

ax5 = fig.add_subplot(1, 3, 3)
ax5.plot(t, interp_func(x,y), linewidth=3, color='r',alpha=1)
ax5.set_title('Z-Axis Closed Loop')

fig = plt.figure()
ax4 = fig.add_subplot(1, 1, 1, projection = '3d')
ax4.plot_surface(Xnew,Ynew,interp_func(Xnew, Ynew), cmap=cm.jet, alpha=0.6)
ax4.scatter(init_x,init_y, interp_func(init_x,init_y), color="green", label="Start")
ax4.plot(x, y, z, linewidth=3, color='r',alpha=1)
ax4.set_title('Closed Loop')

plt.show()
