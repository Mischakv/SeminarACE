import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from matplotlib import cm
from MPC_control_pdf import mpc_control
from stuff import read_prob_map, get_probMap
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn import metrics
import time
matplotlib.rcParams.update({'font.size': 17})

# choose between two different maps
map_nr = 2
if map_nr==1:
  n_components = 32
  map_width, map_height = 600, 600
  box_size_w,box_size_h = 25, 25
  path = "p_map_2_11_5000_600x600_1.txt"
elif map_nr == 2:
  n_components = 18
  map_width, map_height = 800, 800
  box_size_w,box_size_h = 25, 25
  path = "p_map_15_12_5000_800x800_1.txt"
count = 0

# initial values for mpc
steps = 300  # steps to be calculated
dt = 1/5  # time per steps
T = 45  # horizon

# inital x and y values for mpc
init_x = 8
init_y = 6

boxes_number_width = map_width/box_size_w
boxes_number_height = map_height/box_size_h
arr = read_prob_map(path, boxes_number_width, boxes_number_height )

#START BIC component estimator
def gmm_bic_score(estimator, X):
    """Callable to pass to GridSearchCV that will use the BIC score."""
    # Make it negative since GridSearchCV expects a score to maximize
    #clusters = estimator.fit_predict(X)
    #return metrics.silhouette_score(X,clusters)
    return -estimator.bic(X)


param_grid = {
    "n_components": range(2, 25),
    "covariance_type": ["full"],
    "random_state":[0]
}
grid_search = GridSearchCV(
    GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
)
grid_search.fit(arr)

df = pd.DataFrame(grid_search.cv_results_)[
    ["param_n_components", "param_covariance_type", "mean_test_score"]
]
df["mean_test_score"] = -df["mean_test_score"]
df = df.rename(
    columns={
        "param_n_components": "Number of components",
        "param_covariance_type": "Type of covariance",
        "mean_test_score": "BIC score",
    }
)
df.sort_values(by="BIC score").head()
print(df.nlargest(10, 'BIC score'))
fig, ax = plt.subplots(1, 1)
#ax.table(cellText=df.nlargest(5, 'BIC score').values, colLabels=df.nlargest(5, 'BIC score').keys(), loc='center')
ax.plot(df.values[:,0],df.values[:,2]*-1, label='Score')
ax.axvspan(20, 24, color='red', alpha=.5, label='Not viable components')
ax.set_title('BIC score')
ax.set_ylabel('BIC score')
ax.legend()
ax.set_xlabel('Number of components')
#plt.show()

#END BIC
# calculate Gaussian Mixture
gm = GaussianMixture(n_components=n_components, random_state=0).fit(arr)
means = gm.means_
covariances = gm.covariances_

_x, _y = np.mgrid[0:(boxes_number_width-1):0.1, 0:(boxes_number_height-1):0.1]
xy = np.column_stack([_x.flat, _y.flat])
zg = get_probMap(n_components, xy, means, covariances)

# reshape discrete probability map
zg = zg.reshape(_x.shape)

# interpolate discrete probability map
interp_func = LinearNDInterpolator(list(zip(xy[:,0],xy[:,1])), zg.flatten())

# init empty arrays
t = np.linspace(0, steps*dt, steps)
ol_t = np.linspace(0, T*dt, T)

# Closed loop
x = np.zeros((steps,2))
y = np.zeros((steps,2))
z = np.zeros(steps)

# Open loop
ol_x = np.zeros((T,2))
ol_y = np.zeros((T,2))
ol_z = np.zeros(T)

u_x = np.zeros(steps)
u_y = np.zeros(steps)

new_x = [init_x,0]
new_y = [init_y,0]
new_ol_x = [init_x,0]
new_ol_y = [init_y,0]
x[0] = np.array([init_x,0])
y[0] = np.array([init_y,0])
ol_x[0] = np.array([init_x,0])
ol_y[0] = np.array([init_y,0])
z[0] = interp_func(init_x,init_y)

# input weight
Q = 6500

start_time = time.time()
# calculate the optimal input
inputs = mpc_control([init_x, 0],[init_y, 0],interp_func,0,0,T,Q*np.identity(1),np.identity(1),dt)

# first open loop inputs
first_ol_input = inputs
cost = 0
# calculate x and y for open loop input
for i in range(T-1):
    #new_ol_x += first_ol_input[0][i]*dt**2
    #new_ol_y += first_ol_input[1][i]*dt**2
    new_ol_x += dt*np.array([new_ol_x[1], first_ol_input[0][i]])
    new_ol_y += dt*np.array([new_ol_y[1], first_ol_input[1][i]])
    ol_x[i+1] = new_ol_x
    ol_y[i+1] = new_ol_y
    cost += -interp_func(new_ol_x[0],new_ol_y[0])*Q
    #print(np.dot(first_ol_input[0][i], first_ol_input[0][i]))
    cost += np.dot(first_ol_input[0][i], first_ol_input[0][i])
    cost += np.dot(first_ol_input[1][i], first_ol_input[1][i])
    ol_z[i+1] = cost#interp_func(new_ol_x,new_ol_y)

# closed loop for defined steps
for i in tqdm(range(steps-1), desc ="Progress: "):
    #new_x += inputs[0][0]*dt**2
    #new_y += inputs[1][0]*dt**2
    new_x += dt*np.array([new_x[1], inputs[0][0]])
    new_y += dt*np.array([new_y[1], inputs[1][0]])
    x[i+1] = new_x
    y[i+1] = new_y
    z[i+1] = interp_func(new_x[0],new_y[0])
    u_x[i+1] = inputs[0][0]
    u_y[i+1] = inputs[1][0]
    inputs = mpc_control(new_x,new_y,interp_func,inputs[0][0],inputs[1][0],T,Q*np.identity(1),np.identity(1),dt)
print("--- %s seconds ---" % (time.time() - start_time))
# plot solution
xnew = np.linspace(0, boxes_number_width-1, 800)
ynew = np.linspace(0, boxes_number_height-1, 800)
Xnew, Ynew = np.meshgrid(xnew, ynew)
fig = plt.figure()
ax9 = fig.add_subplot(1, 4, 1)
ax9.plot(ol_t, inputs[0], label="Acceleration x")
ax9.plot(ol_t, inputs[1], label='Acceleration y')
ax9.set_xlabel("Time")
ax9.set_ylabel("Acceleration")
ax9.legend()
ax9.set_title('Inputs')

ax3 = fig.add_subplot(1, 4, 2)
ax3.plot(ol_x[:,0], ol_y[:,0], linewidth=1, color='r',alpha=1, label="Path")
ax3.scatter(init_x,init_y, color="green", label="Start")
ax3.set_xlabel("x-direction")
ax3.set_ylabel("y-direction")
ax3.legend()
ax3.set_title('Path on Map')

ax8 = fig.add_subplot(1, 4, 3)
ax8.plot(ol_t, interp_func(ol_x[:,0],ol_y[:,0]), linewidth=1, color='r',alpha=1)
ax8.set_xlabel("Time")
ax8.set_ylabel("Probability")
ax8.set_title('Probability on Path')

ax5 = fig.add_subplot(1, 4, 4)
ax5.plot(ol_t, ol_z, linewidth=1, color='r',alpha=1)
ax5.set_xlabel("Time")
ax5.set_ylabel("Value")
ax5.set_title('Value of Cost Function')

fig = plt.figure()
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(t, u_x, label="Acceleration x")
ax1.plot(t, u_y, label='Acceleration y')
ax1.set_xlabel("Time")
ax1.set_ylabel("Acceleration")
ax1.legend()
ax1.set_title('Inputs')

ax2 = fig.add_subplot(1, 3, 2)
ax2.plot(x[:,0], y[:,0], label="Path")
ax2.scatter(init_x,init_y, color="green", label="Start")
ax2.scatter(means[:,0], means[:,1],means[:,2]*10000, color='r', alpha=1, label="GMM Means")
ax2.grid(True)
ax2.set_xlabel("X-direction")
ax2.set_ylabel("Y-direction")
ax2.legend()
ax2.set_title('Drone trajectory')

ax5 = fig.add_subplot(1, 3, 3)
ax5.plot(t, interp_func(x[:,0],y[:,0]), linewidth=3, color='r',alpha=1)
ax5.set_xlabel("Time")
ax5.set_ylabel("Probability")
ax5.set_title('Probability on Map - Closed Loop')

""" ax4 = fig.add_subplot(1, 4, 4, projection = '3d')
ax4.plot_surface(Xnew,Ynew,interp_func(Xnew, Ynew), cmap=cm.jet, alpha=0.6)
ax4.scatter(init_x,init_y, interp_func(init_x,init_y), color="green", label="Start")
ax4.plot(x[:,0], y[:,0], z, linewidth=3, color='r',alpha=1)
ax4.set_title('Probability Map')
ax4.view_init(elev=58., azim=-45)
ax4.set_xlabel("X-direction")
ax4.set_ylabel("Y-direction") """

fig = plt.figure()
ax4 = fig.add_subplot(1, 1, 1, projection = '3d')
ax4.plot_surface(Xnew,Ynew,interp_func(Xnew, Ynew), cmap=cm.jet, alpha=0.6)
ax4.scatter(init_x,init_y, interp_func(init_x,init_y), color="green", label="Start")
ax4.plot(x[:,0], y[:,0], z, linewidth=3, color='r',alpha=1)
ax4.set_title('Probability Map')
ax4.view_init(elev=58., azim=-45)
ax4.set_xlabel("X-direction")
ax4.set_ylabel("Y-direction")
#ax4.set_zlabel("Probability")
#plt.savefig('histogram.pgf')
plt.show()
