import sys
sys.path.append("./lib")
import numpy as np
import string
from graph import GridGraphWithItems
from scipy.stats import multivariate_normal
import math
import matplotlib.pyplot as plt
from utility import SortedList
from bisect import bisect_left, bisect_right


nS = 100
for i  in range(0,nS,10):
    print(i)

mean_x = 4
mean_y = 5
((x,y),)= np.random.multivariate_normal([mean_x,mean_y],[[2,0],[0,2]],1)
print(x)
print(y)


for i in range(0,10):
    print(string.ascii_lowercase[i])


print(" Start Test:")
print(" ")

k = 2

items_prob_matrix = np.ones([2,16])/100
items_prob_matrix[0][12] = 0.55
items_prob_matrix[0][9] = 0.08
items_prob_matrix[0][5] = 0.23
items_prob_matrix[0][6] = 0.02
items_prob_matrix[1][10] = 0.40
items_prob_matrix[1][11] = 0.13
items_prob_matrix[1][14] = 0.25
items_prob_matrix[1][9] = 0.10
'''


items_prob_matrix = np.zeros([2,16])
items_prob_matrix[0][2] = 0.04
items_prob_matrix[0][1] = 0.03
items_prob_matrix[0][13] = 0.05
items_prob_matrix[0][12] = 0.55
items_prob_matrix[0][9] = 0.08
items_prob_matrix[0][5] = 0.23
items_prob_matrix[0][6] = 0.02
items_prob_matrix[1][10] = 0.46
items_prob_matrix[1][11] = 0.13
items_prob_matrix[1][14] = 0.31
items_prob_matrix[1][9] = 0.10
'''
'''
items_prob_matrix = np.zeros([2,16])
items_prob_matrix[0][0] = 0.1
items_prob_matrix[0][1] = 0.2
items_prob_matrix[0][2] = 0.3
items_prob_matrix[0][3] = 0.4

items_prob_matrix[1][0] = 0.4
items_prob_matrix[1][1] = 0.4
items_prob_matrix[1][2] = 0.05
items_prob_matrix[1][3] = 0.15
'''
g = GridGraphWithItems(4,4,k,items_prob_matrix)
path = [[3,1]]
nn, steps = g.get_nearest_neighbor_aggregate_prob(3)
print("nn:{}".format(nn))
path.append((nn, steps))
while nn is not None:
    nn,steps = g.get_nearest_neighbor_aggregate_prob(nn)
    path.append((nn,steps))
    print("nn:{}, steps:{}".format(nn,steps))

print("path:"+ str(path))
path_exp_cost = g.calc_path_cost(path)
print(path_exp_cost)

'''
print("do grid..")
x, y = np.mgrid[-5:5:.1, -5:5:.1]
print("init pos...")
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
print(pos)
print("generate random variable...")
rv = multivariate_normal([3, 4], [[7, 0], [0, 4]])
print("plotting...")
print(rv.pdf(pos))
plt.contourf(x, y, rv.pdf(pos))
plt.show()
'''
from scipy.stats import norm
print(norm.cdf([-1,0,1]))
x = np.arange(-1,1,0.1)
#print(x)
print(norm.pdf(x))

var = multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
print(var.cdf([0,math.inf]))

