import sys
import numpy as np
import string
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


arr  = [[0 for i in range(0,5)] for i in range(0,2)]
print(arr)
arr[1][2] = 1
print(arr)

x=int(35/20)
print(x)

already_visited = 16 * [0]
print(already_visited)

class Item():
    def __init__(self):
        self.name = "a"

items = [Item(),Item(),Item()]
''.join(items)