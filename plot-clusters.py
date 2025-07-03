import matplotlib.pyplot as plt

# Example 2D point lists

# cluster1 = [(17, -57), (30, -59), (31, -57), (31, -59), (62, -49), (30, -58), (31, -58), (14, -58), (18, -60), (19, -58), (33, -58), (10, -58), (25, -51), (40, -51), (39, -52), (23, -52), (62, -32), (60, -40), (54, -52), (59, -48), (60, -48)]
# cluster2 = [(-30, 27), (-12, 27), (-22, 28), (-13, 24), (-7, 26), (-11, 31), (-26, 16), (-25, 23), (-1, 25), (-6, 24), (-24, 28), (-5, 29), (-28, 6), (-29, 19), (-14, 27), (-27, 19), (-12, 32), (-21, 28), (-30, 13), (-13, 28), (-25, 31), (-2, 24), (-27, 23), (-26, 6), (-19, 33), (-8, 33), (-28, 7), (-30, 14), (-30, 24), (-30, 25), (-20, 9), (-21, 10), (-21, 23)]
# cluster4 = [(61, 10), (63, -5), (61, 21), (61, -7), (63, 5), (63, -1), (61, 22), (61, -4), (63, 10), (61, -9), (63, 33), (63, -3), (45, 33), (48, 24), (48, 31), (62, -20), (58, 28), (62, 17)]
# cluster3 = [(4, -22), (4, -23), (0, -11), (-24, -4), (-28, -32), (-24, -5), (-30, 2), (-24, -2), (-25, -30), (-28, 2), (-26, -7), (-25, -10), (-26, -2), (-26, -9), (-24, -32), (-29, -32), (-28, -13), (-23, -32), (-20, -13), (-14, -20), (-22, -17)]

import os
# read from cluster1.txt
cluster0 = []
cur_folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(cur_folder, 'cluster-0.txt'), 'r') as file:
    for line in file:
        id, x, y, sv, svc = map(float, line.strip().split(','))
        cluster0.append((x, y))

cluster1 = []
with open(os.path.join(cur_folder, 'cluster-1.txt'), 'r') as file:
    for line in file:
        id, x, y, sv, svc = map(float, line.strip().split(','))
        cluster1.append((x, y))
cluster2 = []
with open(os.path.join(cur_folder, 'cluster-2.txt'), 'r') as file:
    for line in file:
        id, x, y, sv, svc = map(float, line.strip().split(','))
        cluster2.append((x, y))
cluster3 = []
with open(os.path.join(cur_folder, 'cluster-3.txt'), 'r') as file:
    for line in file:
        id, x, y, sv, svc = map(float, line.strip().split(','))
        cluster3.append((x, y))

# Unpack the points for each cluster
x0, y0 = zip(*cluster0)
x1, y1 = zip(*cluster1)
x2, y2 = zip(*cluster2)
x3, y3 = zip(*cluster3)

# Plot each cluster with a distinct color
plt.scatter(x0, y0, color='red', label='Cluster 0')
plt.scatter(x1, y1, color='green', label='Cluster 1')
plt.scatter(x2, y2, color='blue', label='Cluster 2')
plt.scatter(x3, y3, color='yellow', label='Cluster 3')

# Add legend and labels
plt.title('KMeans Clusters')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.axis('equal')

# Show the plot
plt.show()
