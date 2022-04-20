from collections import deque
import numpy as np

k_size = 3
a1=[]
a2=[]
instruments = ['guitar', 'bass', 'trumpet', 'drumm']
feature_data = ['holes', 'circularity', 'compactness', 'elongation', 'thiness', 'intensity']

a = sorted(np.random.randint(100, size=len(feature_data)))
b = sorted(np.random.randint(100, size=len(feature_data)))
c = sorted(np.random.randint(100, size=len(feature_data)))
d = sorted(np.random.randint(100, size=len(feature_data)))


print(f'a = {a}, b = {b}, c = {c}, d = {d}')
dist = np.concatenate((a,b,c,d))
temp_dist = sorted(dist)


for i in range(k_size):
    a1.append(np.where(dist == temp_dist[i])[0][0])
print(f'intervals = {a1}')
data_interval = []
for i in range(len(instruments)):
    data_interval.append(sum(1 for n in a1 if n > (i)*len(feature_data)-1 and n <= (i+1)*len(feature_data)-1))

print(data_interval)
print(instruments[data_interval.index(max(data_interval))])

