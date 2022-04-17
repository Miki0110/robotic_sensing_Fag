from collections import deque
import numpy as np

k_size = 3

a = sorted(np.random.randint(100, size=(6)))
b = sorted(np.random.randint(100, size=(6)))
c = sorted(np.random.randint(100, size=(6)))
d = sorted(np.random.randint(100, size=(6)))

print(f'a = {a}, b = {b}, c = {c}, d = {d}')
a1,a2,a3,a4 = 0,0,0,0

for i in range(k_size):
    if a[0] < b[0] and a[0] < c[0] and a[0] < d[0]:
        a.remove(a[0])
        a1 += 1
    elif b[0] < a[0] and b[0] < c[0] and b[0] < d[0]:
        b.remove(b[0])
        a2 += 1
    elif c[0] < a[0] and c[0] < b[0] and c[0] < d[0]:
        c.remove(c[0])
        a3 += 1
    else:
        d.remove(d[0])
        a4 += 1
if a1 > a2 and a1 > a3 and a1 > a4:
    print('a')
elif a2 > a1 and a2 > a3 and a2 > a4:
    print('b')
elif a3 > a1 and a3 > a2 and a3 > a4:
    print('c')
else:
    print('d')

a = sorted(np.random.randint(100, size=(6)))
b = sorted(np.random.randint(100, size=(6)))
c = sorted(np.random.randint(100, size=(6)))
d = sorted(np.random.randint(100, size=(6)))

print(f'a = {a}, b = {b}, c = {c}, d = {d}')
temp_a = deque()
temp_b = deque()
temp_c = deque()
temp_d = deque()
for i in range(len(a)):
    temp_a.append(a[i])
    temp_b.append(b[i])
    temp_c.append(c[i])
    temp_d.append(d[i])


for i in range(k_size):
    if temp_a[0] < temp_b[0] and temp_a[0] < temp_c[0] and temp_a[0] < temp_d[0]:
        temp_a.popleft()
        a = 0+i
    elif temp_b[0] < temp_a[0] and temp_b[0] < temp_c[0] and temp_b[0] < temp_d[0]:
        temp_b.popleft()
        b = 0+i
    elif temp_c[0] < temp_a[0] and temp_c[0] < temp_b[0] and temp_c[0] < temp_d[0]:
        temp_c.popleft()
        c = 0+i
    else:
        temp_d.popleft()
        d = 0+i

print(a,b,c,d)

