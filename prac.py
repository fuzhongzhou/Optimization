import numpy as np

D0 = 6000
r = 0.15
n = 12
d = 1 + r/n
f = 0.01

# a
D = D0
i = 0
while D >= 0.5*D0:
    i += 1
    P = max(25, (f + r/n)*D)
    D = (1+r/n)*D - P
print(i)
pass

# b
D = D0
i = 0
P = 10000
while P > 25:
    i += 1
    P = max(25, (f + r/n)*D)
    D = (1+r/n)*D - P
print(i)

# c
D = D0
i = 0
while D >= 0:
    i += 1
    P = max(25, (f + r/n)*D)
    D = (1+r/n)*D - P
    print(D)
print(i)

# d
D = D0
P = 135
while D >= 0:
    i += 1
    D = (1+r/n)*D - P
    print(D)
print(i)

pass