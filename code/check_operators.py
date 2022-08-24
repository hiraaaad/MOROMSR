import numpy as np

def inverse(lst,a,b):
    for i in range(0,int(np.ceil((b-a)/2))):
        temp = lst[a+i]
        lst[a+i] = lst[b-i]
        lst[b-i] = temp

def insert(lst,a,b): # move the second omponent ahead of the first componet
    temp = lst[b]
    for i in range(0,b-a-1):
        lst[b-i] = lst[b-(i+1)]
    lst[a+1] = temp

def swap(lst,a,b):
    temp = lst[a]
    lst[a] = lst[b]
    lst[b] = temp

lst = np.arange(10)
insert(lst,1,4)
print('x')


