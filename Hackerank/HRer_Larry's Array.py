from random import shuffle
List = list(range(1,11))
shuffle(List)

def rotate_List(l,index):
    (a,b,c) = (l[index-1], l[index], l[index+1])
    l[index-1] = b
    l[index] = c
    l[index+1] = a
    return l
def cal_inversion(l):
    Cont = 0
    for i in range(0,len(l)-1):
        for j in range(i+1, len(l)):
            if l[i] > l[j]:
                Cont += 1
    return Cont


print(List)
print(cal_inversion(List))
i = 1
N = len(List)
while i <= N-2:
    I = List.index(i)
    if I + 1 == i:
        i += 1
        continue
    elif I == N - 1:
        rotate_List(List, I-1)
        continue
    rotate_List(List, I)
    print(List)
if List[-1] == N and List[-2] == N-1:
    print("YES")
else:
    print("NO")