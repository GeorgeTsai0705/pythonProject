import random
List = [random.randrange(0,10) for i in range(10)]
print(List)
M = max(List)
B = min(List)
for i in List:
    if M > i > B:
        B = i
print(B)