    45import numpy as np

Exist = []
Predecessor = []
Distance = []
Queue = []

original_point = (0, 0)
target_point = (5, 5)
Size = 5
distance = 0

#initial
Exist.append(original_point)
Queue.append(original_point)
Predecessor.append(-1)
Distance.append(0)

while len(Queue) != 0:

    Old_P = Queue[0]
    Old_P_I = Exist.index(Old_P)
    Old_P_D = Distance[Old_P_I]

    # Direction 1
    New_P = (Queue[0][0]+1, Queue[0][1])
    if New_P[0] < Size and New_P not in Exist:
        Exist.append(New_P)
        Queue.append(New_P)
        Predecessor.append(Old_P_I)
        Distance.append(Old_P_D + 1)

    # Direction 2
    New_P = (Queue[0][0], Queue[0][1]+2)
    if New_P[1] < Size and New_P not in Exist:
        Exist.append(New_P)
        Queue.append(New_P)
        Predecessor.append(Old_P_I)
        Distance.append(Old_P_D + 1)
    # delet Queue
    Queue.pop(0)

print("Exist", Exist)
print("Distance", Distance)
print(Distance[Exist.index((4, 4))])