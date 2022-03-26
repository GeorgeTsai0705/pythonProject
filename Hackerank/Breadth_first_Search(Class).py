
size = 5
queue = []

class Point:
    def __init__(self,x ,y):
        self.distance = 0
        self.predecessor = None
        self.exsit = False
        self.x = x
        self.y = y

    def next_step(self, Old_P, queue):
        if not self.exsit:
            self.exsit = True
            self.distance = Old_P.distance + 1
            self.predecessor = Old_P
            queue.append(self)


map = [[Point(i,j) for j in range(size)] for i in range(size)]

#initial
map[0][0].exsit = True
queue.append(map[0][0])

while len(queue) != 0:
    Old_P = queue[0]

    # direction_1
    if Old_P.x + 1 < size:
        New_P = map[Old_P.x + 1][Old_P.y]
        New_P.next_step(Old_P, queue)

    # direction_2
    if Old_P.y + 2 < size:
        New_P = map[Old_P.x][Old_P.y + 2]
        New_P.next_step(Old_P, queue)

    queue.pop(0)

print(map[4][4].distance)