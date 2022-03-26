class Point:
    def __init__(self, index):
        self.discover = 0
        self.finish = 1
        self.color = "White"
        self.direction = []
        self.predecessor = None
        self.index = index

map = [Point(i) for i in range(8)]

#map setting
map[0].direction = [1, 2]
map[1].direction = [3]
map[2].direction = [1, 5]
map[3].direction = [4, 5]
map[4].direction = []
map[5].direction = [1]
map[6].direction = [4, 7]
map[7].direction = [6]

map[0].discover = 1
map[0].color = 'Gray'

queue = [i for i in range(len(map))]
time = 1
Level = 0
def Search(A, t, L):
    L += 1
    if A.color == 'Black':
        return 0
    if A.direction == []:
        A.color = 'Black'
        A.discover = t
        t += 1
        A.finish = t
        A = A.predecessor

    for i in A.direction:
        if map[i].color == 'White':
            t += 1
            map[i].color = 'Gray'
            map[i].discover = t
            map[i].predecessor = A
            print("index: ", map[i].index, 'Time: ', t)
            Search(map[i], t, L)
    if A.color == 'Black':
        return 0
    t += 1
    A.finish = t
    A.color = 'Black'
    if A.predecessor == None:
        return 0
    else:
        A = A.predecessor
        print('Index2: ', A.index,'Time: ', t)
        Search(A, t, L)

Last = map[0]

while len(queue) != 0:
    if map[queue[0]].color == 'Black':
        queue.pop(0)
    else:
        print('Index?: ', Last.index, 'queue: ', queue[0], 'Time: ', Last.finish)
        map[queue[0]].color = 'Gray'
        map[queue[0]].discover = Last.finish+1
        Search(map[queue[0]], Last.finish, Level)
        Last = map[queue[0]]
print(map[6].discover)


