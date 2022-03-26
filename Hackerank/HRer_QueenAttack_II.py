n = 5
x = 3
y = 4
obstacles = [[5, 5], [4, 2], [2, 3]]


def Boundary_Set(n, x, y):
    Boundary_Position = []
    # Set direction 1
    if x - 1 <= n - y:
        Boundary_Position.append((y + (x - 1) + 1, 0))
    else:
        Boundary_Position.append((n + 1, x - (n - y) - 1))

    # Set direction 2
    Boundary_Position.append((n + 1, x))

    # Set direciton 3
    if n-x <= n-y:
        Boundary_Position.append((y+(n-x) + 1, n + 1))
    else:
        Boundary_Position.append((n+1 , x+(n-y)+1))

    # Set direction 4
    Boundary_Position.append((y, n+1))

    #Set direciton 5
    if n-x <= y-1:
        Boundary_Position.append((y-(n-x)-1, n+1))
    else:
        Boundary_Position.append((0, x+(y-1)+1))

    #Set direction 6
    Boundary_Position.append((0, x))

    #Set direciton 7
    if x < y:
        Boundary_Position.append(((y-x), 0))
    else:
        Boundary_Position.append((0, (x-y)))

    #Set direciton 8
    Boundary_Position.append((y, 0))
    return Boundary_Position

def Clean_Obstacles(Boundary, x, y, obstacles):
    for ele in obstacles:
        if ele[1] == x:
            if ele[0] > y and Boundary[1][0] > ele[0]:
                Boundary[1] = (ele[0], x)
            elif ele[0] < y and Boundary[5][0] < ele[0]:
                print(ele)
                Boundary[5] = (ele[0], x)
        elif ele[0] == y:
            if ele[1] > x and Boundary[3][1] > ele[1]:
                Boundary[3] = (y, ele[1])
            elif ele[1] < x and Boundary[7][1] < ele[1]:
                Boundary[7] = (y, ele[1])
        else:
            Slope = ( ele[1] - x ) / ( ele[0] - y)

            if abs(Slope) != 1:
                continue
            else:
                if Slope ==1:
                    if ele[1] > x and Boundary[2][1] > ele[1]:
                        Boundary[2] = (ele[0], ele[1])
                    elif ele[1] < x and Boundary[6][1] < ele[1]:
                        Boundary[6] = (ele[0], ele[1])
                elif Slope == -1:
                    if ele[1] > x and Boundary[4][1] > ele[1]:
                        Boundary[4] = (ele[0], ele[1])
                    elif ele[1] < x and Boundary[0][1] < ele[1]:
                        Boundary[0] = (ele[0], ele[1])
    return Boundary

def Cal_C(x, y, obstacles):
    Out = 0
    if abs(obstacles[0][1] - x) > 1:
        Out += (abs(obstacles[0][1] - x) - 1)
    if abs(obstacles[1][0] - y) > 1:
        Out += (abs(obstacles[1][0] - y) - 1)
    if abs(obstacles[2][1] - x) > 1:
        Out += (abs(obstacles[2][1] - x) - 1)
    if abs(obstacles[3][1] - x) > 1:
        Out += (abs(obstacles[3][1] - x) - 1)
    if abs(obstacles[4][1] - x) > 1:
        Out += (abs(obstacles[4][1] - x) - 1)
    if abs(obstacles[5][0] - y) > 1:
        Out += (abs(obstacles[5][0] - y) - 1)
    if abs(obstacles[6][1] - x) > 1:
        Out += (abs(obstacles[6][1] - x) - 1)
    if abs(obstacles[7][1] - x) > 1:
        Out += (abs(obstacles[7][1] - x) - 1)
    return Out

B = Boundary_Set(n,x,y)
print(B)
C = Clean_Obstacles(B,x,y,obstacles)
print(C)
print(Cal_C(x, y, C))