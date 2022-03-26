Input = [7, 2, 4, 6, 5, 9, 12, 11 ]
k = 2

f = open('input08.txt','r')
Data = f.readlines()
k = int(Data[0].split(' ')[1].replace('\n',''))
Input = map(int, Data[1].split(' '))

f.close()

House = sorted(Input)
Count = 0
Index = 0
print(House)

def furthest_point(i, House, k):
    if House[i+1] > House[i] + k:
        return 0
    elif House[i+1] == House[i] + k:
        return i+2
    for j in range(1, k+1):
        print(House[i+j+1],House[i] + k ,House[i+j])
        if i+j+1 > len(House)-1:
            return i+j+1
        if House[i+j+1] > House[i] + k >= House[i+j]:
            return i+j+1
        elif House[i+j+1] == House[i] + k:
            return i+j+2

while Index < len(House) - 1:
    Result = furthest_point(Index, House, k)
    if Result == 0:
        Count += 1
        Index += 1
        continue
    else:
        #Find House_X
        X_Index = Result - 1
        print("Find House_X", X_Index)

    if X_Index >= len(House) - 1:
        Index = X_Index
        break

    Result = furthest_point(X_Index, House, k)

    if Result == 0:
        Count += 1
        Index = X_Index + k
        continue
    else:
        #Find House_Y
        Count += 1
        Y_Index = Result - 1
        Index = Result
        print("Find House_Y", Y_Index)
        print(Index)
if Index == len(House) - 1:
    Count +=1
print(Index, Count)

