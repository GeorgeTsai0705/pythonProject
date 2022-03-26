L = [1, 0, 0, 0, 0]

Left_I = 0
Right_I = len(L)-1
Index = (Left_I + Right_I)//2
Left_C = sum(L[:Index])
Right_C = sum(L[Index+1:])


while Left_C != Right_C:
    if Left_I == Index or Right_I == Index:
        print("NO")
        break
    if Left_C > Right_C:
        Right_I = Index
        Index = (Index + Left_I) // 2
    else:
        Left_I = Index
        Index = (Index + Right_I) // 2
    Left_C = sum(L[:Index])
    Right_C = sum(L[Index + 1:])
    print(Left_C, Right_C, Left_I, Right_I, Index)
print("YES")
