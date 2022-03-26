grid = []
#grid = ['GGGGGGGG', 'GBGBGGBG', 'GBGBGGBG', 'GGGGGGGG','GBGBGGBG', 'GGGGGGGG','GBGBGGBG', 'GGGGGGGG']
grid = ['GGGGG', 'GGGGG', 'GGGGG', 'GGGGG', 'GGGGG']
#grid = ['GBBBBBBGGGBGGBB', 'GBBBBBBGGGBGGBB', 'GBBBBBBGGGBGGBB', 'GBBBBBBGGGBGGBB', 'GGGGGGGGGGGGGGG', 'GGGGGGGGGGGGGGG', 'GBBBBBBGGGBGGBB', 'GBBBBBBGGGBGGBB', 'GGGGGGGGGGGGGGG', 'GBBBBBBGGGBGGBB', 'GBBBBBBGGGBGGBB', 'GGGGGGGGGGGGGGG', 'GGGGGGGGGGGGGGG', 'GBBBBBBGGGBGGBB']

def B_string(num):
    Out=''
    for i in range(num):
        Out += 'B'
    return Out

def G_string(num):
    Out=''
    for i in range(num):
        Out += 'G'
    return Out

def print_grid(grid):
    for ele in grid:
        print(ele)
    print("############")

print("Original Grid")
print_grid(grid)

def check_fun(grid, n, X, Y):
    if grid[X][Y+1-n:Y+n] != G_string(2*n-1):
        return False
    else:
        for i in range(1,n):
            if not (grid[X-i][Y] == grid[X+i][Y] == "G"):
                return False
    return True

def replace_fun(grid, n, X, Y):
    grid[X] = grid[X][:Y+1-n] + B_string(2*n-1) + grid[X][Y+n:]
    for i in range(1,n):
        grid[X-i] = grid[X-i][:Y] + "B" + grid[X-i][Y+1:]
        grid[X+i] = grid[X+i][:Y] + "B" + grid[X+i][Y+1:]
    return grid
print(check_fun(grid, 3, 2, 2))
print_grid(replace_fun(grid, 3, 2, 2))
