import itertools
c = 0

def var_naming():
    global c
    res = c
    c+=1
    return f'V:{res}'