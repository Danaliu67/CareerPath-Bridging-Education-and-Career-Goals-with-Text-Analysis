import copy


def find_max_index(list_, max_num=3):
    t = copy.deepcopy(list_)
    max_number = []
    max_index = []
    for _ in range(max_num):
        number = max(t)
        index = t.index(number)
        t[index] = 0
        max_number.append(number)
        max_index.append(index)
    t = []
    if(0.0 in max_number):
        discard = []
        for i in range(len(max_number)):
            if max_number[i] == 0.0:
                discard.append(max_index[i])
        max_number.remove(0.0)
        for item in discard:
            max_index.remove(item)
    return (max_number, max_index)
 