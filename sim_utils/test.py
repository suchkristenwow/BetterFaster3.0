def last_index_of_change(lst):
    for i in range(len(lst) - 1, 0, -1):
        if lst[i] != lst[i - 1]:
            return i
    return -1  # If there is no change

# Example usage:
my_list = [0, 0, 1]
last_change_index = last_index_of_change(my_list)
print("Last index of change:", last_change_index)

