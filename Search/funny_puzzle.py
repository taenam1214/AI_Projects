import heapq
import numpy as np
import copy

def get_manhattan_distance(from_state, to_state=[1, 2, 3, 4, 5, 6, 7, 0, 0]):
    """
    TODO: implement this function. This function will not be tested directly by the grader. 

    INPUT: 
        Two states (if second state is omitted then it is assumed that it is the goal state)

    RETURNS:
        A scalar that is the sum of Manhattan distances for all tiles.
    """
    
    distance = 0
    reshaped_current_state = np.reshape(from_state, (3, 3))
    reshaped_goal_state = np.reshape(to_state, (3, 3))
    for row in range(3):
        for col in range(3):
            if reshaped_current_state[row][col] == 0:
                continue

            goal_row = int(np.where(reshaped_goal_state == reshaped_current_state[row][col])[0])
            goal_col = int(np.where(reshaped_goal_state == reshaped_current_state[row][col])[1])

            distance += abs(row - goal_row) + abs(col - goal_col)

    return distance

def print_succ(state):
    """
    TODO: This is based on get_succ function below, so should implement that function.

    INPUT: 
        A state (list of length 9)

    WHAT IT DOES:
        Prints the list of all the valid successors in the puzzle. 
    """
    succ_states = get_succ(state)

    for succ_state in succ_states:
        print(succ_state, "h={}".format(get_manhattan_distance(succ_state)))


def get_succ(state):
    """
    TODO: implement this function.

    INPUT: 
        A state (list of length 9)

    RETURNS:
        A list of all the valid successors in the puzzle (don't forget to sort the result as done below). 
    """
    successors = []
    valid_right_indices = [1, 2, 4, 5, 7, 8]
    valid_left_indices = [0, 1, 3, 4, 6, 7]
    valid_up_indices = [0, 1, 2, 3, 4, 5]
    valid_down_indices = [3, 4, 5, 6, 7, 8]

    for index in range(len(state)):
        right = index + 1
        left = index - 1
        up = index - 3
        down = index + 3

        if right in valid_right_indices and state[right] == 0:
            new_state = copy.deepcopy(state)
            new_state[right], new_state[index] = new_state[index], new_state[right]
            if new_state not in successors and new_state != state:
                successors.append(new_state)
        if left in valid_left_indices and state[left] == 0:
            new_state = copy.deepcopy(state)
            new_state[left], new_state[index] = new_state[index], new_state[left]
            if new_state not in successors and new_state != state:
                successors.append(new_state)
        if up in valid_up_indices and state[up] == 0:
            new_state = copy.deepcopy(state)
            new_state[up], new_state[index] = new_state[index], new_state[up]
            if new_state not in successors and new_state != state:
                successors.append(new_state)
        if down in valid_down_indices and state[down] == 0:
            new_state = copy.deepcopy(state)
            new_state[down], new_state[index] = new_state[index], new_state[down]
            if new_state not in successors and new_state != state:
                successors.append(new_state)

    return sorted(successors)

def solve(state):
    pq = []
    heapq.heappush(pq, (get_manhattan_distance(state), state, 0))
    visited = {tuple(state): 0}  # Stores state and cost to get there (g)
    parent_map = {tuple(state): None}

    while pq:
        current_cost, current_state, g = heapq.heappop(pq)
        if current_state == [1, 2, 3, 4, 5, 6, 7, 0, 0]:
            path = []
            state = tuple(current_state)
            while state is not None:
                path.append((list(state), visited[state]))
                state = parent_map[state]
            path.reverse()
            for st, cost in path:
                print(st, f"h={get_manhattan_distance(st)} moves: {cost}")
            return
        successors = get_succ(current_state)
        for succ in successors:
            succ_tuple = tuple(succ)
            g_succ = g + 1
            new_cost = g_succ + get_manhattan_distance(succ)
            if succ_tuple not in visited or g_succ < visited[succ_tuple]:
                visited[succ_tuple] = g_succ
                heapq.heappush(pq, (new_cost, succ, g_succ))
                parent_map[succ_tuple] = tuple(current_state)

if __name__ == "__main__":
    """
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    """
    # print_succ([2,5,1,4,0,6,7,0,3])
    # print()
    #
    # print(get_manhattan_distance([2,5,1,4,0,6,7,0,3], [1, 2, 3, 4, 5, 6, 7, 0, 0]))
    # print()

    solve([2,5,1,4,0,6,7,0,3])
    print()