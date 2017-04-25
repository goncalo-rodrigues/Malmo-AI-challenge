from collections import deque
from heapq import heapify, heappush, heappop
import operator
import numpy as np
#start (0,(x,y))
#directions: 0:up  1:right  2:down  3:left

state = np.array([[u'grass', u'grass', u'grass', u'grass', u'grass', u'grass',
        u'grass', u'grass', u'grass'],
       [u'grass', u'sand', u'sand', u'sand', u'sand', u'sand', u'sand',
        u'sand', u'grass'],
       [u'grass', u'sand', u'grass', u'grass', u'grass', u'grass',
        u'grass', u'sand', u'grass'],
       [u'sand', u'sand', u'grass', u'sand', u'grass', u'sand',
        u'grass/Agent_2', u'sand', u'sand'],
       [u'sand', u'lapis_block', u'grass', u'grass', u'grass',
        u'grass/Agent_1', u'grass/Pig', u'lapis_block', u'sand'],
       [u'sand', u'sand', u'grass', u'sand', u'grass', u'sand', u'grass',
        u'sand', u'sand'],
       [u'grass', u'sand', u'grass', u'grass', u'grass', u'grass',
        u'grass', u'sand', u'grass'],
       [u'grass', u'sand', u'sand', u'sand', u'sand', u'sand', u'sand',
        u'sand', u'grass'],
       [u'grass', u'grass', u'grass', u'grass', u'grass', u'grass',
        u'grass', u'grass', u'grass']], dtype=object)


def a_star(start, goal, map):
    came_from, cost = {}, {}
    closed_nodes = []
    opened_nodes = []

    heapify(opened_nodes)
    #state, cost
    heappush(opened_nodes, (0, start))
    came_from[start]=None
    cost[start]=0
    current = None

    while len(opened_nodes)>0:
        _, current = heappop(opened_nodes)
        if current[1] == goal:
           break
        for nb in neighbors(current, state):
            ccost = 1
            new_cost = cost[current] + ccost
            if nb not in cost or new_cost < cost[nb]:
                cost[nb] = new_cost
                priority = new_cost + heuristic(goal, nb[1])
                heappush(opened_nodes, (priority, nb))
                came_from[nb] = current

    # build path:
    path = deque()
    #final cost
    c = cost[current]
    while current is not start:
        path.appendleft(current[2])
        current = came_from[current]
    return path, c


def neighbors(pos, state):
    #up, right, down, left
    dir = [(-1,0), (0,1), (1, 0), (0,-1)]
    result = []
    result.append(((pos[0]+1)%4,pos[1], "turn 1"))
    result.append(((pos[0]-1)%4, pos[1], "turn -1"))
    result.append((pos[0], tuple(map(operator.add, pos[1], dir[pos[0]])), "move 1"))
    print(result)
    result = [n for n in result if
              n[1][0] >= 0 and n[1][0] < state.shape[0] and n[1][1] >= 0 and n[1][1] < state.shape[1] and state[n[1][0], n[1][1]] != 'sand']


    return result

def heuristic(a, b):
    (x1, y1) = (a[0], a[1])
    (x2, y2) = (b[0], b[1])
    return abs(x1 - x2) + abs(y1 - y2)

