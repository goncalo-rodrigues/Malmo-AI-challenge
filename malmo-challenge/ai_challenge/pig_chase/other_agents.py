from a_star import a_star
from common import ENV_ACTIONS
from malmopy.agent import BaseAgent


class DefectiveAgent(BaseAgent):
    DOOR1_POS = (4, 1)
    DOOR2_POS = (4, 7)

    def __init__(self, name, visualizer=None):
        super(DefectiveAgent, self).__init__(name, len(ENV_ACTIONS), visualizer=visualizer)

    def act(self, state2, reward, done, is_training=False):
        if done:
            return 0
        try:
            entities = state2[1]
            state = state2[0]
            me = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k][0]
            me_details = [e for e in entities if e['name'] == self.name][0]
            yaw = int(me_details['yaw'])
            direction = ((((yaw - 45) % 360) // 90) - 1) % 4
            me = (direction, me)

            me_acc_door1, me_cost_now_door1 = self.compute_distance_to_door1(me, state)
            me_acc_door2, me_cost_now_door2 = self.compute_distance_to_door2(me, state)

            if me_cost_now_door1 < me_cost_now_door2:
                return ENV_ACTIONS.index(me_acc_door1.popleft())
            else:
                return ENV_ACTIONS.index(me_acc_door2.popleft())
        except Exception as e:
            print('error', e)
            return 0

    def compute_distance_to_door1(self, curr_pos, state):
        if curr_pos is None:
            return [], 0
        acc, door1_cost = a_star(curr_pos, self.DOOR1_POS, state)
        return acc, door1_cost

    def compute_distance_to_door2(self, curr_pos, state):
        if curr_pos is None:
            return [], 0
        acc, door2_cost = a_star(curr_pos, self.DOOR2_POS, state)
        return acc, door2_cost


class TitForTatAgent(BaseAgent):
    DOOR1_POS = (4, 1)
    DOOR2_POS = (4, 7)

    def __init__(self, name, other, target, visualizer=None):
        super(TitForTatAgent, self).__init__(name, len(ENV_ACTIONS), visualizer=visualizer)
        self._cooperated = True
        self._other_agent = other
        self._target = target
        self._distance_doors = 0
        self._distance_pig = 0

    def act(self, state2, reward, done, is_training=False):
        if done:
            if reward <= 0:
                self._cooperated = False
            elif reward <= 5:
                if self._distance_pig < self._distance_doors:
                    self._cooperated = True
                else:
                    self._cooperated = False
            else:
                self._cooperated = True
            return 0
        try:
            entities = state2[1]
            state = state2[0]
            me = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k][0]
            me_details = [e for e in entities if e['name'] == self.name][0]

            other = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self._other_agent in k][0]
            other_details = [e for e in entities if e['name'] == self._other_agent][0]

            yaw = int(me_details['yaw'])
            direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

            other_yaw = int(other_details['yaw'])
            other_direction = (
                                  (((
                                    other_yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

            other = (other_direction, other)
            me = (direction, me)

            target = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self._target in k][0]

            me_acc_pig, me_cost_now_pig = self.compute_distance_to_pig(me, target, state)
            _, other_cost_now_pig = self.compute_distance_to_pig(other, target, state)
            _, other_cost_now_door1 = self.compute_distance_to_door1(other, state)
            _, other_cost_now_door2 = self.compute_distance_to_door2(other, state)
            self._distance_pig = other_cost_now_pig
            self._distance_doors = min(other_cost_now_door1, other_cost_now_door2)
            neighbrs = self.neighbors(target, state)
            if len(me_acc_pig) == 0:
                acc, cost = self.compute_distance_to_pig(me, neighbrs[0], state)
                acc2, cost2 = self.compute_distance_to_pig(me, neighbrs[1], state)
                if cost < cost2:
                    return ENV_ACTIONS.index(acc.popleft())
                else:
                    return ENV_ACTIONS.index(acc2.popleft())
            if len(me_acc_pig) > 1:
                return ENV_ACTIONS.index(me_acc_pig.popleft())
            else:
                if other[1] == me[1]:
                    import random
                    return random.randint(0,1)
                else:
                    return 1

        except Exception as e:
            print('error', e)
            return 1

    def compute_distance_to_door1(self, curr_pos, state):
        if curr_pos is None:
            return [], 0
        acc, door1_cost = a_star(curr_pos, self.DOOR1_POS, state)
        return acc, door1_cost

    def compute_distance_to_door2(self, curr_pos, state):
        if curr_pos is None:
            return [], 0
        acc, door2_cost = a_star(curr_pos, self.DOOR2_POS, state)
        return acc, door2_cost

    def compute_distance_to_pig(self, curr_pos, pig_pos, state, output_states = False):
        if curr_pos is None or pig_pos is None:
            return [], 0
        acc, pig_cost = a_star(curr_pos, pig_pos, state, output_states)
        return acc, pig_cost

    def neighbors(self, pos, state):
        # up, right, down, left
        dirs = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        result = []
        for d in dirs:
            result.append((pos[0]+d[0], pos[1]+d[1]))

        result = [n for n in result if
                  n[0] >= 0 and n[0] < state.shape[0] and n[1] >= 0 and n[1] < state.shape[1] and state[
                      n[0], n[1]] != 'sand']

        return result