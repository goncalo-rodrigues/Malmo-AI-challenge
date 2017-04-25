
from malmopy.agent import BaseAgent
from common import visualize_training, Entity, ENV_TARGET_NAMES, ENV_ENTITIES, ENV_AGENT_NAMES,\
        ENV_ACTIONS, ENV_CAUGHT_REWARD, ENV_BOARD_SHAPE
from a_star import a_star
class AASMAAgent(BaseAgent):
    ACTIONS = ENV_ACTIONS
    DOOR1_POS = (4, 1)
    DOOR2_POS = (4, 7)
    alpha = 0.4
    initial_belief = 1
    threshold = 0.5

    def __init__(self, name, other_agent, target, visualizer = None):
        super(AASMAAgent, self).__init__(name, len(AASMAAgent.ACTIONS), visualizer=visualizer)
        self._target = str(target)
        self._other_agent = str(other_agent)
        self._previous_target_pos = None
        self._previous_other_pos = None
        self._previous_pos = None
        self._action_list = []
        self._belief = self.initial_belief
        self._desires = []
        self._intentions = []

    def act(self, state2, reward, done, is_training=False):
        if done:
            self._action_list = []
            self._previous_target_pos = None
            self._previous_other_pos = None
            self._previous_pos = None
            self._desires = []
            self._intentions = []
            if reward <= 0:
                print('i was fooled')
                self._belief = min(0.25, self._belief)
            elif reward <= 5:
                print('i fooled him')
            else:
                self._belief = max(0.75, self._belief)
                print('we cooperated')
            print('new episode')
            return 0

        try:
            entities = state2[1]
            state = state2[0]
            # retrieve information from state

            me = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k][0]
            me_details = [e for e in entities if e['name'] == self.name][0]

            other = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self._other_agent in k][0]
            other_details = [e for e in entities if e['name'] == self._other_agent][0]

            yaw = int(me_details['yaw'])
            direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

            other_yaw = int(other_details['yaw'])
            other_direction = (
                              (((other_yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

            other = (other_direction, other)
            me = (direction, me)

            target = [(i, j) for i, v in enumerate(state) for j, k in enumerate(v) if self._target in k][0]

            if self._previous_target_pos is None:
                self._previous_target_pos = target
            if self._previous_other_pos is None:
                self._previous_other_pos = other
            if self._previous_pos is None:
                self._previous_pos = me

            _, me_cost_now_door1 = self.compute_distance_to_door1(me, state)
            _, me_cost_prev_door1 = self.compute_distance_to_door1(self._previous_pos, state)
            _, me_cost_now_door2 = self.compute_distance_to_door2(me, state)
            _, me_cost_prev_door2 = self.compute_distance_to_door2(self._previous_pos, state)
            _, me_cost_now_pig = self.compute_distance_to_pig(me, target, state)
            _, me_cost_prev_pig = self.compute_distance_to_pig(self._previous_pos,target, state)

            _, other_cost_now_door1 = self.compute_distance_to_door1(other, state)
            _, other_cost_prev_door1 = self.compute_distance_to_door1(self._previous_other_pos, state)
            _, other_cost_now_door2 = self.compute_distance_to_door2(other, state)
            _, other_cost_prev_door2 = self.compute_distance_to_door2(self._previous_other_pos, state)
            other_acc_now_pig, other_cost_now_pig = self.compute_distance_to_pig(other, self._previous_target_pos, state, True)
            _, other_cost_prev_pig = self.compute_distance_to_pig(self._previous_other_pos, self._previous_target_pos, state)

            other_delta_pig = other_cost_now_pig - other_cost_prev_pig
            other_delta_doors = min(other_cost_now_door1 - other_cost_prev_door1, other_cost_now_door2 - other_cost_prev_door2)

            print()
            print('pig was at %s and now is at %s' % (self._previous_target_pos, target))
            print('prev_pig vs now_pig', other_cost_prev_pig, other_cost_now_pig)
            print('prev_d1 vs now_d1', other_cost_prev_door1, other_cost_now_door1)
            print('prev_d2 vs now_d2', other_cost_prev_door2, other_cost_now_door2)

            self._belief = self.belief_update(self._belief, other_delta_pig, other_delta_doors)
            self._desires = self.options(self._belief, min(me_cost_now_door1, me_cost_now_door2),
                                         min(other_cost_now_door1, other_cost_now_door2), target, state)
            self._intentions = self.filter(self._belief, self._desires, self._intentions, target,
                                           self.DOOR1_POS if me_cost_now_door2 > me_cost_now_door1 else self.DOOR2_POS,
                                           other_acc_now_pig[-2][1] if len(other_acc_now_pig) > 1 else other[1], state)
            self._action_list = self.plan(self._belief, self._intentions, me, state)

            # print('belief', self._belief)
            # print('desires', self._desires)
            # print('intention', self._intentions)
            # if not self._previous_target_pos == target:
            #     # Target has moved, or this is the first action of a new mission - calculate a new action list
            #     self._previous_target_pos = target
            #
            #     path = self.plan(self._belief, self._intentions, me, state)
            #     self._action_list = path
            self._previous_other_pos = other
            self._previous_pos = me
            self._previous_target_pos = target

            
            if self._action_list is not None and len(self._action_list) > 0:
                action = self._action_list.popleft()
                return AASMAAgent.ACTIONS.index(action)


            # reached end of action list - turn on the spot
            return AASMAAgent.ACTIONS.index("turn 1")  # substitutes for a no-op command

        except Exception as e:
            print(state2)
            print('error')
            print(e)
            return AASMAAgent.ACTIONS.index("turn 1")  # substitutes for a no-op command

    def belief_update(self, belief, other_delta_pig, other_delta_doors):
        if other_delta_doors == 0:
            # not going to doors
            print ("good")
            return min(1., belief + self.alpha)
        elif other_delta_pig == 0:
            # very weird behavior. random actions?
            print("something fishy")
            return belief - self.alpha ** 2
        if other_delta_pig < other_delta_doors:
            # he has come closer to the pig! (good guy)
            print("good guy")
            return min(belief + self.alpha, 1.0)
        elif other_delta_pig > other_delta_doors:
            # he has come closer to the doors! (traitor)
            print("traitor")
            return max(0.0, belief-self.alpha)
        else:
            if other_delta_pig > 0:
                # wtf.... random agent detected!
                return max(0.0, belief - self.alpha)
            # i dont know what he is up to
            print("might be good, might be bad")
            return belief

    def options(self, belief, me_cost_door, other_cost_door, pig_pos, state):
        adj_positions = self.neighbors(pig_pos, state)
        if len(adj_positions) > 2:
            # impossible to catch!
            if me_cost_door < other_cost_door:
                # I'm closer
                if abs(me_cost_door - other_cost_door) <= 1:
                    return "defect"
                else:
                    return "stay"
            else:
                return "cooperate"

        if len(adj_positions) == 1:
            return "cooperate"

        if belief > self.threshold:
            # he is going to cooperate
            return "cooperate"
        else:
            if me_cost_door > other_cost_door:
                return "cooperate"
            else:
                return "defect"

    def filter(self, belief, desires, intentions, pig_pos, best_door_pos, other_state, state):

        if desires == "cooperate":
            adj_positions = self.neighbors(pig_pos, state)
            if len(adj_positions) == 2:
                # possible to catch
                if other_state == adj_positions[0]:
                    return adj_positions[1]
                else:
                    return adj_positions[0]

            return pig_pos
        elif desires == "stay":
            return best_door_pos
        else:
            return best_door_pos

    def plan(self, belief, intentions, curr_pos, state):
        actions, _ = a_star(curr_pos, intentions, state)
        if intentions == "cooperate":
            return actions[:-1]
        else:
            return actions

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
