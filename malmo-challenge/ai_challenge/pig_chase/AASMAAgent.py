
from malmopy.agent import BaseAgent
from common import visualize_training, Entity, ENV_TARGET_NAMES, ENV_ENTITIES, ENV_AGENT_NAMES,\
        ENV_ACTIONS, ENV_CAUGHT_REWARD, ENV_BOARD_SHAPE
class AASMAAgent(BaseAgent):
    ACTIONS = ENV_ACTIONS
    DOOR1_POS = (4, 1)
    DOOR2_POS = (4, 7)
    alpha = 0.1
    initial_belief = 1
    def __init__(self, name, other_agent, target, visualizer = None):
        super(AASMAAgent, self).__init__(self, name, len(AASMAAgent.ACTIONS))
        self._target = str(target)
        self._other_agent = str(other_agent)
        self._previous_target_pos = None
        self._previous_other_pos = None
        self._previous_pos = None
        self._action_list = []
        self._belief = self.initial_belief
        self._desires = []
        self._intentions = []

    def act(self, state, reward, done, is_training=False):
        if done:
            self._action_list = []
            self._previous_target_pos = None
            self._previous_other_pos = None
            self._desires = []
            self._intentions = []

        entities = state[1]
        state = state[0]


        # retrieve information from state

        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k]
        me_details = [e for e in entities if e['name'] == self.name][0]

        other = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self._other_agent in k]
        other_details = [e for e in entities if e['name'] == self._other_agent][0]

        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

        other_yaw = int(other_details['yaw'])
        other_direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.

        other = (other, other_direction)
        me = (me, direction)

        target = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self._target in k]

        other_delta_pig, other_delta_doors = self.compute_deltas(self._previous_other_pos, other, self._previous_target_pos, state)
        me_delta_pig, me_delta_doors = self.compute_deltas(self._previous_pos, me, self._previous_target_pos, state)

        self._belief = self.belief_update(self._belief, other_delta_pig, other_delta_doors)
        self._desires = self.options(self._belief, me, other)
        self._intentions = self.filter(self._belief, self._desires, self._intentions)
        self._action_list = self.plan(self._belief, self._intentions)


        if not self._previous_target_pos == target:
            # Target has moved, or this is the first action of a new mission - calculate a new action list
            self._previous_target_pos = target

            path = self.plan(self._belief, self._intentions)
            self._action_list = []
            for point in path:
                self._action_list.append(point.action)

        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return AASMAAgent.ACTIONS.index(action)

        self._previous_other_pos = other
        self._previous_pos = me
        # reached end of action list - turn on the spot
        return AASMAAgent.ACTIONS.index("turn 1")  # substitutes for a no-op command

    def belief_update(self, belief, other_delta_pig, other_delta_doors):
        if other_delta_pig < other_delta_doors:
            # he has come closer to the pig! (good guy)
            return min(belief + self.alpha, 1.0)
        elif other_delta_pig > other_delta_doors:
            # he has come closer to the doors! (traitor)
            return max(0.0, belief-self.alpha)
        else:
            # i dont know what he is up to
            return belief

    def options(self, belief, current_pos, other_curr_pos):
        pass

    def filter(self, belief, desires, intentions):
        pass

    def plan(self, belief, intentions):
        pass

    def compute_deltas(self, prev_pos, curr_pos, pig_pos, state):
        _, pig_cost_before = a_star(prev_pos, pig_pos, state)
        _, pig_cost_now = a_star(curr_pos, pig_pos, state)

        _, door1_cost_before = a_star(prev_pos, self.DOOR1_POS, state)
        _, door1_cost_now = a_star(curr_pos, self.DOOR1_POS, state)

        _, door2_cost_before = a_star(prev_pos, self.DOOR2_POS, state)
        _, door2_cost_now = a_star(curr_pos, self.DOOR2_POS, state)

        delta_pig = pig_cost_now - pig_cost_before
        delta_doors = min(door1_cost_now - door1_cost_before, door2_cost_now - door2_cost_before)

        return delta_pig, delta_doors