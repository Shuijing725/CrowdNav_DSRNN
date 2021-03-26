from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState


class Human(Agent):
    # see Agent class in agent.py for details!!!
    def __init__(self, config, section):
        super().__init__(config, section)
        self.isObstacle = False # whether the human is a static obstacle (part of wall) or a moving agent

    def act(self, ob):
        """
        The state for human is its full state and all other agents' observable states
        :param ob:
        :return:
        """

        state = JointState(self.get_full_state(), ob)
        action = self.policy.predict(state)
        return action
