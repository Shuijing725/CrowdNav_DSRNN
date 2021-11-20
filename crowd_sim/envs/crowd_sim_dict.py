import gym
import numpy as np
from numpy.linalg import norm
import copy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs import CrowdSim

# The class for the simulation environment used for training a DSRNN policy

class CrowdSimDict(CrowdSim):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()

        self.desiredVelocity=[0.0,0.0]


    # define the observation space and the action space
    def set_robot(self, robot):
        self.robot = robot

        # set observation space and action space
        # we set the max and min of action/observation space as inf
        # clip the action and observation as you need

        d={}
        # robot node: px, py, r, gx, gy, v_pref, theta
        d['robot_node'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,7,), dtype = np.float32)
        # only consider the robot temporal edge and spatial edges pointing from robot to each human
        d['temporal_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,2,), dtype=np.float32)
        d['spatial_edges'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.human_num, 2), dtype=np.float32)
        self.observation_space=gym.spaces.Dict(d)

        high = np.inf * np.ones([2, ])
        self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)


    # generate observation for each timestep
    # reset = True: reset calls this function; reset = False: step calls this function
    def generate_ob(self, reset):
        ob = {}

        # nodes
        visible_humans, num_visibles, human_visibility = self.get_num_human_in_fov()

        ob['robot_node'] = self.robot.get_full_state_list_noV()

        self.update_last_human_states(human_visibility, reset=reset)

        # edges
        # temporal edge: robot's velocity
        ob['temporal_edges'] = np.array([self.robot.vx, self.robot.vy])
        # spatial edges: the vector pointing from the robot position to each human's position
        ob['spatial_edges'] = np.zeros((self.human_num, 2))
        for i in range(self.human_num):
            relative_pos = np.array([self.last_human_states[i, 0] - self.robot.px, self.last_human_states[i, 1] - self.robot.py])
            ob['spatial_edges'][i] = relative_pos

        return ob


    # reset function
    def reset(self, phase='train', test_case=None):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """

        if self.phase is not None:
            phase = self.phase
        if self.test_case is not None:
            test_case=self.test_case

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']
        if test_case is not None:
            self.case_counter[phase] = test_case # test case is passed in to calculate specific seed to generate case
        self.global_time = 0

        self.desiredVelocity = [0.0, 0.0]
        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter
        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.generate_robot_humans(phase)


        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]

        # get robot observation
        ob = self.generate_ob(reset=True)

        # initialize potential
        self.potential = -abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))


        return ob


    # step function
    def step(self, action, update=True):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        action = self.robot.policy.clip_action(action, self.robot.v_pref)

        if self.robot.kinematics == 'unicycle':
            self.desiredVelocity[0] = np.clip(self.desiredVelocity[0]+action.v,-self.robot.v_pref,self.robot.v_pref)
            action=ActionRot(self.desiredVelocity[0], action.r)


        human_actions = self.get_human_actions()

        # compute reward and episode info
        reward, done, episode_info = self.calc_reward(action)


        # apply action and update all agents
        self.robot.step(action)
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
        self.global_time += self.time_step # max episode length=time_limit/time_step


        # compute the observation
        ob = self.generate_ob(reset=False)

        info={'info':episode_info}


        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()
        
        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for human in self.humans:
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human)


        return ob, reward, done, info

