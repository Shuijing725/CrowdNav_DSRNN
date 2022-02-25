
class BaseConfig(object):
    def __init__(self):
        pass


class Config(object):
    # environment settings
    env = BaseConfig()
    env.env_name = 'CrowdSimDict-v0'  # name of the environment
    env.time_limit = 50 # time limit of each episode (second)
    env.time_step = 0.25 # length of each timestep/control frequency (second)
    env.val_size = 100
    env.test_size = 500 # number of episodes for test.py
    env.randomize_attributes = True # randomize the preferred velocity and radius of humans or not
    env.seed = 0  # random seed for environment

    # reward function
    reward = BaseConfig()
    reward.success_reward = 10
    reward.collision_penalty = -20
    # discomfort distance for the front half of the robot
    reward.discomfort_dist_front = 0.25
    # discomfort distance for the back half of the robot
    reward.discomfort_dist_back = 0.25
    reward.discomfort_penalty_factor = 10
    reward.gamma = 0.99  # discount factor for rewards

    # environment settings
    sim = BaseConfig()
    sim.render = False # show GUI for visualization
    sim.circle_radius = 6 # radius of the circle where all humans start on
    sim.human_num = 5 # total number of humans
    # Group environment: set to true; FoV environment: false
    sim.group_human = False

    # human settings
    humans = BaseConfig()
    humans.visible = True # a human is visible to other humans and the robot
    # policy to control the humans: orca or social_force
    humans.policy = "orca"
    humans.radius = 0.3 # radius of each human
    humans.v_pref = 1 # max velocity of each human
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = True
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    # robot settings
    robot = BaseConfig()
    robot.visible = False  # the robot is visible to humans
    # robot policy: srnn for now
    robot.policy = 'srnn'
    robot.radius = 0.3  # radius of the robot
    robot.v_pref = 1  # max velocity of the robot
    # robot FOV = this values * PI
    robot.FOV = 2.

    # add noise to observation or not
    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    # robot action type
    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic"

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 2.
    sf.B = 1
    sf.KI = 1

    # cofig for RL ppo
    ppo = BaseConfig()
    ppo.num_mini_batch = 2  # number of batches for ppo
    ppo.num_steps = 30  # number of forward steps
    ppo.recurrent_policy = True  # use a recurrent policy
    ppo.epoch = 5  # number of ppo epochs
    ppo.clip_param = 0.2  # ppo clip parameter
    ppo.value_loss_coef = 0.5  # value loss coefficient
    ppo.entropy_coef = 0.0  # entropy term coefficient
    ppo.use_gae = True  # use generalized advantage estimation
    ppo.gae_lambda = 0.95  # gae lambda parameter

    # SRNN config
    SRNN = BaseConfig()
    # RNN size
    SRNN.human_node_rnn_size = 128  # Size of Human Node RNN hidden state
    SRNN.human_human_edge_rnn_size = 256  # Size of Human Human Edge RNN hidden state

    # Input and output size
    SRNN.human_node_input_size = 3  # Dimension of the node features
    SRNN.human_human_edge_input_size = 2  # Dimension of the edge features
    SRNN.human_node_output_size = 256  # Dimension of the node output

    # Embedding size
    SRNN.human_node_embedding_size = 64  # Embedding size of node features
    SRNN.human_human_edge_embedding_size = 64  # Embedding size of edge features

    # Attention vector dimension
    SRNN.attention_size = 64  # Attention size

    # training config
    training = BaseConfig()
    training.lr = 4e-5  # learning rate (default: 7e-4)
    training.eps = 1e-5  # RMSprop optimizer epsilon
    training.alpha = 0.99  # RMSprop optimizer alpha
    training.max_grad_norm = 0.5  # max norm of gradients
    training.num_env_steps = 10e6  # number of environment steps to train: 10e6 for holonomic, 20e6 for unicycle
    training.use_linear_lr_decay = False  # use a linear schedule on the learning rate: True for unicycle, False for holonomic
    training.save_interval = 200  # save interval, one save per n updates
    training.log_interval = 20  # log interval, one log per n updates
    training.use_proper_time_limits = False  # compute returns taking into account time limits
    training.cuda_deterministic = False  # sets flags for determinism when using CUDA (potentially slow!)
    training.cuda = True  # use CUDA for training
    training.num_processes = 12 # how many training CPU processes to use
    training.output_dir = 'data/dummy'  # the saving directory for train.py
    training.resume = False  # resume training from an existing checkpoint or not
    training.load_path = 'data/example_model/checkpoints/27776.pt'  # if resume = True, load from the following checkpoint
    training.overwrite = True  # whether to overwrite the output directory in training
    training.num_threads = 1  # number of threads used for intraop parallelism on CPU