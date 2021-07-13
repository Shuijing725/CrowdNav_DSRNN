import logging
import argparse
import os
import sys
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.evaluation import evaluate
from crowd_sim import *
from pytorchBaselines.a2c_ppo_acktr.model import Policy


def main():
	# the following parameters will be determined for each test run
	parser = argparse.ArgumentParser('Parse configuration file')
	# the model directory that we are testing
	parser.add_argument('--model_dir', type=str, default='data/example_model')
	parser.add_argument('--visualize', default=False, action='store_true')
	# if -1, it will run 500 different cases; if >=0, it will run the specified test case repeatedly
	parser.add_argument('--test_case', type=int, default=-1)
	# model weight file you want to test
	parser.add_argument('--test_model', type=str, default='27776.pt')
	test_args = parser.parse_args()

	from importlib import import_module
	model_dir_temp = test_args.model_dir
	if model_dir_temp.endswith('/'):
		model_dir_temp = model_dir_temp[:-1]
	# import config class from saved directory
	# if not found, import from the default directory
	try:
		model_dir_string = model_dir_temp.replace('/', '.') + '.configs.config'
		model_arguments = import_module(model_dir_string)
		Config = getattr(model_arguments, 'Config')
	except:
		print('Failed to get Config function from ', test_args.model_dir, '/config.py')
		from crowd_nav.configs.config import Config


	config = Config()



	# configure logging and device
	# print test result in log file
	log_file = os.path.join(test_args.model_dir,'test')
	if not os.path.exists(log_file):
		os.mkdir(log_file)
	if test_args.visualize:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_visual.log')
	else:
		log_file = os.path.join(test_args.model_dir, 'test', 'test_'+test_args.test_model+'.log')



	file_handler = logging.FileHandler(log_file, mode='w')
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	logging.info('robot FOV %f', config.robot.FOV)
	logging.info('humans FOV %f', config.humans.FOV)

	torch.manual_seed(config.env.seed)
	torch.cuda.manual_seed_all(config.env.seed)
	if config.training.cuda:
		if config.training.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False


	torch.set_num_threads(1)
	device = torch.device("cuda" if config.training.cuda else "cpu")

	logging.info('Create other envs with new settings')


	if test_args.visualize:
		fig, ax = plt.subplots(figsize=(7, 7))
		ax.set_xlim(-6, 6)
		ax.set_ylim(-6, 6)
		ax.set_xlabel('x(m)', fontsize=16)
		ax.set_ylabel('y(m)', fontsize=16)
		plt.ion()
		plt.show()
	else:
		ax = None


	load_path=os.path.join(test_args.model_dir,'checkpoints', test_args.test_model)
	print(load_path)


	env_name = config.env.env_name

	recurrent_cell = 'GRU'

	eval_dir = os.path.join(test_args.model_dir,'eval')
	if not os.path.exists(eval_dir):
		os.mkdir(eval_dir)

	envs = make_vec_envs(env_name, config.env.seed, 1,
						 config.reward.gamma, eval_dir, device, allow_early_resets=True,
						 config=config, ax=ax, test_case=test_args.test_case)

	actor_critic = Policy(
		envs.observation_space.spaces,  # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=config,
		base=config.robot.policy)

	actor_critic.load_state_dict(torch.load(load_path, map_location=device))
	actor_critic.base.nenv = 1

	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)

	ob_rms = False

	# actor_critic, ob_rms, eval_envs, num_processes, device, num_episodes
	evaluate(actor_critic, ob_rms, envs, 1, device, config, logging, test_args.visualize, recurrent_cell)


if __name__ == '__main__':
	main()
