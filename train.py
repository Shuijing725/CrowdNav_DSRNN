import sys
import logging
import os
import shutil
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from pytorchBaselines.a2c_ppo_acktr import algo, utils
from arguments import get_args
from pytorchBaselines.a2c_ppo_acktr.envs import make_vec_envs
from pytorchBaselines.a2c_ppo_acktr.model import Policy
from pytorchBaselines.a2c_ppo_acktr.storage import RolloutStorage


from crowd_nav.configs.config import Config
from crowd_sim import *

def main():
	# arguments
	algo_args = get_args()


	# save policy to output_dir
	if os.path.exists(algo_args.output_dir) and algo_args.overwrite: # if I want to overwrite the directory
		shutil.rmtree(algo_args.output_dir)  # delete an entire directory tree

	if not os.path.exists(algo_args.output_dir):
		os.makedirs(algo_args.output_dir)

	shutil.copytree('crowd_nav/configs', os.path.join(algo_args.output_dir, 'configs'))
	shutil.copy('arguments.py', algo_args.output_dir)


	# configure logging
	log_file = os.path.join(algo_args.output_dir, 'output.log')
	mode = 'a' if algo_args.resume else 'w'
	file_handler = logging.FileHandler(log_file, mode=mode)
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")


	config = Config()



	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False



	torch.set_num_threads(algo_args.num_threads)
	device = torch.device("cuda" if algo_args.cuda else "cpu")


	logging.info('Create other envs with new settings')


	# For fastest training: use GRU
	env_name = algo_args.env_name
	recurrent_cell = 'GRU'

	# Create a wrapped, monitored VecEnv
	envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
						 algo_args.gamma, None, device, False, envConfig=config)



	actor_critic = Policy(
		envs.observation_space.spaces, # pass the Dict into policy to parse
		envs.action_space,
		base_kwargs=algo_args,
		base=config.robot.policy)

	rollouts = RolloutStorage(algo_args.num_steps,
							  algo_args.num_processes,
							  envs.observation_space.spaces,
							  envs.action_space,
							  algo_args.human_node_rnn_size,
							  algo_args.human_human_edge_rnn_size,
							  recurrent_cell_type=recurrent_cell)

	if algo_args.resume: #retrieve the model if resume = True
		load_path = algo_args.load_path
		actor_critic, _ = torch.load(load_path)


	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	nn.DataParallel(actor_critic).to(device)


	agent = algo.PPO(
		actor_critic,
		algo_args.clip_param,
		algo_args.ppo_epoch,
		algo_args.num_mini_batch,
		algo_args.value_loss_coef,
		algo_args.entropy_coef,
		lr=algo_args.lr,
		eps=algo_args.eps,
		max_grad_norm=algo_args.max_grad_norm)



	obs = envs.reset()
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)

	rollouts.to(device)

	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(
		algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes

	for j in range(num_updates):

		if algo_args.use_linear_lr_decay:
			utils.update_linear_schedule(
				agent.optimizer, j, num_updates,
				agent.optimizer.lr if algo_args.algo == "acktr" else algo_args.lr)

		for step in range(algo_args.num_steps):
			# Sample actions
			with torch.no_grad():

				rollouts_obs = {}
				for key in rollouts.obs:
					rollouts_obs[key] = rollouts.obs[key][step]
				rollouts_hidden_s = {}
				for key in rollouts.recurrent_hidden_states:
					rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][step]
				value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
					rollouts_obs, rollouts_hidden_s,
					rollouts.masks[step])


			# Obser reward and next obs
			obs, reward, done, infos = envs.step(action)
			# print(done)

			for info in infos:
				# print(info.keys())
				if 'episode' in info.keys():
					episode_rewards.append(info['episode']['r'])

			# If done then clean the history of observations.
			masks = torch.FloatTensor(
				[[0.0] if done_ else [1.0] for done_ in done])
			bad_masks = torch.FloatTensor(
				[[0.0] if 'bad_transition' in info.keys() else [1.0]
				 for info in infos])
			rollouts.insert(obs, recurrent_hidden_states, action,
							action_log_prob, value, reward, masks, bad_masks)

		with torch.no_grad():
			rollouts_obs = {}
			for key in rollouts.obs:
				rollouts_obs[key] = rollouts.obs[key][-1]
			rollouts_hidden_s = {}
			for key in rollouts.recurrent_hidden_states:
				rollouts_hidden_s[key] = rollouts.recurrent_hidden_states[key][-1]
			next_value = actor_critic.get_value(
				rollouts_obs, rollouts_hidden_s,
				rollouts.masks[-1]).detach()




		rollouts.compute_returns(next_value, algo_args.use_gae, algo_args.gamma,
								 algo_args.gae_lambda, algo_args.use_proper_time_limits)

		value_loss, action_loss, dist_entropy = agent.update(rollouts)

		rollouts.after_update()

		# save the model for every interval-th episode or for the last epoch
		if (j % algo_args.save_interval == 0
			or j == num_updates - 1) :
			save_path = os.path.join(algo_args.output_dir, 'checkpoints')
			if not os.path.exists(save_path):
				os.mkdir(save_path)

			torch.save([
				actor_critic,
				getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
			], os.path.join(save_path, '%.5i'%j + ".pt"))

		if j % algo_args.log_interval == 0 and len(episode_rewards) > 1:
			total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
			end = time.time()
			print(
				"Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward "
				"{:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
					.format(j, total_num_steps,
							int(total_num_steps / (end - start)),
							len(episode_rewards), np.mean(episode_rewards),
							np.median(episode_rewards), np.min(episode_rewards),
							np.max(episode_rewards), dist_entropy, value_loss,
							action_loss))

			df = pd.DataFrame({'misc/nupdates': [j], 'misc/total_timesteps': [total_num_steps],
							   'fps': int(total_num_steps / (end - start)), 'eprewmean': [np.mean(episode_rewards)],
							   'loss/policy_entropy': dist_entropy, 'loss/policy_loss': action_loss,
							   'loss/value_loss': value_loss})

			if os.path.exists(os.path.join(algo_args.output_dir, 'progress.csv')) and j > 20:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='a', header=False, index=False)
			else:
				df.to_csv(os.path.join(algo_args.output_dir, 'progress.csv'), mode='w', header=True, index=False)




if __name__ == '__main__':
	main()
