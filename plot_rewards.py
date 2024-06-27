# import matplotlib.pyplot as plt
# import pandas as pd
# path1 = './summary/CartPole-v1_custom_reward_False_env_rewards_True_500000_1719342762.csv'
# path2 = './summary/CartPole-v1_custom_reward_True_env_rewards_False_500000_1719336126.csv'
# edf = pd.read_csv(path1)
# cdf = pd.read_csv(path2)

# id_1 = path1.split('_')[-1].split('.')[0]
# id_2 = path2.split('_')[-1].split('.')[0]

# log_episodic_returns = edf['env_reward']
# log_custom_episode_returns = cdf['env_reward']

# # x_values = range(len(log_env_rewards))
# x_values_env = range(len(log_episodic_returns))
# x_values_cutom = range(len(log_custom_episode_returns))
# print(f"{len(log_episodic_returns) = }")
# print(f"{len(log_custom_episode_returns) = }")
# # print(f"{log_episodic_returns[:5] = }")
# # print(f"{log_custom_episode_returns[:5] = }")
# # plt.plot(x_values, log_env_rewards, label='env_rewards')
# # plt.plot(x_values, log_custom_rewards, label='custom_rewards')
# from scipy.ndimage.filters import gaussian_filter1d
# sigma=10
# smoothed_episodic_returns = gaussian_filter1d(log_episodic_returns, sigma=sigma)
# smoothed_custom_episode_returns = gaussian_filter1d(log_custom_episode_returns, sigma=sigma)

# # Plot the smoothed values
# plt.plot(x_values_env, smoothed_episodic_returns, label='Env', color='blue')
# plt.plot(x_values_cutom, smoothed_custom_episode_returns, label='Custom', color='red')

# plt.plot(x_values_env, log_episodic_returns, alpha=0.2, color='blue')
# plt.plot(x_values_cutom, log_custom_episode_returns, alpha=0.2, color='red')
# plt.xlabel('Number of Episodes')
# plt.ylabel('Episodic Returns')
# plt.legend()
# plt.title(f"{'cartpole-v1'}")
# plt.savefig(f"rewards_{id_1}_{id_2}.png")

import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter1d
import pandas as pd

def plot_rewards(path, label, color):
    df = pd.read_csv(path)
    rewards = df['env_reward']
    x_values = range(len(rewards))
    smoothed_rewards = gaussian_filter1d(rewards, sigma=sigma)
    plt.plot(x_values, smoothed_rewards, label=label, color=color)
    plt.plot(x_values, rewards, alpha=0.2, color=color)

def path_id(path):
  if path != '':
    return '_' + path.split('_')[-1].split('.')[0]
  return ''

path1 = './summary/CartPole-v1_custom_reward_False_env_rewards_True_500000_1719342762.csv'
path2 = './summary/CartPole-v1_custom_reward_True_env_rewards_False_500000_1719336126.csv'
path3 = ''


sigma = 15
plot_rewards(path1, 'Env', 'blue')
plot_rewards(path2, 'Custom', 'red')
plot_rewards(path3, 'Custom + Env', 'green') if path3 != '' else None

plt.xlabel('Number of Episodes')
plt.ylabel('Episodic Returns')
plt.legend()
plt.title('cartpole-v1')
plt.savefig(f"rewards{path_id(path1)}{path_id(path2)}{path_id(path3)}.png")
