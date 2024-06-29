# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import sys
import random
import time
from dataclasses import dataclass
print(os.getcwd())  # /home/.../cleanrl/cleanrl => dir where python run command is called

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from utils.catdogmodel import CatAndDogConvNet
# from RlvlmfCNNmodel import gen_image_net
from utils.utils_train_reward_model import prepocess_image
import matplotlib.pyplot as plt
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device='cpu'
print(f"\n{device = }")
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import csv

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1 # previous value = 4
    """the number of parallel game environments"""
    num_steps: int = 128 # previous value: 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    use_custom_reward: bool = False
    use_env_reward: bool = False
    record_every: int = 100
    agent_model_path: str = os.path.join(os.getcwd(), "trained_policies/Agent_CartPole-v0_True_with_env_rewards_50000_1718960761.pth")


def make_env(env_id, idx, capture_video, run_name, record_every):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}", episode_trigger=lambda x: (x+1) % record_every == 0)
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

def designed_reward_fun(cart_position, pole_angle):
    max_distance  = 2.4
    max_angle     = 0.2095
    return (max_distance - abs(cart_position)) * (max_angle - abs(pole_angle)) / (max_distance * max_angle)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_time = int(time.time())
    run_name = f"trained_{args.env_id}__{args.exp_name}__{args.seed}__{args.use_custom_reward}__env_rewards_{args.use_env_reward}__{args.num_steps}__{args.total_timesteps}__{run_time}"

    print(f"{args.num_iterations = }")
    # load reward model
    # reward_models_dir = os.path.join(os.getcwd(), 'trained_reward_models')
    # env_name_0 = args.env_id.split("-")[0]
    # model_path = os.path.join(reward_models_dir, f"reward_model_catanddog_{env_name_0}.pth")
    # h, w = 224, 224
    # if 'rlvlmf' in model_path:
    #     reward_model = gen_image_net(image_height=h, image_width=w)
    # else:
    #     reward_model = CatAndDogConvNet()
    # reward_model.load_state_dict(torch.load(model_path))
    # reward_model.eval()

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.record_every) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs).to(device)

    if args.use_custom_reward and args.use_env_reward:
        args.agent_model_path = os.path.join(os.getcwd(), "trained_policies/Agent_CartPole-v0_True_with_env_rewards_50000_1718960761.pth")
    elif args.use_custom_reward:
        args.agent_model_path = os.path.join(os.getcwd(), "trained_policies/Agent_CartPole-v0_True_without_env_rewards_50000_1718963690.pth")
    
    agent.load_state_dict(torch.load(args.agent_model_path))
    agent.eval()
    print(f"Using env rewards : {args.use_env_reward}")
    print(f"Using agent       : {args.agent_model_path}")
    
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    log_env_rewards = []
    log_custom_rewards = []
    log_episodic_returns = []
    episode_rewards = []
    log_custom_episode_returns = []
    count_episodes = 0
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            state = next_obs[0].tolist()
            designed_reward = designed_reward_fun(state[0], state[2])

            #################### get reward from custom reward model  ###########################
            img_b = envs.call('render')
            img = img_b[0]
            # save image of state
            image_dir = f'replay/{args.env_id}/state_images'
            os.system(f'mkdir -p {image_dir}')
            filename = f"{str(global_step).zfill(5)}_{str(step).zfill(3)}.png"
            plt.imsave(f"{image_dir}/{filename}", img) # (400, 600, 3)

            # save state vector and reward
            with open(os.path.join(os.getcwd(), f'replay/{args.env_id}/state_values_{args.total_timesteps}_{run_time}.csv'), 'a', newline='') as file:
                csv_writer = csv.writer(file)
                if global_step - args.num_envs == 0:
                    csv_writer.writerow(['filename', 'cart_position', 'cart_velocity', 'pole_angle','pole_velocity', 'reward'])
                row = [filename] + state + [reward[0]]
                csv_writer.writerow(row)      
            # preprocess image
            # img = Image.fromarray(img)    # (600, 400) pil image
            # img = prepocess_image(img)    # (3,224,224) tensor image
            ############## for square image dimensions  ##########################
            # img = img.resize((500,500))
            # preprocess = transforms.Compose([
            #                 transforms.Resize(224),
            #                 transforms.ToTensor(),
            #                 transforms.Normalize((0.5,), (0.5,))
            #             ])
            # img = preprocess(img)
            ######################################################################
            # img = transforms.ToTensor()(img)

            # img_pil = Image.fromarray(img)    # (600, 400) pil image
            # img = prepocess_image(img_pil)    # (3,224,224) tensor image
            # print(f"{img.shape = } -- preprocessed")
            
            # img_pil.save(filepath)
            # print(f"{img_pil.size = }")

            # from torchvision.transforms import ToPILImage
            # img_tmp = img.cpu()
            # pil_image = ToPILImage()(img_tmp)
            # pil_image.save("pil_image_ppo.png", 'png')
            # print(f"{pil_image.size = }")

            # query reward model
            custom_reward = torch.tensor([0.0]).to(device) 
            if args.use_custom_reward:
                with torch.no_grad():
                    custom_reward = reward_model(img.unsqueeze(0).to(device))
                    # if 'sigmoid' in args.model_file_name:
                    #     custom_reward = (reward_model(img.unsqueeze(0).to(device))).cpu().view(-1).numpy()
                    # else:
                    #     custom_reward = torch.nn.Sigmoid()(reward_model(img.unsqueeze(0).to(device))).cpu().view(-1).numpy()
            
            env_reward = torch.tensor([0.0]).to(device) 
            if args.use_env_reward:
                env_reward = torch.tensor(reward).to(device) 
            
            combined_reward = custom_reward + env_reward 
            
            rewards[step] = combined_reward.view(-1) # default: 'reward' instead of 'combined_reward'
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            log_env_rewards.append(reward[0])
            log_custom_rewards.append(custom_reward[0])
            episode_rewards.append(custom_reward[0])
            # print(reward)

            # save state image with reward value
            # image_dir = f'state_images/{args.env_id}'
            # os.system(f'mkdir -p {image_dir}')
            # filename = f"step_{str(global_step).zfill(5)}_reward_{round(custom_reward[0])}.png"
            # pil_image.save(filename, 'png')


            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        log_episodic_returns.append(info['episode']['r'][0])
                        log_custom_episode_returns.append(sum(episode_rewards))
                        episode_rewards = []
                        count_episodes += 1
                        # print(f"{count_episodes = }")
                        # print(f"\n{iteration}\t{step}\t{info = }")
            
            # save state values and reward value
            # with open(os.path.join(os.getcwd(), f'state_images_{args.env_id}', 'state_image_data.csv'), 'a', newline='') as file:
            #   csv_writer = csv.writer(file)
            #   if global_step - args.num_envs == 0:
            #     csv_writer.writerow(['filename', 'cart_position', 'cart_velocity', 'pole_angle','pole_velocity', 'reward'])
            #   row = [filename] + state + [custom_reward[0]]
            #   csv_writer.writerow(row)
        
    envs.close()
    writer.close()

    # plot rewards
    import matplotlib.pyplot as plt

    # x_values = range(len(log_env_rewards))
    x_values = range(len(log_episodic_returns))
    print(f"{len(x_values) = }")
    print(f"{len(log_episodic_returns) = }")
    print(f"{log_episodic_returns[:5] = }")
    print(f"{log_custom_episode_returns[:5] = }")
    # plt.plot(x_values, log_env_rewards, label='env_rewards')
    # plt.plot(x_values, log_custom_rewards, label='custom_rewards')
    from scipy.ndimage.filters import gaussian_filter1d
    sigma=4
    smoothed_episodic_returns = gaussian_filter1d(log_episodic_returns, sigma=sigma)
    smoothed_custom_episode_returns = gaussian_filter1d(log_custom_episode_returns, sigma=sigma)

    # Plot the smoothed values
    plt.plot(x_values, smoothed_episodic_returns, label='Env Episodic Returns', color='blue')
    plt.plot(x_values, smoothed_custom_episode_returns, label='Custom Episodic Returns', color='red')
    
    plt.plot(x_values, log_episodic_returns, alpha=0.2, color='blue')
    plt.plot(x_values, log_custom_episode_returns, alpha=0.2, color='red')
    plt.xlabel('Steps')
    plt.ylabel('Returns')
    plt.title(f"{args.env_id}")
    plt.legend()
    plt.savefig(f"trained_policy_rewards_plot_custom_reward_{args.use_custom_reward}_env_rewards_{args.use_env_reward}_{args.total_timesteps}_{args.num_steps}_{run_time}.png")

    print(f"cur time: {time.time()}")