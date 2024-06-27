# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import sys
sys.path.append(os.getcwd()) # previous: "/raid/infolab/veerendra/shariq/workdir"
import random
import time
from dataclasses import dataclass
print(os.getcwd())  # /home/.../cleanrl/cleanrl => dir where python run command is called
# sys.exit()
# file path: /home/.../cleanrl/cleanrl/cleanrl/ppo.py

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
from utils.catdogmodel import CatAndDogConvNet, CatAndDogConvNetCustom
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
    capture_video: bool = True
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
    num_steps: int = 256 # previous value: 128
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
    use_designed_reward: bool = False
    record_every: int = 100
    model_name: str = 'catanddog'
    model_file_name: str = 'reward_model_catanddog_CartPole_imageSize_224.pth'


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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__custom_reward_{args.use_custom_reward}_env_rewards_{args.use_env_reward}__{args.num_steps}__{args.total_timesteps}__{run_time}"

    # load reward model
    reward_models_dir = os.path.join(os.getcwd(), 'trained_reward_models')
    env_name_0 = args.env_id.split("-")[0]

    if 'rlvlmf' in args.model_file_name:
        h, w = 224, 224
        reward_model = gen_image_net(image_height=h, image_width=w)
    elif args.model_name == 'catanddog':
        reward_model = CatAndDogConvNet()
    elif args.model_name == 'CatAndDogConvNetCustom':
        reward_model = CatAndDogConvNetCustom(image_height=400, image_width=600)

    model_path = os.path.join(reward_models_dir, args.model_file_name)
    reward_model.load_state_dict(torch.load(model_path))
    reward_model = reward_model.to(device)
    reward_model.eval()
    num_parameters = sum(p.numel() for p in reward_model.parameters() if p.requires_grad)
    print(f"Model Name      : {args.model_name}")
    print(f"Model File Name : {args.model_file_name}")
    print(f"Num Paramteres  : {num_parameters = }")
    print(f"custom: {args.use_custom_reward} env: {args.use_env_reward}")

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
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

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
    for iteration in tqdm(range(1, args.num_iterations + 1)):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

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

            #################### get reward from custom reward model  ###########################
            img_b = envs.call('render')
            img = img_b[0]
            # filename = f"image_globalstep_{str(global_step).zfill(4)}_envstep_{str(step).zfill(4)}_envid_{0}.png"
            # plt.imsave(f"{filename}", img)      # (400, 600, 3)
            # preprocess image
            img = Image.fromarray(img)    # (600, 400) pil image
            # img = prepocess_image(img)    # (3,224,224) tensor image
            ############## to resize (400,600) image to square image dimensions  ##########################
            if args.model_name == 'catanddog':
                img = img.resize((500,500))
                preprocess = transforms.Compose([
                                transforms.Resize(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                            ])
                img = preprocess(img)
            ######################################################################
            else:
                img = transforms.ToTensor()(img)

            # from torchvision.transforms import ToPILImage
            # img_tmp = img.cpu()
            # pil_image = ToPILImage()(img_tmp)
            # pil_image.save("pil_image_ppo.png", 'png')
            # print(f"{pil_image.size = }")

            # query reward model
            # reward.shape = (1,) type(reward) = <class 'numpy.ndarray'>
            custom_reward = np.array([0.0])
            if args.use_custom_reward:
                with torch.no_grad():
                    if 'sigmoid' in args.model_file_name:
                        custom_reward = (reward_model(img.unsqueeze(0).to(device))).cpu().view(-1).numpy()
                    else:
                        custom_reward = torch.nn.Sigmoid()(reward_model(img.unsqueeze(0).to(device))).cpu().view(-1).numpy()
            
            env_reward = np.array([0.0])
            if args.use_env_reward:
                env_reward = reward 
            
            designed_reward = np.array([0.0])
            if args.use_designed_reward:
                designed_reward = np.array(designed_reward_fun(state[0], state[2]))

            combined_reward = custom_reward + env_reward 
            #####################################################################################
            
            rewards[step] = torch.tensor(combined_reward).to(device).view(-1) # default: 'reward' instead of 'combined_reward'
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            episode_rewards.append(combined_reward[0])
            # log_env_rewards.append(reward[0])
            # log_custom_rewards.append(combined_reward[0])
            # print(reward)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                        log_episodic_returns.append(info['episode']['r'][0])
                        log_custom_episode_returns.append(sum(episode_rewards))
                        episode_rewards = []
                        # print(f"\n{iteration}\t{step}\t{info = }")
                        # if iteration % 3 == 0:
                        #     sys.exit()
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # Save logs to csv
    import csv
    os.system('mkdir -p summary')
    with open(os.path.join(os.getcwd(), f'summary/{args.env_id}_custom_reward_{args.use_custom_reward}_env_rewards_{args.use_env_reward}_{args.total_timesteps}_{run_time}.csv'), 'w', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['env_reward', 'custom_reward'])
        for x, y in zip(log_episodic_returns, log_custom_episode_returns):
            csv_writer.writerow([x,y])

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
    plt.plot(x_values, smoothed_custom_episode_returns, label='Custom Episodic Returns', color='green')
    
    plt.plot(x_values, log_episodic_returns, alpha=0.2, color='blue')
    plt.plot(x_values, log_custom_episode_returns, alpha=0.2, color='green')
    plt.xlabel('Steps')
    plt.ylabel('Returns')
    plt.legend()
    plt.title(f"{args.env_id}  custom: {args.use_custom_reward}, env: {args.use_env_reward}")
    plt.savefig(f"rewards_plot_custom_reward_{args.use_custom_reward}_env_rewards_{args.use_env_reward}_{args.total_timesteps}_{args.num_steps}_{run_time}.png")

    # save model
    os.system('mkdir -p trained_policies')
    torch.save(agent.state_dict(), os.path.join(os.getcwd(), f"trained_policies/Agent_{args.env_id}_custom_reward_{args.use_custom_reward}_env_rewards_{args.use_env_reward}_{args.total_timesteps}_{run_time}.pth"))

'''
python ppo.py --use_custom_reward --total-timesteps 500000 --model_name catanddog --model_file_name reward_model_catanddog_CartPole_imageSize_224.pth

nohup python ppo.py --use_env_reward --use_custom_reward --total-timesteps 500000 --model_name catanddog --model_file_name reward_model_catanddog_CartPole_imageSize_224.pth > "nohup_$(date +%s).txt" 2>&1 &

nohup python ppo.py --use_custom_reward --total-timesteps 5000 --model_name CatAndDogConvNetCustom --model_file_name reward_model_curated_prefs_sigmoid.pth > "nohup_$(( $(date +%s) + 2 )).txt" 2>&1 &
'''
