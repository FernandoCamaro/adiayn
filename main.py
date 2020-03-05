import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import torch.nn.functional as F
from torch.optim import Adam
from sac import SAC
from model import Discriminator
from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory
import pybullet
import pybullet_envs
import time
from envs.navigationEnv import NavigationEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="2DNavigation",
                    help='Environment (default: 2DNavigation)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 10000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--diayn', action="store_true",
                    help='do diayn (default: False)')
parser.add_argument('--num_skills', type=int, default=2, metavar='N',
                    help='number of skills for diayn (default: 2)')
parser.add_argument('--disc_start_epi', type=int, default=500, metavar='N',
                    help='episode in which the discriminator starts being trained (default: 500)')
parser.add_argument('--max_episode_steps', type=int, default=300, metavar='N',
                    help='maximum steps allowed in an episode (default: 300)')
args = parser.parse_args()

# Environment
env = NavigationEnv()
width = env.observation_space.shape[0]
height = env.observation_space.shape[1]
torch.manual_seed(args.seed)
np.random.seed(args.seed)
env.seed(args.seed)
pybullet_env = "Bullet" in args.env_name

# Agents
agent0 = SAC(width, height,env.action_space, args, handicaped=False)
agent1 = SAC(width, height, env.action_space, args, handicaped=True)

# Discriminator
# TODO: remove hardcoded num_inputs for discriminator
# TODO: num_agents should be handled properly once we know it works for the case of one agent in DIAYN.
num_agents = 1
discriminator = Discriminator( 256, int(args.num_skills*num_agents), args.hidden_size).to(agent0.device)
discriminator_optim = Adam(discriminator.parameters(), lr=args.lr)

#TesnorboardX
writer = SummaryWriter('runs/{}_SAC_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name, args.policy, "autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    if args.diayn:
        skill = np.random.randint(args.num_skills)
        z_one_hot = np.zeros(args.num_skills)
        z_one_hot[skill] = 1.
    # sample agent
    agent_id = 0 #np.random.randint(2)
    agent = agent0 if agent_id == 0 else agent1
    while (not done) and (episode_steps < args.max_episode_steps):
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state) if not args.diayn else agent.select_action(state, z_one_hot)# Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters the two agents
                critic_1_loss_0, critic_2_loss_0, policy_loss_0, ent_loss_0, alpha_0 = agent0.update_parameters(memory, args.batch_size, updates, discriminator)
                critic_1_loss_1, critic_2_loss_1, policy_loss_1, ent_loss_1, alpha_1 = agent1.update_parameters(memory, args.batch_size, updates, discriminator)
                
                # update discriminator
                if args.diayn and i_episode > args.disc_start_epi:
                    _, _, _, next_state, _, skills, one_hot_skills, agent_ids = memory.sample(batch_size=args.batch_size)
                    next_state = torch.FloatTensor(next_state).to(agent.device)
                    one_hot_skills = torch.FloatTensor(one_hot_skills).to(agent.device)
                    _ = agent.policy(next_state, one_hot_skills)#next_state[:,0:-args.num_skills]
                    next_state_for_disc = agent.policy.flat_state
                    skills     = torch.LongTensor(skills + args.num_skills*agent_ids ).to(agent.device)
                    logits     = discriminator(next_state_for_disc) 
                    discriminator_loss       = F.cross_entropy(logits, skills)
                    
                    discriminator_optim.zero_grad()
                    discriminator_loss.backward()
                    discriminator_optim.step()
                if updates % 1000 == 0:
                    writer.add_scalar('loss/critic_1_0', critic_1_loss_0, updates)
                    writer.add_scalar('loss/critic_2_0', critic_2_loss_0, updates)
                    writer.add_scalar('loss/policy_0', policy_loss_0, updates)
                    writer.add_scalar('loss/entropy_loss_0', ent_loss_0, updates)
                    writer.add_scalar('entropy_temprature/alpha_0', alpha_0, updates)
                    if args.diayn and i_episode > args.disc_start_epi:
                        writer.add_scalar('loss/discriminator_loss', discriminator_loss, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        if args.diayn:
            with torch.no_grad():
                skills     = torch.LongTensor([skill]).to(agent.device)
                one_hot_skills = torch.FloatTensor([z_one_hot]).to(agent.device)
                next_state_for_disc = torch.FloatTensor(next_state).unsqueeze(0).to(agent.device)
                _ = agent.policy(next_state_for_disc, one_hot_skills)#next_state[:,0:-args.num_skills]
                next_state_for_disc = agent.policy.flat_state
                logits     = discriminator(next_state_for_disc)
                # reward     = (-F.cross_entropy(logits, skills) - np.log(1/args.num_skills + 1e-6) -np.log(1/2. + 1E-6)).item()
                reward = F.softmax(logits, dim=1).gather(dim=1, index=skills.view(-1,1)).item()
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        if args.diayn:
            memory.push(state, action, reward, next_state, mask, skill, z_one_hot, agent_id)
        else:    
            memory.push(state, action, reward, next_state, mask) # Append transition to memory
        
        state = next_state

    if total_numsteps > args.num_steps:
        break
    
    skill_str =  "" if not args.diayn else str(skill)
    agent_id_str = "" if not args.diayn else str(agent_id)
    writer.add_scalar('reward/train_'+skill_str+'_'+agent_id_str, episode_reward, i_episode)
    writer.add_scalar('episode_len/train_'+skill_str+'_'+agent_id_str, episode_steps, i_episode)
    print("Episode: {}, skill: {}, agent: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, skill_str, agent_id_str, total_numsteps, episode_steps, round(episode_reward, 2)))

    if i_episode % 2000 == 0:
        agent0.save_model(args.env_name,'adiayn_0_'+str(i_episode))
        agent1.save_model(args.env_name,'adiayn_1_'+str(i_episode))

env.close()

