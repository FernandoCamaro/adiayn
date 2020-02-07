from model import GaussianPolicy
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np
from matplotlib import animation

actor_path = "models/sac_actor_Pendulum-v0_diayn_1100.pth" # 700
env_name = 'Pendulum-v0'
cuda = True
seed = 123456
env = gym.make(env_name)
env.seed(seed)
env._max_episode_steps = 500
num_skills = 2
skill = 0

num_inputs = env.observation_space.shape[0]
action_space = env.action_space
hidden_size = 256
device = torch.device("cuda" if cuda else "cpu")
policy = GaussianPolicy(num_inputs+num_skills, action_space.shape[0], hidden_size, action_space).to(device)
policy.load_state_dict(torch.load(actor_path))

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=300)
    #display(anim)

state = env.reset()
cum_reward = 0
frames = []
max_steps = 9000
policy.eval()
z_one_hot = np.zeros(num_skills)
z_one_hot[skill] = 1.
for t in range(max_steps):
    state = np.concatenate((state,z_one_hot))
    # Render into buffer. 
    frames.append(env.render(mode = 'rgb_array'))
    state = torch.FloatTensor(state).to(device).unsqueeze(0)
    _, _, action = policy.sample(state) # evaluate is true
    #action, _, _ = policy.sample(state) # evaluate is False
    action = action.detach().cpu().numpy()[0]
    state, reward, done, info = env.step(action)
    cum_reward += reward
    if done:
        break
print(done, t)
print(cum_reward)
env.close()
display_frames_as_gif(frames)

