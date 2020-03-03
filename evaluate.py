from model import GaussianPolicy
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np
from matplotlib import animation
import pybullet
import pybullet_envs
import time

# actor_path = "models/sac_actor_BipedalWalkerHardcore-v3_adiayn_1_8000.pth" # 700
# env_name = 'BipedalWalkerHardcore-v3'
# env_name = 'Pendulum-v0'
env_name = 'AntBulletEnv-v0'
# actor_path = "models/sac_actor_Pendulum-v0_adiayn_0_4000.pth" # 700
# actor_path = "models/sac_actor_AntBulletEnv-v0_adiayn_0_24000.pth"
actor_path = "/home/fernando/Downloads/adiayn/sac_actor_AntBulletEnv-v0_adiayn_0_1329.pth"


cuda = True
seed = 123456
env = gym.make(env_name, render="human")
# env.seed(seed)
env._max_episode_steps = 1000
num_skills = 2
skill = 1
pybullet_env = "Bullet" in env_name

num_inputs = env.observation_space.shape[0]
action_space = env.action_space
hidden_size = 256
device = torch.device("cuda" if cuda else "cpu")
policy = GaussianPolicy(num_inputs+num_skills, action_space.shape[0], hidden_size, action_space).to(device)
policy.load_state_dict(torch.load(actor_path))
policy.eval()

def display_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    #plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        print(frames[i].shape)
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=300)
    #display(anim)

time.sleep(2)
state = env.reset()
time.sleep(2)
# if pybullet_env:
#     print("sleep")
#     pybullet.setRealTimeSimulation(1)
#     time.sleep(2)
#     pybullet.setRealTimeSimulation(0)
cum_reward = 0
frames = []
max_steps = 250
# policy.eval()
z_one_hot = np.zeros(num_skills)
z_one_hot[skill] = 1.
for t in range(max_steps):
    print(state)
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

from PIL import Image
for i, frame in enumerate(frames):
    img = Image.fromarray(frame)
    img.save("/tmp/"+str(i).zfill(3)+'.png')

