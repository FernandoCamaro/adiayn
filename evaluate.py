from model import GaussianPolicy
import matplotlib.pyplot as plt
import gym
import torch
import numpy as np
from matplotlib import animation
import pybullet
import time
from envs.navigationEnv import NavigationEnv
import os
import subprocess

actor_path = "/home/fernando/git/others/pytorch-soft-actor-critic/models/sac_actor_2DNavigation_adiayn_0_4000.pth"


cuda = True
seed = 123456
env = NavigationEnv()
width = env.observation_space.shape[0]
height = env.observation_space.shape[1]
env.seed(seed)
num_skills = 8


action_space = env.action_space
hidden_size = 256
device = torch.device("cuda" if cuda else "cpu")
policy = GaussianPolicy(width, height, num_skills, action_space.shape[0], hidden_size, action_space).to(device)
policy.load_state_dict(torch.load(actor_path))
policy.eval()

def save_images(frames, skill):
    dest_folder = os.path.join('/tmp',str(skill))
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    for i, frame in enumerate(frames):
        filename = os.path.join(dest_folder,"image_"+str(i).zfill(3)+'.png')
        frame.save(filename)
    p = subprocess.Popen(["convert", "-delay","10", "-loop", "0", "*.png", "skill_"+str(skill)+".gif"], cwd=dest_folder)
    p.wait()

max_steps = 20
for skill in range(num_skills):
    print(skill)
    obs = env.reset()
    cum_reward = 0
    frames = []
    
    # policy.eval()
    z_one_hot = np.zeros(num_skills)
    z_one_hot[skill] = 1.
    z_one_hot = torch.FloatTensor(z_one_hot).to(device).unsqueeze(0)
    for t in range(max_steps):
        
        # Render into buffer. 
        frames.append(env.render())
        

        obs = torch.FloatTensor(obs).to(device).unsqueeze(0)
        _, _, action = policy.sample(obs, z_one_hot) # evaluate is true
        action = action.detach().cpu().numpy()[0]
        obs, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    print(done, t)
    print(cum_reward)
    save_images(frames, skill)


env.close()
