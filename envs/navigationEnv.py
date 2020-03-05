import gym
import numpy as np
import gym.spaces
from PIL import Image, ImageDraw

class NavigationEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    MAX_STEPS = 100
    HEIGHT = 128
    WIDTH = 128
    N_CHANNELS = 1
    MAX_DISP = 6
    RADII = 5

    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=np.array([-1., -1.]), high=np.array([1., 1.]), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, self.N_CHANNELS), dtype=np.uint8)
        self.reward = 0
        self.done = False
        self._max_episode_steps = self.MAX_STEPS

    def step(self, action):
        #Execute one time step within the environment
        next_y = self.pos_y + int(action[0]*self.MAX_DISP)
        next_x = self.pos_x + int(action[1]*self.MAX_DISP)
        next_y = np.clip(next_y, 0, self.HEIGHT-1)
        next_x = np.clip(next_x, 0, self.WIDTH-1)

        self.pos_y = next_y
        self.pos_x = next_x

        return self._observation(), self.reward, self.done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.pos_y = int(self.HEIGHT/2)
        self.pos_x = int(self.WIDTH/2)

        return self._observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        obs = self._observation(render=True)
        return Image.fromarray(obs)

    def _observation(self, render=False):
        img = Image.new(mode="RGB", size=[self.WIDTH, self.HEIGHT])
        draw = ImageDraw.Draw(img)
        draw.ellipse((self.pos_x - self.RADII, self.pos_y - self.RADII, self.pos_x + self.RADII, self.pos_y + self.RADII), fill = 'blue', outline ='blue')
        img_np = np.array(img)
        obs = img_np[:,:,2]
        if not render:
            obs = obs.astype(np.float32)/255.
            obs = np.expand_dims(obs, axis=0)
        return obs

# env = NavigationEnv()
# obs = env.reset()
# Image.fromarray((obs[0]*255).astype(np.float32)).show()