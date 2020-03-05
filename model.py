import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)




class QNetwork(nn.Module):
    def __init__(self,  width, height, num_skills, num_actions, hidden_dim):
        super(QNetwork, self).__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(), # 64x64
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(), # 32x32
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(), # 16x16
            nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1, bias=False) # 8x8
            
        )

        fc_num_inputs = int(width/2**4)*int(height/2**4)*int(4) + num_skills

        # Q1 architecture
        self.linear1 = nn.Linear(fc_num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(fc_num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, observation, skill, action):

        state = self.gate(observation)
        state = state.view(state.shape[0], -1)

        xu = torch.cat([state, skill, action], 1)
        
        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)

        return x1, x2


class GaussianPolicy(nn.Module):
    def __init__(self, width, height, num_skills, num_actions, hidden_dim, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.gate = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(), # 64x64
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(), # 32x32
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(), # 16x16
            nn.Conv2d(16, 4, kernel_size=3, stride=2, padding=1, bias=False) # 8x8
            
        )
        fc_num_inputs = int(width/2**4)*int(height/2**4)*int(4) + num_skills
        self.linear1 = nn.Linear(fc_num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

        self.flat_state = None

    def forward(self, observation, skill):
        
        bs = observation.shape[0]
        state = self.gate(observation)
        state = state.view(bs, -1)
        self.flat_state = state.detach()
        state = torch.cat([state, skill], 1)

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, observation, skill):
        mean, log_std = self.forward(observation, skill)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)

class Discriminator(nn.Module):
    def __init__(self, num_inputs, num_skills, hidden_size, init_w=3e-3):
        super(Discriminator, self).__init__()
         
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        
        self.linear3 = nn.Linear(hidden_size, num_skills)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
        
    def forward(self, state):
        if len(state.shape) > 2:
            state = state.view(state.shape[0], -1)
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        
        return x


# TEST
# num_skills = 8
# num_actions = 2
# hidden_size = 64
# width, height = 128, 128
# net = GaussianPolicy(width, height, num_skills, num_actions, hidden_size).cuda()
# obs = torch.rand((1,1,width, height)).cuda()
# skill = torch.tensor([[1,0,0,0, 0,0,0,0]], dtype=torch.float32).cuda()



# y = net(obs, skill)
# value = net.flat_state

# D = Discriminator(256, num_skills, hidden_size).cuda()
# d = D(value)

# Q = QNetwork(width, height, num_skills, num_actions, hidden_size).cuda()
# action = torch.rand(1,num_actions).cuda()
# q = Q(obs, skill, action)