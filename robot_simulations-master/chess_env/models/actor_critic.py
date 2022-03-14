import torch.nn as nn
import torch.nn.functional as F

class ActorCriticLSTM(nn.Module):

    """
        Neural network that implements choreographer
    """

    def __init__(self, input_size, weights_init_fn):
        super(ActorCriticLSTM, self).__init__()

        self.input_size = input_size
        self.lstm = nn.LSTMCell(self.input_size, 32)
        self.critic_linear = nn.Linear(32, 1)
        self.actor_linear = nn.Linear(32, 4)

        self.apply(weights_init_fn)

    def forward(self, x, hx, cx):

        # Feedforward features to get behaviour and state value.
        lstm_input = x.view(-1, self.input_size)
        hx, cx = self.lstm(lstm_input, (hx, cx))
        state_value = self.critic_linear(hx)
        actions = F.softmax(self.actor_linear(hx), dim=1)
        # actions = self.actor_linear(hx)
        output = {}
        output["state"] = state_value
        output["actions"] = actions
        output["hidden"] = hx
        output["cell"] = cx
        
        return output

class ActorCritic(nn.Module):

    """
        Neural network that implements choreographer
    """

    def __init__(self, input_size, weights_init_fn):
        super(ActorCritic, self).__init__()

        # self.fc1 = nn.Linear(input_size, 64)
        # self.fc2 = nn.Linear(64, 20)
        self.critic_linear = nn.Linear(input_size, 1)
        self.actor_linear = nn.Linear(input_size, 4)

        self.apply(weights_init_fn)

    def forward(self, x):

        # Feedforward features to get behaviour and state value.
        # x = F.elu(self.fc1(x))
        # x = F.elu(self.fc2(x))
        state_value = self.critic_linear(x)
        actions = F.softmax(self.actor_linear(x), dim=-1)

        output = {}
        output["state"] = state_value
        output["actions"] = actions

        return output