import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal

from models.vae import LinearEncoder, LinearDecoder, ImageEncoder, ImageDecoder
from models.actor_critic import ActorCritic, ActorCriticLSTM
from models.behaviours import Approach, Grasp, Place, Retract


def init_weights(m):
    class_name = m.__class__.__name__
    if class_name.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight)


LOW_LEVEL_BEHAVIOURS = ["approach", "grasp", "retract", "place"]


class FeatureNet(nn.Module):
    """
        Implements the feature extraction layers.
    """

    def __init__(self, input_size, internal_states=False):

        super(FeatureNet, self).__init__()
        # Feature extraction layers
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.internal_states = internal_states
        self.apply(init_weights)

    def forward(self, x):
        # Feedforward observation input to extract features
        x = F.elu(self.fc1(x))
        output = F.elu(self.fc2(x))

        if self.internal_states:
            return x, output
        return output

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False


class ReactiveNet(nn.Module):

    LOG_SIG_MAX = 20
    LOG_SIG_MIN = -20
    """
        Neural network that implements the reactive network
    """

    def __init__(self, input_size, weights_path, behaviour=None, feature_net=None):
        super(ReactiveNet, self).__init__()

        self.weights_path = weights_path
        self.feature_net = feature_net
        self.behaviour = behaviour
        self.approach = Approach(
            input_size, self.LOG_SIG_MAX, self.LOG_SIG_MIN, init_weights)
        self.grasp = Grasp(input_size, self.LOG_SIG_MAX,
                           self.LOG_SIG_MIN, init_weights)
        self.retract = Retract(input_size, self.LOG_SIG_MAX,
                               self.LOG_SIG_MIN, init_weights)
        self.place = Place(input_size, self.LOG_SIG_MAX,
                           self.LOG_SIG_MIN, init_weights)

        if self.behaviour in LOW_LEVEL_BEHAVIOURS:
            self.freeze_weights()

    def forward(self, x):
        """
            Receives the pro-prioception internal state and outputs actions
        """
        if self.feature_net:
            x = self.feature_net(x)

        approach_means, approach_stds = self.approach(x)
        grasp_means, grasp_stds = self.grasp(x)
        retract_means, retract_stds = self.retract(x)
        place_means, place_stds = self.place(x)

        # self.output = {}
        # self.output["approach_means"] = approach_means
        # self.output["approach_stds"] = approach_stds
        # self.output["grasp_means"] = grasp_means
        # self.output["grasp_stds"] = grasp_stds
        # self.output["retract_means"] = retract_means
        # self.output["retract_stds"] = retract_stds
        self.output = (approach_means, approach_stds, grasp_means,
                       grasp_stds, retract_means, retract_stds, place_means, place_stds)
        return self.output

    def freeze_weights(self):
        # Freezes weights except the ones needed to be trained
        if self.behaviour == LOW_LEVEL_BEHAVIOURS[0]:
            for param in self.grasp.parameters():
                param.requires_grad = False
            for param in self.retract.parameters():
                param.requires_grad = False
            for param in self.place.parameters():
                param.requires_grad = False
        elif self.behaviour == LOW_LEVEL_BEHAVIOURS[1]:
            for param in self.approach.parameters():
                param.requires_grad = False
            for param in self.retract.parameters():
                param.requires_grad = False
            for param in self.place.parameters():
                param.requires_grad = False
        elif self.behaviour == LOW_LEVEL_BEHAVIOURS[2]:
            for param in self.approach.parameters():
                param.requires_grad = False
            for param in self.grasp.parameters():
                param.requires_grad = False
            for param in self.place.parameters():
                param.requires_grad = False
        else:
            for param in self.approach.parameters():
                param.requires_grad = False
            for param in self.grasp.parameters():
                param.requires_grad = False
            for param in self.retract.parameters():
                param.requires_grad = False

    def save_model(self):
        if self.behaviour == LOW_LEVEL_BEHAVIOURS[0]:
            model = self.approach
            weights_name = "approach.pth"
        elif self.behaviour == LOW_LEVEL_BEHAVIOURS[1]:
            model = self.grasp
            weights_name = "grasp.pth"
        elif self.behaviour == LOW_LEVEL_BEHAVIOURS[2]:
            model = self.retract
            weights_name = "retract.pth"
        elif self.behaviour == LOW_LEVEL_BEHAVIOURS[3]:
            model = self.place
            weights_name = "place.pth"
        torch.save(model.state_dict(), os.path.join(
            self.weights_path, weights_name))

    def load_model(self):
        self.approach.load_state_dict(torch.load(
            os.path.join(self.weights_path, "approach.pth")))
        self.grasp.load_state_dict(torch.load(
            os.path.join(self.weights_path, "grasp.pth")))
        self.retract.load_state_dict(torch.load(
            os.path.join(self.weights_path, "retract.pth")))
        self.place.load_state_dict(torch.load(
            os.path.join(self.weights_path, "place.pth")))

    def get_behaviour_output(self, behaviour):
        """
            Receives behaviour from choreographer and returns
            corresponding output
        """

        behaviour_means, behaviour_stds = self.output[behaviour * 2], self.output[(
            behaviour * 2) + 1]
        actions = torch.zeros(len(behaviour_means)).type(torch.FloatTensor)
        for i in range(len(behaviour_means)):
            normal = Normal(behaviour_means[i], behaviour_stds[i])
            action = normal.rsample()
            actions[i] = torch.tanh(action)
        return actions


class Choreographer(nn.Module):

    MODELS = ["means_fe_concat", "means_variance"]
    """
        Neural network that implements choreographer
    """

    def __init__(self, input_size, weights_path, feature_net=None, internal_states=False, **kwargs):
        super(Choreographer, self).__init__()

        self.input_size = input_size
        self.feature_net = feature_net
        self.internal_states = internal_states
        self.weights_path = weights_path
        self.actor_critic = ActorCritic(self.input_size, init_weights)

        if self.internal_states:
            self.model = kwargs["model"]

    def forward(self, x, hx=None, cx=None):

        if not self.internal_states:
            ac_input_features = self.feature_net(x)
        else:
            if self.feature_net:
                first_layer_features, second_layer_features = self.feature_net(
                    x)
                if len(first_layer_features.size()) == 1:
                    vae_input = torch.cat(
                        (first_layer_features, second_layer_features))
                else:
                    vae_input = torch.cat(
                        (first_layer_features, second_layer_features), dim=1)
            else:
                vae_input = x
            z_means, z_logvar = self.vae.encoder(vae_input)

            if self.model == self.MODELS[0]:
                # This is the case of concatenating means to FE output
                ac_input_features = torch.cat((second_layer_features, z_means))
            elif self.model == self.MODELS[1]:
                # This is the case of concatenating means and log variance
                if len(z_means.size()) == 1:
                    ac_input_features = torch.cat((z_means, z_logvar))
                else:
                    ac_input_features = torch.cat((z_means, z_logvar), dim=1)
            else:
                # TODO: add case of new model with vision internal state
                pass
        # output = self.actor_critic(ac_input_features, hx, cx)
        output = self.actor_critic(ac_input_features)

        # Pass internal state to output for error classification
        if self.internal_states:
            output["means"] = z_means
            output["logvar"] = z_logvar
        return output

    def save_model(self, **kwargs):
        if len(kwargs) > 0:
            torch.save(self.state_dict(), "{}_{}_{}.pth".format(
                kwargs["name"], kwargs["timesteps"], kwargs["success"]))
        else:
            torch.save(self.actor_critic.state_dict(), os.path.join(
                self.weights_path, "actor_critic.pth"))

