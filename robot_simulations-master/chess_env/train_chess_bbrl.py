import os
import math
import random
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from stockfish import Stockfish
import chess

from torch.utils.tensorboard import SummaryWriter

BEHAVIOUR_EPISODES = {"approach": 2500, "grasp": 2000,
                      "retract": 3000, "choreograph": 3500, "place": 3000}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behaviour Based Reinforcement Learning")
    parser.add_argument("command", metavar="<command>",
                        help="[approach|grasp|retract|place|choreograph]")
    parser.add_argument("--max-grad-norm", type=float, default=250,
                        help="Max gradient norm")
    parser.add_argument("--save-freq", type=int, default=50,
                        help="Model save interval")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor for rewards")
    parser.add_argument("--tau", type=float, default=1.00,
                        help="Parameter for GAE")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy term coefficient")
    parser.add_argument("--value-loss-coef", type=float, default=0.5,
                        help="Value loss coefficient")
    parser.add_argument('--epsilon', type=float, default=0.2,
                        help='parameter for Clipped Surrogate Objective')
    parser.add_argument("--weights-path", help="Path to weights folder", default="./")
    parser.add_argument("--seed", help="RNG seed")
    args = parser.parse_args()

    device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    writer = SummaryWriter('experiments/behaviours/{}'.format(args.command))
    from models import ReactiveNet, FeatureNet, Choreographer
    from environment import PickAndPlace
    from pick_place import PickPlaceScene
    from utils import *

    if os.name == 'nt':
        stockfish = Stockfish(
            r'C:\Users\fream\Downloads\robot_simulations-master\robot_simulations-master\chess_env\stockfish2.exe')
    else:
        stockfish = Stockfish(
            os.path.abspath("/home/pitsill0s/Desktop/ChessBBRL/robot_simulations-master/chess_env/stockfish_14.1_linux_x64")
        )
            
    scene = PickPlaceScene(guiMode=True, isTraining=True)
    robot = scene.loadRobot()
    env = PickAndPlace(scene, robot, step_fn=step_fn_low_level_behaviours)

    feature_net = FeatureNet(env.observation_space.shape[0])
    feature_net.to(device)
    if args.command != "approach":
        feature_net.load_state_dict(torch.load(
            os.path.join(args.weights_path, "fe_network.pth")))
        feature_net.freeze()
        feature_net.eval()

    behaviour_net = ReactiveNet(feature_net.fc2.out_features,
                                args.weights_path, args.command, feature_net=feature_net)
    optimizer = optim.Adam(behaviour_net.parameters(), lr=args.lr)
    behaviour_net_criterion = nn.MSELoss()
    behaviour_net.train()
    behaviour_net.to(device)

    if args.command == "choreograph":
        choreograph_feature_net = FeatureNet(env.observation_space.shape[0])
        choreograph_feature_net.load_state_dict(torch.load(
            os.path.join(args.weights_path, "fe_network.pth")))
        choreograph_feature_net.freeze()
        choreograph_feature_net.eval()
        choreographer = Choreographer(choreograph_feature_net.fc2.out_features,
                                      args.weights_path, feature_net=choreograph_feature_net, internal_states=False)
        optimizer = optim.Adam(
            choreographer.actor_critic.parameters(), lr=args.lr)
        choreographer.train()
        choreographer.to(device)
        behaviour_net.load_model()
        behaviour_net.eval()

    i = 1
    success = 0
    success_per_100 = 0
    done = True
    while i <= BEHAVIOUR_EPISODES[args.command]:
        print("Episode {}".format(i))
        obs = env.reset()
        timestep = 0

        if args.command == "choreograph":

            if done:
                cx = torch.zeros(1, 32, requires_grad=True).type(
                    torch.FloatTensor).to(device)
                hx = torch.zeros(1, 32, requires_grad=True).type(
                    torch.FloatTensor).to(device)
            else:
                cx = cx.clone().detach().requires_grad_(True).to(device)
                hx = hx.clone().detach().requires_grad_(True).to(device)
            values, log_probs, rewards, entropies = [], [], [], []
            choreographer_input = torch.from_numpy(
                obs).type(torch.FloatTensor).to(device)
            choreographer_output = choreographer(choreographer_input)
            state_value = choreographer_output["state"]
            action_probs = choreographer_output["actions"].squeeze()
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_out = action.item()
            entropies.append(dist.entropy()), log_probs.append(
                log_prob.item()), values.append(torch.squeeze(state_value))
            # writer.add_scalar("approach_value", torch.squeeze(state_value), i)
        else:
            action_out = 0

        # Approach solution here is the object relative position measured with (object_pos - arm_pos)
        object_rel_pos_arm = obs[26:29].copy()
        object_rel_pos_arm[2] += 0.05
        while np.linalg.norm(object_rel_pos_arm) > 0.03 and timestep <= env.max_steps:
            model_input = torch.from_numpy(obs).type(
                torch.FloatTensor).to(device)
            behaviour_net(model_input)
            behaviour_action = behaviour_net.get_behaviour_output(action_out)
            expected = torch.from_numpy(
                object_rel_pos_arm).type(torch.FloatTensor)

            if args.command == "approach":
                error = torch.zeros(3, dtype=torch.float32).to(device)
                optimizer.zero_grad()
                for j in range(3):
                    error[j] = behaviour_net_criterion(
                        behaviour_action[j], expected[j])
                loss = torch.sum(error).type(torch.FloatTensor).to(device)
                loss.backward()
                writer.add_scalar("loss", loss.cpu().detach().item(), i)
                optimizer.step()

            if args.command == "choreograph" or args.command == "approach":
                action = np.array([behaviour_action[0].item(
                ), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
            else:
                action = np.array(
                    [expected[0].item(), expected[1].item(), expected[2].item(), 0, -1])
            obs, reward, done, info = env.step(action)
            object_rel_pos_arm = obs[26:29].copy()
            object_rel_pos_arm[2] += 0.05
            timestep += 1

        # If only training approach continue from here.
        if args.command == "approach":
            i += 1
            if i % 100 == 0 and i != 0:
                torch.save(feature_net.state_dict(), "fe_network.pth")
                behaviour_net.save_model()
            if env.has_approached:
                success += 1
                print("Success")
            continue

        if args.command == "choreograph":
            if timestep < env.max_steps:
                rewards.append(torch.tensor([1.0]).to(device))
            else:
                rewards.append(torch.tensor([-1.0]).to(device))
            choreographer_input = torch.from_numpy(
                obs).type(torch.FloatTensor).to(device)
            choreographer_output = choreographer(choreographer_input)
            state_value = choreographer_output["state"]
            action_probs = choreographer_output["actions"].squeeze()
            print("Action probs on grasp: {}".format(
                action_probs.cpu().detach().numpy()))
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_out = action.item()
            print(action_out)
            entropies.append(dist.entropy()), log_probs.append(
                log_prob.item()), values.append(torch.squeeze(state_value))
            writer.add_scalar("grasp_value", torch.squeeze(state_value), i)
        else:
            action_out = 1

        object_rel_pos_arm = obs[26:29].copy()
        object_rel_pos_arm[2] += 0.02
        while np.linalg.norm(object_rel_pos_arm) > 0.01 and timestep <= env.max_steps:
            model_input = torch.from_numpy(obs).type(
                torch.FloatTensor).to(device)
            behaviour_net(model_input)
            behaviour_action = behaviour_net.get_behaviour_output(action_out)
            # Expected value of how much to close gripper found by trial and error
            # by twicking the gripRestPos in the robot constructor
            expected_np = np.array(
                [object_rel_pos_arm[0], object_rel_pos_arm[1], object_rel_pos_arm[2], 0, -1])
            expected = torch.from_numpy(expected_np).type(torch.FloatTensor)

            if args.command == "grasp":
                error = torch.zeros(3, dtype=torch.float32).to(device)
                optimizer.zero_grad()
                for j in range(3):
                    error[j] = behaviour_net_criterion(
                        behaviour_action[j], expected[j])
                loss = torch.sum(error[:3]).type(torch.FloatTensor).to(device)
                loss.backward()
                writer.add_scalar("loss", loss.cpu().detach().item(), i)
                optimizer.step()

            if args.command in ["choreograph", "grasp"]:
                if action_out == 1:
                    action = np.array([behaviour_action[0].item(
                    ), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
                else:
                    action = expected_np
            obs, reward, done, info = env.step(action)
            object_rel_pos_arm = obs[26:29].copy()
            timestep += 1

        while not env.has_grasped and timestep < env.max_steps:
            env.robot.applyGripAction([1])
            env.scene.step(20)
            env.grasped()
            timestep += 1
        obs = env._get_obs()
        if args.command == "grasp":
            i += 1
            if i % 100 == 0 and i != 0:
                behaviour_net.save_model()
            if env.has_grasped:
                success += 1
                print("Success")
            continue
        obs = env._get_obs()
        if args.command == "choreograph":
            if timestep < env.max_steps:
                rewards.append(torch.tensor([1.0]).to(device))
            else:
                rewards.append(torch.tensor([-1.0]).to(device))
            choreographer_input = torch.from_numpy(
                obs).type(torch.FloatTensor).to(device)
            choreographer_output = choreographer(choreographer_input)
            state_value = choreographer_output["state"]
            action_probs = choreographer_output["actions"].squeeze()
            print("Action probs on retract: {}".format(
                action_probs.cpu().detach().numpy()))
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_out = action.item()
            print(action_out)
            entropies.append(dist.entropy()), log_probs.append(
                log_prob.item()), values.append(torch.squeeze(state_value))
            writer.add_scalar("retract_value", torch.squeeze(state_value), i)
        else:
            action_out = 2
        tar = obs[32:35].copy()
        #print("GETTING TAR",tar)
        tar[2] = 0.7
        object_rel_pos_target = tar - obs[0:3]
        # print(obs)
        while np.linalg.norm(object_rel_pos_target) > 0.03 and timestep <= env.max_steps:
            model_input = torch.from_numpy(obs).type(
                torch.FloatTensor).to(device)
            behaviour_net(model_input)
            behaviour_action = behaviour_net.get_behaviour_output(action_out)
            expected = torch.from_numpy(
                object_rel_pos_target).type(torch.FloatTensor)
            if args.command == "retract":
                error = torch.zeros(3, dtype=torch.float32).to(device)
                optimizer.zero_grad()
                for j in range(3):
                    error[j] = behaviour_net_criterion(
                        behaviour_action[j], expected[j])
                loss = torch.sum(error)
                loss.backward()
                writer.add_scalar("loss", loss.cpu().detach().item(), i)
                optimizer.step()

            if args.command == "choreograph" or args.command == "retract":
                action = np.array([behaviour_action[0].item(
                ), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
            else:
                action = np.array(
                    [expected[0].item(), expected[1].item(), expected[2].item(), 0, -1])
            obs, reward, done, info = env.step(action)
            tar = obs[32:35].copy()
            #print("GETTING TAR",tar)
            tar[2] = 0.7
            object_rel_pos_target = tar - obs[0:3]
            timestep += 1

        obs = env._get_obs()
        if args.command == "retract":
            i += 1
            if i % 100 == 0 and i != 0:
                behaviour_net.save_model()
            if env.has_retracted:
                success += 1
                print("Success")
            continue

        if args.command == "choreograph":
            if timestep < env.max_steps:
                rewards.append(torch.tensor([1.0]).to(device))
            else:
                rewards.append(torch.tensor([-1.0]).to(device))
            choreographer_input = torch.from_numpy(
                obs).type(torch.FloatTensor).to(device)
            choreographer_output = choreographer(choreographer_input)
            state_value = choreographer_output["state"]
            action_probs = choreographer_output["actions"].squeeze()
            print("Action probs on place: {}".format(
                action_probs.cpu().detach().numpy()))
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_out = action.item()
            print(action_out)
            entropies.append(dist.entropy()), log_probs.append(
                log_prob.item()), values.append(torch.squeeze(state_value))
            writer.add_scalar("place_value", torch.squeeze(state_value), i)
        else:
            action_out = 3

        # replace this with Final action?
        final_target = obs[29:32].copy()
        final_target[2] = 0.64
        final_target = final_target - obs[13:16]
        while np.linalg.norm(final_target) > 0.02 and timestep <= env.max_steps:
            model_input = torch.from_numpy(obs).type(
                torch.FloatTensor).to(device)
            behaviour_net(model_input)
            behaviour_action = behaviour_net.get_behaviour_output(action_out)
            expected_np = np.array(
                [final_target[0], final_target[1], final_target[2], 0, -1])
            expected = torch.from_numpy(expected_np).type(torch.FloatTensor)

            if args.command == "place":
                error = torch.zeros(3, dtype=torch.float32).to(device)
                optimizer.zero_grad()
                for j in range(3):
                    error[j] = behaviour_net_criterion(
                        behaviour_action[j], expected[j])
                loss = torch.sum(error[:3]).type(torch.FloatTensor).to(device)
                loss.backward()
                writer.add_scalar("loss", loss.cpu().detach().item(), i)
                optimizer.step()

            if args.command in ["choreograph", "place"]:
                if action_out == 3:
                    action = np.array([behaviour_action[0].item(
                    ), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
                else:
                    action = expected_np
            obs, reward, done, info = env.step(action)
            final_target = obs[29:32].copy()
            final_target[2] = 0.64
            final_target = final_target - obs[13:16]
            timestep += 1

        obs = env._get_obs()
        while not env.has_placed and timestep < env.max_steps:
            env.robot.applyGripAction([0])
            env.scene.step(20)
            env.placed()
            timestep += 1
        obs = env._get_obs()
        if args.command == "place":
            i += 1
            if i % 100 == 0 and i != 0:
                behaviour_net.save_model()
            if env.has_placed:
                success += 1
                print("Success")
            continue

        obs = env._get_obs()
        if info["is_success"]:
            success += 1
            success_per_100 += 1
            print("Dis Success")

        if i % 100 == 0:
            writer.add_scalar("success", success_per_100/100, i)
            success_per_100 = 0

        if args.command == "choreograph":
            i += 1
            if info["is_success"]:
                rewards.append(torch.tensor([1.0]).to(device))
            else:
                rewards.append(torch.tensor([-1.0]).to(device))
            print("Rewards: {}".format(rewards))
            # A3C loss function
            R = torch.zeros(1, requires_grad=True).to(device)
            values.append(R)
            policy_loss = 0
            value_loss = 0
            gae = torch.zeros(1, requires_grad=True).to(device)
            for k in reversed(range(len(rewards))):
                R = args.gamma * R + rewards[k]
                advantage = R - values[k]
                value_loss = value_loss + 0.5 * advantage.pow(2)

                delta_t = rewards[k] + args.gamma * \
                    values[k + 1] - values[k]
                gae = gae * args.gamma * args.tau + delta_t

                policy_loss = policy_loss - \
                    log_probs[k] * gae

            total_loss = policy_loss + args.value_loss_coef * value_loss
            optimizer.zero_grad()
            total_loss.backward()
            writer.add_scalar(
                "policy_loss", policy_loss.cpu().detach().item(), i)
            writer.add_scalar(
                "value_loss", value_loss.cpu().detach().item(), i)
            writer.add_scalar(
                "total_loss", total_loss.cpu().detach().item(), i)
            torch.nn.utils.clip_grad_norm_(
                choreographer.actor_critic.parameters(), args.max_grad_norm)
            optimizer.step()
    print(success)
    if args.command == "choreograph":
        choreographer.save_model()
