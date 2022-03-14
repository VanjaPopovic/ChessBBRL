import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.manual_seed(1)
    from models import ReactiveNet, FeatureNet, Choreographer
    from bullet_env import PickAndPlace, PickPlaceScene
    from bullet_env.utils import *

    feature_net = FeatureNet(33)
    feature_net.load_state_dict(torch.load("./weights/fe_network.pth"))
    reactive_net = ReactiveNet(feature_net.fc2.out_features, './weights', feature_net=feature_net)
    reactive_net.load_model()
    for param in reactive_net.parameters():
        param.requires_grad = False
    reactive_net.to(device)
    reactive_net.feature_net.fc1.register_forward_hook(get_activation("fc1"))
    reactive_net.feature_net.fc2.register_forward_hook(get_activation("fc2"))
    scene = PickPlaceScene(guiMode=True)
    robot = scene.loadRobot()
    env = PickAndPlace(scene, robot, step_fn_rn_execute_single_action, reactive_net=reactive_net, device=device)
    num_action_to_str = ["approach", "grasp", "retract", "place"]
    episodes = 2500
    acts_data = {}
    success = 0
    i = 0
    with torch.no_grad():
        while i < episodes:
            print("Episode {}".format(i + 1))
            obs = env.reset()
            acts_data[i] = {"data": [], "labels": []}
            timestep = 0

            object_rel_pos_arm = obs[26:29].copy()
            object_rel_pos_arm[2] += 0.05
            while np.linalg.norm(object_rel_pos_arm) > 0.03 and timestep <= env.max_steps:
                model_input = torch.from_numpy(obs).type(torch.FloatTensor).to(device)
                reactive_net(model_input)
                behaviour_action = reactive_net.get_behaviour_output(0)

                action = np.array([behaviour_action[0].item(), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
                activations_tensors = []
                for key in ["fc1", "fc2"]:
                    activations_tensor = activations[key]
                    activations_tensors.append(activations_tensor.cpu().detach())
                activations_tensors = torch.cat(activations_tensors, dim=0)
                acts_data[i]["data"].append(activations_tensors)
                acts_data[i]["labels"].append(num_action_to_str[0])

                obs, reward, done, info = env.step(0)
                object_rel_pos_arm = obs[26:29].copy()
                object_rel_pos_arm[2] += 0.05
                timestep += 1

            object_rel_pos_arm = obs[26:29].copy()
            while np.linalg.norm(object_rel_pos_arm) > 0.01 and timestep <= env.max_steps:
                model_input = torch.from_numpy(obs).type(torch.FloatTensor).to(device)
                reactive_net(model_input)
                behaviour_action = reactive_net.get_behaviour_output(1)
                action = np.array([behaviour_action[0].item(), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
                obs, reward, done, info = env.step(1)
                object_rel_pos_arm = obs[26:29].copy()
                timestep += 1

            while not env.has_grasped and timestep <= env.max_steps:
                env.robot.applyGripAction([1])
                env.scene.step(20)
                env.grasped()
                timestep += 1
            
            object_rel_pos_target = obs[29:32] - obs[13:16]
            while np.linalg.norm(object_rel_pos_target) > 0.03 and timestep <= env.max_steps:
                model_input = torch.from_numpy(obs).type(torch.FloatTensor).to(device)
                reactive_net(model_input)
                behaviour_action = reactive_net.get_behaviour_output(2)
                action = np.array([behaviour_action[0].item(), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
                obs, reward, done, info = env.step(2)
                object_rel_pos_target = obs[29:32] - obs[13:16]
                timestep += 1
            
            final_target = obs[29:32].copy()
            final_target[2] = 0.64
            final_target = final_target - obs[13:16]
            while np.linalg.norm(final_target) > 0.02 and timestep <= env.max_steps:
                model_input = torch.from_numpy(obs).type(torch.FloatTensor).to(device)
                reactive_net(model_input)
                behaviour_action = reactive_net.get_behaviour_output(3)
                action = np.array([behaviour_action[0].item(), behaviour_action[1].item(), behaviour_action[2].item(), 0, -1])
                obs, reward, done, info = env.step(3)
                final_target = obs[29:32].copy()
                final_target[2] = 0.64
                final_target = final_target - obs[13:16]
                timestep += 1
            while not env.has_placed and timestep < env.max_steps:
                env.robot.applyGripAction([0])
                env.scene.step(20)
                env.placed()
                print(env.has_placed)
                timestep += 1

            if info["is_success"]:
                success += 1
                print("success")
            i += 1
    print(success)
    torch.save(acts_data, "ppo_choreographer_rn_fe_acts.pt")