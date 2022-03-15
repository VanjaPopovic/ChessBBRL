import torch
import numpy as np

from behaviour_gym.utils import quaternion as q
from behaviour_gym.utils import transforms as t


def step_fn_trained_rn(env, action):
    if env._current_behaviour == "approach":
        while not env.has_approached and env._step < env.max_steps:
            env.reactive_net(
                torch.from_numpy(env.obs).type(torch.FloatTensor).to(env.device)
            )
            behaviour_action = env.reactive_net.get_behaviour_output(action)
            curr_pos, curr_orn = env.robot.getPose()
            behaviour_action = np.clip(behaviour_action.cpu().detach().numpy(), -1, 1)
            pos_actions = behaviour_action[:3] * env._max_move

            goal_pos = np.add(curr_pos, pos_actions)
            goal_orn = curr_orn

            # Interpolate poses
            poses = t.interpolate(curr_pos, curr_orn, goal_pos, goal_orn, 2)

            # Step through poses
            for i in range(2):
                # Calculate IK solution
                env.robot.applyPose(poses[i][0], poses[i][1], relative=True)

                # Step through scene 1/8 of a second
                env.scene.step(30, env.timestep)

            # Step through scene for 1/12 of a second to reach final pose
            env.scene.step(20)

            env.obs = env._get_obs()
            env.approached()
            env._step += 1
        if env.has_approached:
            env._current_behaviour = "grasp"
        if env._step >= env.max_steps:
            reward = -1
        else:
            reward = 1
        return env.obs, reward, env._is_done(), env._get_info()
    elif env._current_behaviour == "grasp":
        while np.linalg.norm(env.obs[26:29]) > 0.01 and env._step < env.max_steps:
            env.reactive_net(
                torch.from_numpy(env.obs).type(torch.FloatTensor).to(env.device)
            )
            behaviour_action = env.reactive_net.get_behaviour_output(action)
            curr_pos, curr_orn = env.robot.getPose()
            behaviour_action = np.clip(behaviour_action.cpu().detach().numpy(), -1, 1)
            pos_actions = behaviour_action[:3] * env._max_move

            goal_pos = np.add(curr_pos, pos_actions)
            goal_orn = curr_orn

            # Interpolate poses
            poses = t.interpolate(curr_pos, curr_orn, goal_pos, goal_orn, 2)

            # Step through poses
            for i in range(2):
                # Calculate IK solution
                env.robot.applyPose(poses[i][0], poses[i][1], relative=True)

                # Step through scene 1/8 of a second
                env.scene.step(30, env.timestep)

            # Step through scene for 1/12 of a second to reach final pose
            env.scene.step(20)
            env.obs = env._get_obs()
            env._step += 1
        while not env.has_grasped and env._step < env.max_steps:
            env.robot.applyGripAction([1])
            env.scene.step(20)
            env.grasped()
            env._step += 1
        env.obs = env._get_obs()
        if env._step >= env.max_steps:
            reward = -1
        else:
            reward = 1
        if env.has_grasped:
            env._current_behaviour = "retract"
        # return env.obs, reward, env._is_done(), env._get_info()
    elif env._current_behaviour == "retract":
        while not env._is_success() and env._step < env.max_steps:
            env.reactive_net(
                torch.from_numpy(env.obs).type(torch.FloatTensor).to(env.device)
            )
            behaviour_action = env.reactive_net.get_behaviour_output(action)
            curr_pos, curr_orn = env.robot.getPose()
            behaviour_action = np.clip(behaviour_action.cpu().detach().numpy(), -1, 1)
            pos_actions = behaviour_action[:3] * env._max_move

            goal_pos = np.add(curr_pos, pos_actions)
            goal_orn = curr_orn

            # Interpolate poses
            poses = t.interpolate(curr_pos, curr_orn, goal_pos, goal_orn, 2)

            # Step through poses
            for i in range(2):
                # Calculate IK solution
                env.robot.applyPose(poses[i][0], poses[i][1], relative=True)

                # Step through scene 1/8 of a second
                env.scene.step(30, env.timestep)

            # Step through scene for 1/12 of a second to reach final pose
            env.scene.step(20)

            # env.obs = env._get_obs()
            env._step += 1
        if env._step >= env.max_steps:
            reward = -1
        else:
            reward = 1
        return env.obs, reward, env._is_done(), env._get_info()


def step_fn_expert_behaviours(env, action):
    if action == 0 and not env.has_approached:
        robPos, robOrn = env.robot.getPose()
        goalOrn = env.robot.restOrn
        blockPos = env.scene.getTarget()

        print("BLOCK POS ", blockPos)
        robPos = np.array(robPos)
        if np.linalg.norm(robPos - blockPos) < 0.02:
            return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()

        goalPos = np.array(robPos).copy()
        goalPos[2] = 1.0
        poses = t.interpolate(robPos, robOrn, goalPos, robOrn, 5)
        for pose in poses:
            env.robot.applyPose(*pose, relative=True)
            env.scene.step(20)
        env.robot.applyPose(goalPos, robOrn)
        env.scene.step(50)

        robPos, robOrn = env.robot.getPose()
        blockPos = env.scene.getTarget()
        goalPos = np.add(blockPos, [0, 0, 0.3])
        poses = t.interpolate(robPos, robOrn, goalPos, goalOrn, 30)
        for pose in poses:
            env.robot.applyPose(*pose, relative=True)
            env.scene.step(20)
        env.robot.applyPose(goalPos, goalOrn, relative=True)
        env.scene.step(50)
        env.robot.applyPose(goalPos, goalOrn, relative=True)
        env.scene.step(50)

        robPos, robOrn = env.robot.getPose()
        blockPos = env.scene.getTarget()
        goalPos = np.add(blockPos, [0, 0, 0.03])
        poses = t.interpolate(robPos, robOrn, goalPos, goalOrn, 4)
        env.robot.applyPose(goalPos, goalOrn, relative=True)
        for pose in poses:
            env.robot.applyPose(*pose, relative=True)
            env.scene.step(20)
        env.robot.applyPose(goalPos, goalOrn, relative=True)
        env.scene.step(50)
        print("Approaching")
        print(blockPos)
        print(robPos)
    elif action == 1 and not env.has_grasped:
        # Grasp
        print("Grasp")
        robPos, robOrn = env.robot.getPose()
        blockPos = env.scene.getTarget()
        robPos = np.array(robPos)
        blockPos = np.array(blockPos)
        # if env.robot.getGripState()[0] > 0.6:
        #     return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()
        # if np.linalg.norm(robPos - blockPos) > 0.02:
        #     return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()

        env.robot.applyGripAction([1])
        env.scene.step(20)
        env.robot.applyGripAction([1])
        env.scene.step(20)
        env.robot.applyGripAction([1])
        env.scene.step(20)
        env.robot.applyGripAction([1])
        env.scene.step(20)
        env.robot.applyGripAction([1])
        env.scene.step(20)
        env.robot.applyGripAction([1])
        env.scene.step(20)
        env.robot.applyGripAction([0.5])
        env.scene.step(20)

        robPos, robOrn = env.robot.getPose()
        goalPos = np.add(robPos, [0, 0, 0.05])
        env.robot.applyPose(goalPos, robOrn, relative=True)
        env.scene.step(50)
    elif action == 2 and env.has_grasped:
        # Place
        print("Retracting")
        robPos, robOrn = env.robot.getPose()
        robPos = np.array(robPos)
        goalPos = np.add(env.scene.getDestTarget(), [0, 0, 0.05])
        # print("TARGET")
        # print(env.scene.getTarget())
        # print("DEST TARGET")
        # print(env.scene.getDestTarget())
        if np.linalg.norm(robPos - goalPos) < 0.03:
            return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()
        poses = t.interpolate(robPos, robOrn, goalPos, robOrn, 30)
        for pose in poses:
            env.robot.applyPose(*pose, relative=True)
            env.scene.step(20)
        env.robot.applyPose(goalPos, robOrn)
        env.scene.step(50)
    elif action == 3 and env.has_retracted:
         # Grasp
        print("Placing")
        robPos, robOrn = env.robot.getPose()
        blockPos = env.scene.getTarget()
        robPos = np.array(robPos)
        blockPos = np.array(blockPos)
        # if env.robot.getGripState()[0] > 0.6:
        #     return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()
        # if np.linalg.norm(robPos - blockPos) > 0.02:
        #     return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()

        env.robot.applyGripAction([0])
        env.scene.step(20)
        env.robot.applyGripAction([0])
        env.scene.step(20)
        env.robot.applyGripAction([0])
        env.scene.step(20)
        env.robot.applyGripAction([0])
        env.scene.step(20)
        env.robot.applyGripAction([0])
        env.scene.step(20)
        env.robot.applyGripAction([0])
        env.scene.step(20)
        env.robot.applyGripAction([0])
        env.scene.step(20)

        # robPos, robOrn = env.robot.getPose()
        # goalPos = np.add(robPos, [0, 0, 0.01])
        # env.robot.applyPose(goalPos, robOrn, relative=True)
        # env.scene.step(50)
    env.approached()
    env.grasped()
    env.retracted()
    env.placed()
    if env.has_approached:
        env._current_behaviour = "grasp"
    if env.has_grasped:
        env._current_behaviour = "retract"
    if env.has_retracted:
        env._current_behaviour = "place"
    return env._get_obs(), env._get_reward(), env._is_done(), env._get_info()


def step_fn_rn_execute_single_action(env, action):

    if (
        (action == 0 and env._current_behaviour == "approach")
        or (action == 1 and env._current_behaviour == "grasp")
        or (action == 2 and env._current_behaviour == "retract")
    ):
        reward = 1
    else:
        reward = -1

    env.reactive_net(torch.from_numpy(env.obs).type(torch.FloatTensor).to(env.device))
    behaviour_action = env.reactive_net.get_behaviour_output(action)
    curr_pos, curr_orn = env.robot.getPose()
    behaviour_action = np.clip(behaviour_action.cpu().detach().numpy(), -1, 1)
    pos_actions = behaviour_action[:3] * env._max_move

    goal_pos = np.add(curr_pos, pos_actions)
    goal_orn = curr_orn

    # Interpolate poses
    poses = t.interpolate(curr_pos, curr_orn, goal_pos, goal_orn, 2)

    # Step through poses
    for i in range(2):
        # Calculate IK solution
        env.robot.applyPose(poses[i][0], poses[i][1], relative=True)

        # Step through scene 1/8 of a second
        env.scene.step(30, env.timestep)

    # Step through scene for 1/12 of a second to reach final pose
    env.scene.step(20)

    env.obs = env._get_obs()
    env.approached()
    if env.has_approached:
        env._current_behaviour = "grasp"

    if np.linalg.norm(env.obs[26:29]) < 0.01:
        attempts = 0
        while not env.has_grasped:
            env.robot.applyGripAction([1])
            env.scene.step(20)
            env.grasped()
            attempts += 1
            if attempts == 30:
                env._step = 200
                break

    env.obs = env._get_obs()
    if env.has_grasped:
        env._current_behaviour = "retract"
    return env.obs, reward, env._is_done(), env._get_info()


def step_fn_low_level_behaviours(env, action):
    curr_pos, curr_orn = env.robot.getPose()
    behaviour_action = np.clip(action, -1, 1)
    pos_actions = behaviour_action[:3] * env._max_move
    # gripper_rot = None
    # if action == 1:
    #     gripper_rot = behaviour_action[3]

    goal_pos = np.add(curr_pos, pos_actions)
    goal_orn = curr_orn
    # if gripper_rot is not None:
    #     # gripper_rot *= env._max_rot
    #     goal_orn = q.rotateGlobal(curr_orn, 0, 0, gripper_rot)

    # Interpolate poses
    poses = t.interpolate(curr_pos, curr_orn, goal_pos, goal_orn, 2)

    # Step through poses
    for i in range(2):
        # Calculate IK solution
        env.robot.applyPose(poses[i][0], poses[i][1], relative=True)

        # Step through scene 1/8 of a second
        env.scene.step(30, env.timestep)

    # Step through scene for 1/12 of a second to reach final pose
    env.scene.step(20)
    env.obs = env._get_obs()

    env.approached()
    if env.has_approached:
        env.grasped()
    if env.has_grasped:
        env.retracted()
    if env.has_retracted:
        env.placed()

    return env.obs, env._get_reward(), env._is_done(), env._get_info()
