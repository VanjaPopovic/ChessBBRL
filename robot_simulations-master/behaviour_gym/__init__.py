from behaviour_gym.scene import Table
from gym.envs.registration import register

register(
    id='Reach-v0',
    entry_point='behaviour_gym.primitive.reachEasyWorld:ReachEasyWorld',
    max_episode_steps=50,
    reward_threshold=50.0,
    kwargs={
        "robotName": "Ur103f",
        "startPos": [0.0, 0.0, 0.6],
        "goalObject": "block"

    }
)

register(
    id='Reach-Ur103f-v0',
    entry_point='behaviour_gym.table.reach:Reach',
    max_episode_steps=50,
    reward_threshold=50.0,
    kwargs={
        "robotName": "Ur103f",
        "startPos": [0.0, 0.0, 0.6]
    }
)

register(
    id='Reach-Ur10HandLite-v0',
    entry_point='behaviour_gym.table.reach:Reach',
    max_episode_steps=50,
    reward_threshold=50.0,
    kwargs={
        "robotName": "Ur10HandLite",
        "startPos": [0.0, 0.0, 0.6]
    }
)

register(
    id='Grasp-Ur103f-v0',
    entry_point='behaviour_gym.table.grasp:Grasp',
    max_episode_steps=50,
    reward_threshold=150.0,
    kwargs={
        "robotName": "Ur103f",
        "startPos": [0.0, 0.0, 0.6]
    }
)


register(
    id='Grasp-Ur10HandLite-v0',
    entry_point='behaviour_gym.table.grasp:Grasp',
    max_episode_steps=50,
    reward_threshold=150.0,
    kwargs={
        "robotName": "Ur10HandLite",
        "startPos": [0.0, 0.0, 0.6]
    }
)

register(
    id="PickAndPlace-v0",
    entry_point="behaviour_gym.table.pick_place:PickPlace",
    max_episode_steps=100,
    reward_threshold=150.0,
    kwargs={
        "robotName": "bbrl",
        "startPos": [0.0, 0.0, 0.6]
    }
)
