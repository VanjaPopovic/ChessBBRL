# Behaviour Gym

A Python package containing robotic Gym environments implemented using PyBullet. This package is a work in progress and is likely to change greatly over the course of development. I highly recommend creating your own copy.

<p float="left">
    <img src="docs/imgs/base_env.png" height="250">
    <img src="docs/imgs/grasp_env.png" height="250">
    <img src="docs/imgs/grasp.png" height="250">
    <img src="docs/imgs/grasp_lift.png" height="250">
</p>

## Code description

All code can be found in the behaviour_gym package which contains multiple sub-packages:
* camera:
    * Contains the high level Camera abstract class and Monocular sub-class. Provides functions for controlling the camera and getting images.
    * Contains an implementation of a Mynt-D100050 camera as a monocular camera.
* robot:
    * Contains the high level Robot abstract class. Provides functions for controlling the robot and getting sensor feedback.
    * Contains an implementation of a UR10 arm with a robotiq 3-finger gripper.
* scene:
    * Contains the high level Scene abstract class. Provides functions for controlling the simulation and loading in cameras and robots.
    * Contains implementations of an empty scene, a table scene with a single block and table scene with multiple blocks.  
* primitive:
    * Contains the high level Primitive abstract class implementing the OpenAI gym interface.
    * Contains implementations of far reaching, precision reaching and grasping environments.
* utils:
    * Contains various helpful functions for working with quaternions and transformations.

### Setting up a Gym Environment

To set up a Gym Environment first create a Scene then use it to load in Robots and Cameras. Scenes can contain as many Cameras and Robots as desired. Once the scene is ready, use the setInit() method so that the Scene can be easily reloaded via the reset() method. Finally, provide a Primitive implementation with the Scene, a single robot and optionally a camera to combine into a single Gym Environment.

```python
import math

from behaviour_gym.scene import Table
from behaviour_gym.primitive import ReachGoal

# Load the simulation with a GUI
scene = Table(guiMode=True)

# Load a robot with its base 60cm above the origin
robot = scene.loadRobot("ur103f", startPos=[0,0,0.6])

# Load a camera looking at the table
camera = scene.loadCamera("myntd100050", startPos=[0.5, 1.2, 1.8],
                          startPan=math.pi, startTilt=-math.pi/4)

# Now that the scene is set up nicely initialise it
scene.setInit()

# Create a reaching Gym Environment
goalObj = "block" # Reach for the object named block in the scene
timestep = 1./240. # Render physics steps in real time
reachFar = ReachGoal(goalObj, scene=scene, robot=robot,
                     camera=camera, timestep=timestep)
```

Now the environment can be used like any other Gym Environment. A key advantage of seperating the simulation Scene from the Gym Environment is the ability to share a single simulation across multiple Environments. This allows switching between the various skills to execute more complicated tasks such as pick and place. Note that when switching between environments it may be necessary to specify some additional values which would have otherwise been set during the call to reset. See the example below when switching from the far reaching to the precision reaching agent.

```python
from behaviour_gym.primitive import ReachGoalClose

# Create a precision reaching env using the same objects as the far reaching env
reachClose = ReachGoalClose(goalObj, scene=scene, robot=robot,
                            camera=camera, timestep=timestep)

# For sake of the example, lets pretend we have some trained agents!
agentFar = AgentFar()
agentClose = AgentClose()

# Reset the environment
obs = reachFar.reset()

# Step through with the far reaching agent
for i in range(20):
    obs, _, _, _ = reachFar.step(agentFar.computeAction(obs))

# Fake a reset for the precision reaching environment
reachClose.lastObs = reachFar.lastObs
reachClose.prevError = reachFar.prevError
obs = reachClose.getObs()

# Step through with the close reaching agent
for i in range(20):
    obs, _, _, _ = reachClose.step(agentClose.computeAction(obs))
```

### Additional Files

URDFs and meshes can be found in the models folder.

Scripts that can be run, if the optional requirements are met, can be found in the scripts folder.

Examples can be found in the examples folder.

## Installation instructions

### Requirements

You need Python 3.5 or higher to run behaviour_gym in addition to the following
python packages:

* PyBullet 2.7.4 or higher
* NumPy 1.5 or higher
* Gym
* OpenCV

#### Optional Requirements

To use the included scripts (i.e., expert*, run* and train*) you also need:

* tensorflow 1.14
* stable-baselines 2

To use the RLlib scripts, download and install RLlib and PyTorch (or tensorflow and update script accordingly).

### Installation Script

After installing the dependencies, download and install the behaviour_gym package by running the code below:

```
git clone https://github.com/cvas-ug/robot_simulations.git behaviour_gym
cd behaviour_gym
pip install -e .
```
