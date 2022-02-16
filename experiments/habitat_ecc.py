import math
import os
import random
import sys

import git
import imageio
import magnum as mn
import numpy as np
import cv2
# %matplotlib inline
from matplotlib import pyplot as plt
import rusty_neat
from rusty_neat import htm, ecc

# function to display the topdown map
from PIL import Image

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut

# %cd /content/habitat-sim
# @title Define Observation Display Utility Function { display-mode: "form" }

# @markdown A convenient function that displays sensor observations with matplotlib.

# @markdown (double click to see the code)

laplacian = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]])

w, h = 256, 256
l1 = ecc.CpuEccDense.new_in([w, h], [8, 8], [2, 2], 1, 16, 1)
enc = htm.EncoderBuilder()
img_enc = enc.add_image(w, h, 1, 12)


# This tells imageio to use the system FFMPEG that has hardware acceleration.
os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"

test_scene = "../habitat/scene_datasets/habitat-test-scenes/skokloster-castle.glb"

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": w,  # Spatial resolution of the observations
    "height": h,
}


# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors
def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]

    agent_cfg.sensor_specifications = [rgb_sensor_spec]

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)

sim = habitat_sim.Simulator(cfg)

# initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])

# Set agent state
agent_state = habitat_sim.AgentState()
# agent_state.position = np.array([-0.6, 0.0, 0.0])  # in world space
# agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

# obtain the default, discrete actions that an agent can perform
# default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


total_frames = 0
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
fig, axs = plt.subplots(2)
for ax in axs:
    ax.axis("off")
axs[0].set_title("rgb")
axs[1].set_title("input")
for _ in range(5):
    action = random.choice(action_names)
    print("action", action)
    observations = sim.step(action)
    rgb = observations["color_sensor"]
    axs[0].imshow(rgb)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGRA2GRAY)
    result = cv2.filter2D(gray, -1, laplacian)
    sdr = htm.CpuSDR()
    img_enc.encode(sdr,np.expand_dims(result,2))
    axs[1].imshow(result > 12)
    plt.pause(1)
    total_frames += 1
