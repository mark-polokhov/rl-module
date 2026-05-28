import numpy as np

def simple_reward(observation, action=None, reset=False):
    if not hasattr(simple_reward, "prev_progress") or reset:
        simple_reward.prev_progress = 0.0
        simple_reward.prev_action = np.zeros(2)
        simple_reward.stuck_steps = 0
        simple_reward.prev_dist = observation[4]

    lateral_dist = observation[0]
    progress = observation[1]
    heading_diff = observation[2]
    curvature = observation[3]
    dist_to_checkpoint = observation[4]
    direction_dot = observation[5]
    speed = observation[6]

    reward = 0.0

    delta_progress = progress - simple_reward.prev_progress
    simple_reward.prev_progress = progress

    reward += delta_progress * 100.0

    if delta_progress <= 0:
        reward -= 0.5

    # delta_dist = simple_reward.prev_dist - dist_to_checkpoint
    # simple_reward.prev_dist = dist_to_checkpoint
    #
    # reward += delta_dist * 2.0

    if delta_progress < 1e-5:
        simple_reward.stuck_steps += 1
    else:
        simple_reward.stuck_steps = 0

    reward -= min(simple_reward.stuck_steps * 0.3, 5.0)

    if speed > 5.0 and delta_progress < 1e-5:
        reward -= 1.0

    reward += direction_dot * 1.5

    if direction_dot < 0:
        reward += direction_dot * 3.0

    reward -= abs(heading_diff) * 0.5
    reward -= abs(lateral_dist) * 0.8

    if delta_progress > 0:
        reward += min(speed / 20.0, 1.0)

    if speed < 2.0:
        reward -= 0.5

    if abs(curvature) > 0.1 and speed > 12.0:
        reward -= (speed - 12.0) * abs(curvature) * 0.5

    if action is not None:
        action = np.array(action)
        diff = np.linalg.norm(action - simple_reward.prev_action)
        reward -= diff * 0.05
        simple_reward.prev_action = action

    reward += 0.1

    return float(reward)