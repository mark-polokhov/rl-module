import numpy as np

def analyse_reward_unity(observation, action=None, info=None, **kwargs):
    if kwargs.get("reset"):
        analyse_reward_unity.prev_progress = float(observation[1])
        analyse_reward_unity.prev_dist_to_ckpt = float(observation[4])
        analyse_reward_unity.prev_action = np.zeros(2, dtype=np.float32)
        analyse_reward_unity.stuck_steps = 0
        return 0.0

    if not hasattr(analyse_reward_unity, "prev_progress"):
        analyse_reward_unity.prev_progress = float(observation[1])
        analyse_reward_unity.prev_dist_to_ckpt = float(observation[4])
        analyse_reward_unity.prev_action = np.zeros(2, dtype=np.float32)
        analyse_reward_unity.stuck_steps = 0

    # Observation layout (9):
    # 0 lateral_dist, 1 progress, 2 heading_diff(deg), 3 curvature,
    # 4 dist_to_ckpt, 5 direction_dot, 6 speed, 7 wall_contact_streak_sec,
    # 8 moving_backward_flag
    lateral_dist = float(observation[0])
    progress = float(observation[1])
    heading_diff_deg = float(observation[2])
    curvature = float(observation[3])
    dist_to_ckpt = float(observation[4])
    direction_dot = float(observation[5])
    speed = float(observation[6])
    wall_contact_streak = float(observation[7])
    moving_backward_flag = float(observation[8])

    # Step-wise changes are more stable than absolute progress.
    delta_progress = progress - analyse_reward_unity.prev_progress
    delta_dist_to_ckpt = analyse_reward_unity.prev_dist_to_ckpt - dist_to_ckpt
    analyse_reward_unity.prev_progress = progress
    analyse_reward_unity.prev_dist_to_ckpt = dist_to_ckpt

    reward = 0.0

    # 1) Main driver: forward advancement along route.
    reward += np.clip(delta_progress, -0.05, 0.05) * 140.0

    # 2) Encourage getting closer to the next checkpoint.
    reward += np.clip(delta_dist_to_ckpt, -2.0, 2.0) * 0.9

    # 3) Keep car aligned with route direction.
    reward += direction_dot * 1.2

    # 4) Penalize moving away from center line and large heading errors.
    reward -= min(abs(lateral_dist), 3.0) * 0.8
    reward -= abs(heading_diff_deg) / 180.0 * 0.9

    # 5) On curves, discourage excessive speed.
    curvature_factor = min(abs(curvature) / 45.0, 1.0)
    safe_speed = 14.0 - 6.0 * curvature_factor
    if speed > safe_speed:
        reward -= (speed - safe_speed) * 0.35

    # 6) Strong anti-idle penalty:
    # if the car is almost standing still and does not make meaningful progress,
    # punish aggressively and ramp up with consecutive stuck steps.
    almost_no_progress = abs(delta_progress) < 5e-5
    almost_no_ckpt_change = abs(delta_dist_to_ckpt) < 0.01
    is_almost_standing = speed < 0.35

    if almost_no_progress and almost_no_ckpt_change and is_almost_standing:
        analyse_reward_unity.stuck_steps += 1
    else:
        analyse_reward_unity.stuck_steps = 0

    if analyse_reward_unity.stuck_steps > 0:
        # Immediate strong hit, then quickly increasing pressure.
        reward -= 2.5
        reward -= min(analyse_reward_unity.stuck_steps * 0.35, 8.0)

    # Additional punishment for "jitter in place":
    # tiny speed but non-zero controls and no actual movement.
    if analyse_reward_unity.stuck_steps > 3 and action is not None:
        act = np.asarray(action, dtype=np.float32)
        reward -= min(np.linalg.norm(act), 1.5) * 0.8

    # 7) New observations:
    # - moving_backward_flag should be strongly discouraged.
    # - longer wall-contact streak should add increasing penalty.
    if moving_backward_flag > 0.5:
        reward -= 4.0
        if delta_progress <= 0:
            reward -= 2.0

    reward -= min(wall_contact_streak, 2.0) * 0.8

    # 8) Smooth control regularization.
    if action is not None:
        action = np.asarray(action, dtype=np.float32)
        reward -= np.linalg.norm(action - analyse_reward_unity.prev_action) * 0.05
        reward -= float(action[1] ** 2) * 0.05
        analyse_reward_unity.prev_action = action

    # Small step penalty to avoid idling loops.
    reward -= 0.02
    return float(reward)