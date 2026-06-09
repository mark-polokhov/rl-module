def reward_fn(self, obs, action) -> float:
    reward = (
        -1.0 * obs[0] +
        2.0 * obs[1] +
        0.0 * obs[2] +
        0.0 * obs[3] +
        0.0 * obs[4] +
        0.0 * obs[5] +
        0.1 * obs[6] +
        -10.0 * obs[7] +
        -2.0 * obs[8] +
        0.0 * obs[9] +
        0.0 * obs[10] +
        0.0 * action[0] +
        0.0 * action[1]
    )

    return float(reward)
