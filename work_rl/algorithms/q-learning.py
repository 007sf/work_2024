from work_rl.env.grid_world_in_sssf import GridWorld
import numpy as np



def Q_learning_offline_function_epsilon(env, alpha=0.1, iterations=6000, gamma=0.8,epsilon= 0.1):
    np.random.seed(40)
    policy_matrix_target = np.zeros((env.num_states, len(env.action_space)))
    Q = np.zeros((env.num_states, len(env.action_space)))
    action_map = {0: (0, 1),
                  1: (1, 0),
                  2: (0, -1),
                  3: (-1, 0),
                  4: (0, 0),
                  }
    lengths = []
    for iteration in range(iterations):
        s_a_pairs = []
        r_total = []
        s_0_index = np.random.choice(env.num_states)
        # s_0_index = env.start_state[0] + env.start_state[1] * env.env_size[0]
        s_0 = (s_0_index % env.env_size[0], s_0_index // env.env_size[0])
        if np.random.rand()<epsilon:
            a_0_index = np.random.choice(len(env.action_space))
        else:
            a_0_index = np.argmax(Q[s_0_index])

        s_a_pairs.append((s_0, a_0_index))

        while s_0 != env.target_state["position"] and len(s_a_pairs) < 500:
            # for episode in range(episodes):
            s_0={
                "position":s_0,
                "marks_left":env.start_state["marks_left"],
                "marks_pos":env.start_state["marks_pos"]
            }
            next_s, reward = env._get_next_state_and_reward(s_0, a_0_index)
            next_s = next_s["position"]
            next_s_index = next_s[0] + next_s[1] * env.env_size[0]
            if np.random.rand()<epsilon:
                next_a_index = np.random.choice(len(env.action_space))
            else:
                next_a_index = np.argmax(Q[next_s_index])

            next_s = {
                "position":next_s,
                "marks_left":env.start_state["marks_left"],
                "marks_pos":env.start_state["marks_pos"]
            }
            r_total.append(reward)
            s_a_pairs.append((next_s, next_a_index))

            s_0 = next_s
            a_0_index = next_a_index

        for t in range(len(s_a_pairs) - 1):
            s_t_index = s_a_pairs[t][0][0] + s_a_pairs[t][0][1] * env.env_size[0]
            a_t_index = s_a_pairs[t][1]
            s_t_next_index = s_a_pairs[t + 1][0][0] + s_a_pairs[t + 1][0][1] * env.env_size[0]
            idmax = np.argmax(Q[s_t_next_index])
            Q[s_t_index, a_t_index] = Q[s_t_index, a_t_index] - alpha * (Q[s_t_index, a_t_index] -
                                                                         (r_total[t] + gamma * Q[
                                                                             s_t_next_index, idmax]))

            max = np.argmax(Q[s_t_index])
            policy_matrix_target[s_t_index, :] = 0
            policy_matrix_target[s_t_index, max] = 1

    return policy_matrix_target