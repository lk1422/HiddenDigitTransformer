import numpy as np

LEFT  = 0
RIGHT = 1

class basicMDP():
    """
    Basic RL Enviornment for Debugging RL Algs
    It is a 5 state MDP with model dynamics 
    hard coded below.
    """
    def __init__(self):
        self.states = 6
        self.rewards = np.zeros((6, 6))
        self.pr = np.zeros((6,6,2))
        self.terminal = set([1, 2, 4, 6])
        self._init_rewards()
        self._init_transition_probs()

    def _init_rewards(self):
        """
        Describes all non-zero rewards from transition of
        state1 to state2.
        """
        self.rewards[0, 1] =  1
        self.rewards[0, 2] = -5
        self.rewards[3, 2] = -5
        self.rewards[3, 4] = 10

    def _init_transition_probs(self):
        """
        Describes the transition probabilities
        self.pr[s_t, s_prev, a_prev] = pr(s_t | s_prev, a_prev)
        a_prev = {Left(0), Right(1)}
        """
        self.pr[1, 0, RIGHT] = 1
        self.pr[3, 0,  LEFT] = 0.5
        self.pr[2, 0,  LEFT] = 0.5

        self.pr[4, 3,  LEFT] = 0.5
        self.pr[2, 3,  LEFT] = 0.5
        self.pr[0, 3, RIGHT] = 0.5
        self.pr[2, 3, RIGHT] = 0.5

        self.pr[1,1,  LEFT] = 1
        self.pr[1,1, RIGHT] = 1
        self.pr[2,2,  LEFT] = 1
        self.pr[2,2, RIGHT] = 1
        self.pr[4,4,  LEFT] = 1
        self.pr[4,4, RIGHT] = 1
        self.pr[5,5,  LEFT] = 1
        self.pr[5,5, RIGHT] = 1

    def get_start_state(self, n):
        start =  np.zeros((n, 1, 6))
        start[:, :, 0] = 1
        return start

    def batched_step(self, state, action):
        """
        state : (N, 6) current_state array
        action: (N,) action array 
        Returns (next_states, rewards, Terminal)
        """
        N = state.shape[0]

        state = np.argmax(state, axis=1)

        transition_probs = self.pr[:, state, action].T

        next_state = np.array([np.random.choice(a=6, p=transition_probs[i]) \
                       for i in range(N)]).reshape(-1)

        next_state_binary = np.zeros((N, 6))
        next_state_binary[np.arange(0,N), next_state] = 1

        rewards = self.rewards[state, next_state]
        terminal = (next_state == 1) | (next_state == 2) | (next_state == 4) | (next_state == 6)

        return next_state_binary, rewards, terminal







