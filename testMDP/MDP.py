import numpy as np
from .utils import softmax

LEFT  = 0
RIGHT = 1
UP    = 2
DOWN  = 3

OUT_OF_BOUND_REWARD = -5
WIN_REWARD = 10

class RandomGridWorld():
    """
    Slightly More complicated RL Enviorment.
    To test the ability of PPO of converging upon more
    complex policy
    """
    def __init__(self, rows, cols):
        """
        Initialize RandomGridWorld object

        Parameters:
            rows: # of rows present in random grid (rows > 1)
            cols: # of columbs present in random grid (cols > 1)

        Class Members:
            self.pr = PR(ACTION | row, col, REQUESTED_ACTION)
            ie the P(UP | 1, 2, DOWN) is the probability that the 
            action preformed is down given the state and action.

            self.reward(row, col, action) is the reward for preforming an 
            action in some state.

            self.terminal_states(row, col) <==> row,col is a terminal state

            S : start = (1,1)
            D : dest  = (3,3)
            X : out of grid
            |X|X|X|X|X|
            |X|S|.|.|X|
            |X|.|.|.|X|
            |X|.|.|D|X|
            |X|X|X|X|X|

        """
        assert rows > 1, "rows <= 1 is not valid"
        assert cols > 1, "cols <= 1 is not valid"
        
        self.rows = rows
        self.cols = cols

        self.states = (rows+2) * (cols+2)
        self.SOS = self.states+1
        self.PAD = self.states+2

        self._init_rewards()
        self._init_terminal_states()
        self.pr = softmax(np.random.randn(4, rows+2,cols+2, 4))

    def _init_rewards(self):
        self.rewards = -np.random.uniform(low=0, high=1, size=(self.rows+2, self.cols+2, 4))
        self.rewards[self.rows, :, DOWN] = OUT_OF_BOUND_REWARD
        self.rewards[:, self.cols, RIGHT] = OUT_OF_BOUND_REWARD
        self.rewards[1, :, UP] = OUT_OF_BOUND_REWARD
        self.rewards[:, 1, LEFT] = OUT_OF_BOUND_REWARD
        self.rewards[self.rows-1, self.cols, DOWN] = WIN_REWARD
        self.rewards[self.rows, self.cols-1, RIGHT] = WIN_REWARD

    def _init_terminal_states(self):
        self.terminal_states = np.zeros((self.rows+2, self.cols+2))
        self.terminal_states[self.rows+1, :] = 1
        self.terminal_states[0, :] = 1
        self.terminal_states[:, self.cols+1] = 1
        self.terminal_states[:, 0] = 1

    def get_state_idx(self, row, col):
        """
        Convert row, col to => 0 - self.states
        each row, col pair maps to a unique number 
        in this range. This is to represent the token_idx.
        """
        assert row >= 0 and row < self.rows+2, "OUT OF BOUNDS ROW"
        assert col >= 0 and col < self.rows+2, "OUT OF BOUNDS ROW"
        return row * (self.cols+2) + col

    def get_start_state(self, n):
        return (np.ones((n,1)) * self.get_state_idx(1,1)).astype(np.int32)

    def preform_action(self, state, actions):
        """
        Preform the randomly generated action on the state
        """

        state = state.reshape(-1)

        is_up    = (actions==UP)
        is_down  = (actions==DOWN)
        is_left  = (actions==LEFT)
        is_right = (actions==RIGHT)

        action_up = state - (self.cols+2)
        action_down = state + (self.cols+2)
        action_left = state - 1
        action_right = state + 1
        
        preformed_action = (is_up * action_up) + (is_down * action_down) + \
                (is_left * action_left) + (is_right * action_right)

        terminal = self.terminal_states.reshape(-1)[state]

        return (terminal * state) + \
                (np.logical_not(terminal) * preformed_action)

    def batched_step(self, state, action):
        """
        state : (N, 6) current_state array
        action: (N,) action array 
        Returns (next_states, rewards, Terminal)
        """
        N = state.shape[0]

        print("STATE", state)
        print("ACTION", action)

        row_idx = (state / (self.cols+2)).astype(np.int32).reshape(-1)
        col_idx = (state % (self.cols+2)).astype(np.int32).reshape(-1)

        print(row_idx)
        print(col_idx)


        probs = self.pr[:, row_idx, col_idx, action]
        action_preformed = np.array([np.random.choice(a=4, p=probs[:,i]) \
                       for i in range(N)]).reshape(-1)

        preform_action = self.preform_action(state, action_preformed).astype(np.int32)
        terminal = self.terminal_states.reshape(-1)
        rewards = self.rewards[row_idx,  col_idx, action_preformed]


        return preform_action, rewards, (terminal[preform_action]).astype(np.int32)

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
        self.rewards[0, 1] =  1/50
        self.rewards[0, 2] = -5/50
        self.rewards[3, 2] = -5/50
        self.rewards[3, 4] = 1

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

