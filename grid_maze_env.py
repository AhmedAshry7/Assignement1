import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
import numpy as np
import pygame
import random
import itertools
from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction
from qtable import QTable

class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=5, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Define action space: up, right, down, left
        self.action_space = spaces.Discrete(4)

        # Observation space: agent position (row, col)
        self.observation_space = spaces.Box(
            low=0, high=grid_size - 1, shape=(2,), dtype=np.int32
        )

        # Initialize pygame
        if render_mode == "human":
            pygame.init()
            self.cell_size = 100
            self.screen = pygame.display.set_mode(
                (grid_size * self.cell_size, grid_size * self.cell_size)
            )
            pygame.display.set_caption("Grid Maze")

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array((random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)))
        self.goal_pos = np.array((random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))) 

        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = np.array((random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)))

        # Random walls
        self.walls = []
        while len(self.walls) < 2:
            wall = (
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1)
            )

            # Make sure wall doesn't overlap with agent, goal, or existing walls
            if (
                wall != tuple(self.agent_pos)
                and wall != tuple(self.goal_pos)
                and wall not in self.walls
            ):
                self.walls.append(wall)

        obervation = (self.agent_pos.copy(),self.goal_pos.copy(),list(self.walls))
        return obervation, {}

    def step(self, action):
        # Add stochasticity (e.g., 30% chance to take a random move)
        stochasticity = np.random.rand()
        if  stochasticity< 0.15:
            action = (action-1)%4
        elif  stochasticity< 0.3:
            action = (action+1)%4
        move_map = {
            0: np.array([-1, 0]),  # up
            1: np.array([0, 1]),   # right
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),  # left
        }

        new_pos = self.agent_pos + move_map[action]

        # Check boundaries and walls
        self.agent_pos = new_pos
        if (tuple(new_pos) in self.walls):
            reward = -1  # Penalty for hitting a wall
            terminated=True
        else:
            terminated = np.array_equal(self.agent_pos, self.goal_pos)
            reward = 1.0 if terminated else -0.01
        
        obervation = (self.agent_pos.copy(),self.goal_pos.copy(),list(self.walls))
        return obervation, reward, terminated, False, {}

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill((255, 255, 255))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)
                color = (200, 200, 200)
                if (r, c) in self.walls:
                    color = (50, 50, 50)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 2)

        # Draw agent
        ar, ac = self.agent_pos
        pygame.draw.circle(
            self.screen, (0, 0, 255),
            (ac * self.cell_size + self.cell_size // 2, ar * self.cell_size + self.cell_size // 2),
            self.cell_size // 4
        )

        # Draw goal
        gr, gc = self.goal_pos
        pygame.draw.rect(
            self.screen, (0, 255, 0),
            pygame.Rect(gc * self.cell_size + 20, gr * self.cell_size + 20, self.cell_size - 40, self.cell_size - 40)
        )

        pygame.display.flip()

    def close(self):
        if self.render_mode == "human":
            pygame.quit()

class MDP:
    """ Return all states of this MDP """
    def get_states(self):
        abstract

    """ Return all actions with non-zero probability from this state """
    def get_actions(self, state):
        abstract

    """ Return all non-zero probability transitions for this action
        from this state, as a list of (state, probability) pairs
    """
    def get_transitions(self, state, action):
        abstract

    """ Return the reward for transitioning from state to
        nextState via action
    """
    def get_reward(self, state, action, next_state):
        abstract

    """ Return true if and only if state is a terminal state of this MDP """
    def is_terminal(self, state):
        abstract

    """ Return the discount factor for this MDP """
    def get_discount_factor(self):
        abstract

    """ Return the initial state of this MDP """
    def get_initial_state(self):
        abstract

    """ Return all goal states of this MDP """
    def get_goal_states(self):
        abstract

class GridWorld(MDP):

    def __init__(self,env):
        self.width = env.grid_size
        self.height = env.grid_size
        # flattened cell indices 0..(width*height-1)
        self.n_cells = env.grid_size * env.grid_size
        # precompute coords list so we map index->(x,y) fast
        self._coords = [(x, y) for x in range(env.grid_size) for y in range(env.grid_size)]

    """ Return all states of this MDP """
    def get_states(self):
        """
        Return a list of all valid states.
        State representation: (ax, ay, gx, gy, b1x, b1y, b2x, b2y)
        We require the 4 cells (agent, goal, bad1, bad2) to be distinct.
        """
        states = []
        # iterate over all permutations of 4 distinct cell indices
        for agent_idx, goal_idx, bad1_idx, bad2_idx in itertools.permutations(range(self.n_cells), 4):
            ax, ay = self._coords[agent_idx]
            gx, gy = self._coords[goal_idx]
            b1x, b1y = self._coords[bad1_idx]
            b2x, b2y = self._coords[bad2_idx]
            states.append((ax, ay, gx, gy, b1x, b1y, b2x, b2y))
        return states

    """ Return all actions with non-zero probability from this state """
    def get_actions(self, state):
        actions=[0,1,2,3] # up, right, down, left
        ax, ay = state[0], state[1]

        # Check grid boundaries
        if ax == 0:  # top edge → can't move up
            actions.remove(0)
        if ax == self.grid_size - 1:  # bottom edge → can't move down
            actions.remove(2)
        if ay == 0:  # left edge → can't move left
            actions.remove(3)
        if ay == self.grid_size - 1:  # right edge → can't move right
            actions.remove(1)
        
        return actions


    def get_transitions(self,env, state, action):
        transitions = []

        # Probability of not slipping left or right
        # Add stochasticity (e.g., 30% chance to take a random move)

        move_map = {
            0: np.array([-1, 0]),  # up
            1: np.array([0, 1]),   # right
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),  # left
        }
        new_pos = env.agent_pos + move_map[action]
        new_pos1 = env.agent_pos + move_map[(action-1)%4]
        new_pos2 = env.agent_pos + move_map[(action+1)%4]
        transitions += [(new_pos, 0.7)]
        transitions += [(new_pos1, 0.15)]
        transitions += [(new_pos2, 0.15)]

        return transitions

    def get_reward(self, state,env, action):
        reward = 0.0
        if np.array_equal(env.goal_pos, env.agent_pos):
            reward = 1
        elif tuple(env.agent_pos) in env.walls:
            reward = -1
        else:
            reward = -0.01
        return reward


class PolicyIteration:
    def __init__(self, mdp, policy):
        self.mdp = mdp
        self.policy = policy

    def policy_evaluation(self, policy, values, theta=0.001):

        while True:
            delta = 0.0
            new_values = TabularValueFunction()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                actions = self.mdp.get_actions(state)
                old_value = values.get_value(state)
                new_value = values.get_q_value(
                    self.mdp, state, policy.select_action(state, actions)
                )
                values.add(state, new_value)
                delta = max(delta, abs(old_value - new_value))

            # terminate if the value function has converged
            if delta < theta:
                break

        return values

    """ Implmentation of policy iteration iteration. Returns the number of iterations executed """

    def policy_iteration(self, max_iterations=100, theta=0.001):

        # create a value function to hold details
        values = TabularValueFunction()

        for i in range(1, max_iterations + 1):
            policy_changed = False
            values = self.policy_evaluation(self.policy, values, theta)
            for state in self.mdp.get_states():

                actions = self.mdp.get_actions(state)
                old_action = self.policy.select_action(state, actions)

                q_values = QTable(alpha=1.0)
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = values.get_q_value(self.mdp, state, action)
                    q_values.update(state, action, new_value)
                # V(s) = argmax_a Q(s,a)
                new_action = q_values.get_argmax_q(state, self.mdp.get_actions(state))
                self.policy.update(state, new_action)
                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )

            if not policy_changed:
                return i

        return max_iterations


import math
import random
from typing import Any, Dict, List, Tuple, Optional

class PolicyIteration:
    """
    Policy Iteration for a generic MDP.
    mdp: object exposing get_states(), get_actions(state), get_transitions(...) and get_reward(...)
    env: optional environment passed to mdp methods if required (some implementations expect env as 1st arg)
    gamma: discount factor
    theta: evaluation stopping tolerance
    max_eval_iters: max iterations for policy evaluation
    """

    def __init__(self, mdp: Any, env: Any = None, gamma: float = 0.99,
                 theta: float = 1e-6, max_eval_iters: int = 10000):
        self.mdp = mdp
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.max_eval_iters = max_eval_iters

        # states and initial structures
        self.states = list(self.mdp.get_states())
        self.V: Dict[Any, float] = {s: 0.0 for s in self.states}
        # deterministic policy: state -> action
        self.policy: Dict[Any, Any] = {}
        self._init_random_policy()

    def _init_random_policy(self):
        """Initialize with a random valid action for each state (deterministic)"""
        for s in self.states:
            actions = self._safe_get_actions(s)
            if not actions:
                # terminal or no action states
                self.policy[s] = None
            else:
                self.policy[s] = random.choice(actions)

    # --- wrappers to adapt to various mdp method signatures ---
    def _safe_get_actions(self, state):
        try:
            return self.mdp.get_actions(state)
        except TypeError:
            # maybe signature is get_actions(env, state)
            return self.mdp.get_actions(self.env, state)

    def _safe_get_transitions(self, state, action) -> List[Tuple[Any, float]]:
        """
        Returns list of (next_state, prob).
        Tries common signatures: get_transitions(state, action) or get_transitions(env, state, action)
        """
        # call and then normalize next_state representation (e.g., numpy -> tuple)
        try:
            trans = self.mdp.get_transitions(state, action)
        except TypeError:
            trans = self.mdp.get_transitions(self.env, state, action)

        # trans may be list of (next_pos, prob) or (next_state, prob)
        normed = []
        for (ns, p) in trans:
            ns_key = self._state_key(ns)
            normed.append((ns_key, float(p)))
        return normed

    def _safe_get_reward(self, state, action, next_state) -> float:
        """
        Tries multiple possible reward signatures:
         - get_reward(state, action, next_state)
         - get_reward(state, env, action)  (as in given GridWorld)
         - get_reward(state, action)
        """
        # try most informative signature first
        try:
            return float(self.mdp.get_reward(state, action, next_state))
        except TypeError:
            pass
        try:
            # gridworld-like signature: get_reward(state, env, action)
            return float(self.mdp.get_reward(state, self.env, action))
        except TypeError:
            pass
        try:
            return float(self.mdp.get_reward(state, action))
        except TypeError:
            # fallback: zero reward if nothing matches
            return 0.0

    def _state_key(self, s):
        """Convert possible numpy arrays or lists to hashable tuple keys"""
        try:
            # numpy arrays have .tolist()
            if hasattr(s, "tolist"):
                return tuple(s.tolist())
            # lists -> tuple
            if isinstance(s, list):
                return tuple(s)
            # already a tuple or hashable
            return s
        except Exception:
            return s

    # --- policy evaluation (iterative) ---
    def policy_evaluation(self):
        """
        Iteratively evaluate current self.policy and update self.V in-place.
        Uses the equation:
        V(s) <- sum_{a} pi(a|s) sum_{s'} P(s'|s,a) [ R(s,a,s') + gamma * V(s') ]
        But since policy here is deterministic, we evaluate for the single action pi(s).
        """
        for it in range(self.max_eval_iters):
            delta = 0.0
            new_V = dict(self.V)  # compute updates into new_V then assign
            for s in self.states:
                a = self.policy.get(s)
                if a is None:
                    # terminal/no-action state
                    new_v = 0.0
                else:
                    # sum over next states
                    total = 0.0
                    transitions = self._safe_get_transitions(s, a)
                    if not transitions:
                        new_v = 0.0
                    else:
                        for s_next, prob in transitions:
                            r = self._safe_get_reward(s, a, s_next)
                            v_next = self.V.get(s_next, 0.0)
                            total += prob * (r + self.gamma * v_next)
                        new_v = total
                delta = max(delta, abs(new_v - self.V.get(s, 0.0)))
                new_V[s] = new_v
            self.V = new_V
            if delta < self.theta:
                # converged
                break

    # --- policy improvement (greedy) ---
    def policy_improvement(self) -> bool:
        """
        Make policy greedy wrt current value function V.
        Returns True if policy changed (so another iteration is required).
        """
        policy_stable = True
        for s in self.states:
            actions = self._safe_get_actions(s)
            if not actions:
                self.policy[s] = None
                continue

            # compute action-values
            best_a = None
            best_q = -math.inf
            for a in actions:
                q = 0.0
                transitions = self._safe_get_transitions(s, a)
                for s_next, prob in transitions:
                    r = self._safe_get_reward(s, a, s_next)
                    v_next = self.V.get(s_next, 0.0)
                    q += prob * (r + self.gamma * v_next)
                # break ties deterministically by action order (first encountered)
                if q > best_q:
                    best_q = q
                    best_a = a

            if best_a is None:
                best_a = actions[0]

            if self.policy.get(s) != best_a:
                policy_stable = False
                self.policy[s] = best_a

        return not policy_stable  # return True when changed (need another outer loop)

    # --- top-level policy iteration ---
    def run(self, max_iterations: int = 1000) -> Tuple[Dict[Any, Any], Dict[Any, float]]:
        """
        Run full policy iteration until policy is stable or max_iterations reached.
        Returns (policy, V)
        """
        for i in range(max_iterations):
            self.policy_evaluation()
            changed = self.policy_improvement()
            if not changed:
                # policy is stable -> done
                break
        return self.policy, self.V

# ---------------------------
# Example usage (pseudocode):
#
# from your_module import GridWorld, Env
# env = Env(grid_size=4, ...)      # your environment instance
# mdp = GridWorld(env)
# pi = PolicyIteration(mdp, env=env, gamma=0.99)
# policy, V = pi.run()
#
# policy is a dict mapping states to chosen action (0/1/2/3 in your GridWorld).
# V is a dict mapping states to values.
#
# NOTE: If your mdp.get_transitions returns positions (like numpy arrays) rather than
# complete state tuples, make sure transitions return full next-state descriptions that
# match items from mdp.get_states(). The wrapper tries to convert arrays->tuples,
# but the semantics must match your MDP state's structure.
