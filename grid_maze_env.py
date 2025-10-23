# grid_with_policy_iteration.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple
import pickle
import time 

# -------------------
# Gym environment
# -------------------


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, grid_size=5, render_mode=None):
        super().__init__()

        self.grid_size = grid_size
        self.render_mode = render_mode

        # Define action space: up, right, down, left
        self.action_space = spaces.Discrete(4)

        # Observation space: {agent:Box(2), goal:Box(2), walls: MultiBinary grid or list}
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                "goal": spaces.Box(low=0, high=grid_size - 1, shape=(2,), dtype=np.int32),
                # walls: exactly two wall coordinates as ints
                "walls": spaces.Box(low=0, high=grid_size - 1, shape=(2, 2), dtype=np.int32),
            }
        )

        # Initialize pygame for human render
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
        # agent_pos and goal_pos are (row, col)
        self.agent_pos = np.array(
            (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        )
        self.goal_pos = np.array(
            (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        )

        while np.array_equal(self.goal_pos, self.agent_pos):
            self.goal_pos = np.array(
                (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            )

        # Exactly two walls (as in your GridWorld MDP assumption)
        self.walls = []
        while len(self.walls) < 2:
            wall = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
            if wall != tuple(self.agent_pos) and wall != tuple(self.goal_pos) and wall not in self.walls:
                self.walls.append(wall)

        self.walls.sort() 
        observation = {
            "agent": self.agent_pos.copy(),
            "goal": self.goal_pos.copy(),
            "walls": np.array(self.walls, dtype=np.int32),
        }

        return observation, {}

    def step(self, action):
        # Add stochasticity (30% slip total: 15% left, 15% right)
        stochasticity = np.random.rand()
        if stochasticity < 0.15:
            action = (action - 1) % 4
        elif stochasticity < 0.3:
            action = (action + 1) % 4

        move_map = {
            0: np.array([-1, 0]),  # up
            1: np.array([0, 1]),   # right
            2: np.array([1, 0]),   # down
            3: np.array([0, -1]),  # left
        }

        candidate = self.agent_pos + move_map[action]

        # clamp / bounce: if out of bounds stay in place
        if candidate[0] < 0 or candidate[0] >= self.grid_size or candidate[1] < 0 or candidate[1] >= self.grid_size:
            new_pos = self.agent_pos.copy()
        else:
            new_pos = candidate

        self.agent_pos = new_pos

        if tuple(new_pos) in self.walls:
            reward = -0.75
            terminated = True
        else:
            terminated = np.array_equal(self.agent_pos, self.goal_pos)
            reward = 1 if terminated else -0.1

        observation = {
            "agent": self.agent_pos.copy(),
            "goal": self.goal_pos.copy(),
            "walls": np.array(self.walls, dtype=np.int32),
        }
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self, action):
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
        ar, ac = int(self.agent_pos[0]), int(self.agent_pos[1])
        pygame.draw.circle(
            self.screen, (0, 0, 255),
            (ac * self.cell_size + self.cell_size // 2, ar * self.cell_size + self.cell_size // 2),
            self.cell_size // 4
        )

        if (action is not None):
            move_map = {
                0: (-1, 0),  # up
                1: (0, 1),   # right
                2: (1, 0),   # down
                3: (0, -1),  # left
            }


            # Compute the center of the agent cell in pixels
            start_x = ac * self.cell_size + self.cell_size // 2
            start_y = ar * self.cell_size + self.cell_size // 2

            # Compute end point offset in pixels (half cell in the chosen direction)
            dy, dx = move_map[action]
            end_x = start_x + dx * 1.25 * self.cell_size // 2
            end_y = start_y + dy * 1.25 * self.cell_size // 2

            pygame.draw.line(
                self.screen,
                (255, 0, 0),  # red
                (start_x, start_y),
                (end_x, end_y),
                width=8,
            )

            pygame.draw.circle(self.screen, (255, 0, 0), (end_x, end_y), 6)

        # Draw goal
        gr, gc = int(self.goal_pos[0]), int(self.goal_pos[1])
        pygame.draw.rect(
            self.screen, (0, 255, 0),
            pygame.Rect(gc * self.cell_size + 20, gr * self.cell_size + 20, self.cell_size - 40, self.cell_size - 40)
        )



        pygame.display.flip()

    def close(self):
        if self.render_mode == "human":
            pygame.quit()


# -------------------
# MDP abstract interface and GridWorld MDP
# -------------------
class MDP:
    def get_states(self):
        raise NotImplementedError

    def get_actions(self, state):
        raise NotImplementedError

    def get_transitions(self, state, action):
        raise NotImplementedError

    def get_reward(self, state, action, next_state):
        raise NotImplementedError


class GridWorld(MDP):
    """
    MDP states are tuples:
        (ax, ay, gx, gy, b1x, b1y, b2x, b2y)
    where (ax,ay) is the agent, (gx,gy) the goal and b1,b2 the two bad cells.
    """

    def __init__(self, env: GridMazeEnv):
        # store grid sizes
        self.width = env.grid_size
        self.height = env.grid_size
        self.n_cells = env.grid_size * env.grid_size
        self._coords = [(x, y) for x in range(env.grid_size) for y in range(env.grid_size)]

    def get_states(self):
        states: List[Tuple[int, ...]] = []
        cells = range(self.n_cells)

        # choose 2 distinct bad cell indices (unordered)
        for agent_idx, goal_idx in itertools.permutations(cells, 2):
            remaining = set(cells) - {agent_idx, goal_idx}
            for bad1_idx, bad2_idx in itertools.combinations(remaining, 2):
                ax, ay = self._coords[agent_idx]
                gx, gy = self._coords[goal_idx]
                b1x, b1y = self._coords[bad1_idx]
                b2x, b2y = self._coords[bad2_idx]
                states.append((ax, ay, gx, gy, b1x, b1y, b2x, b2y))
        return states

    def get_actions(self, state):
        actions = [0, 1, 2, 3]  # up, right, down, left
        ax, ay = state[0], state[1]
        if ax == 0:  # top edge => can't move up
            if 0 in actions: actions.remove(0)
        if ax == self.height - 1:  # bottom edge => can't move down
            if 2 in actions: actions.remove(2)
        if ay == 0:  # left edge => can't move left
            if 3 in actions: actions.remove(3)
        if ay == self.width - 1:  # right edge => can't move right
            if 1 in actions: actions.remove(1)
        return actions

    def get_transitions(self, state, action):
        """
        Return list of (next_state, prob) based on slip model.
        Doesn't use an env object; only the state tuple.
        """
        ax, ay, gx, gy, b1x, b1y, b2x, b2y = state

        move_map = {
            0: (-1, 0),  # up
            1: (0, 1),   # right
            2: (1, 0),   # down
            3: (0, -1),  # left
        }

        def apply_move(x, y, move):
            dx, dy = move
            nx, ny = x + dx, y + dy
            # if out of bounds, stay in place
            if nx < 0 or nx >= self.height or ny < 0 or ny >= self.width:
                return x, y
            return nx, ny

        intended = apply_move(ax, ay, move_map[action])
        left = apply_move(ax, ay, move_map[(action - 1) % 4])
        right = apply_move(ax, ay, move_map[(action + 1) % 4])

        prob_map = {}
        for pos, p in [(intended, 0.7), (left, 0.15), (right, 0.15)]:
            prob_map[pos] = prob_map.get(pos, 0.0) + p

        transitions = []
        for (nx, ny), p in prob_map.items():
            next_state = (nx, ny, gx, gy, b1x, b1y, b2x, b2y)
            transitions.append((next_state, p))
        return transitions

    def get_reward(self, state, action, next_state):
        nax, nay = next_state[0], next_state[1]
        _, _, gx, gy, b1x, b1y, b2x, b2y = next_state
        if (nax, nay) == (gx, gy):
            return 1
        if (nax, nay) == (b1x, b1y) or (nax, nay) == (b2x, b2y):
            return -0.75
        return -0.1
    
    def is_terminal(self, state):
        ax, ay, gx, gy, b1x, b1y, b2x, b2y = state
        if (ax, ay) == (gx, gy): return True
        if (ax, ay) == (b1x, b1y) or (ax, ay) == (b2x, b2y): return True
        return False


# -------------------
# PolicyIteration (keeps internal transition/reward model compatible with GridWorld)
# -------------------
State = Tuple[int, int, int, int, int, int, int, int]
Action = int


class PolicyIteration:
    def __init__(self, mdp: MDP, gamma: float = 0.99, theta: float = 1e-6):
        self.mdp = mdp
        self.gamma = gamma
        self.theta = theta

        self.states: List[State] = list(self.mdp.get_states())
        def _manhattan(s: State) -> float:
            ax, ay, gx, gy, *_ = s
            return ((abs(ax - gx) + abs(ay - gy)) / 40)


        # Use negative distance so smaller distance -> larger value
        self.V: Dict[State, float] = {s: -float(_manhattan(s)) for s in self.states}
        self.policy: Dict[State, Optional[Action]] = {}
        for s in self.states:
            actions = self.mdp.get_actions(s)
            self.policy[s] = actions[0] if actions else None

    def policy_evaluation(self) -> None:
        while True:
            delta = 0.0
            for s in self.states:
                if self.mdp.is_terminal(s):
                    continue
                pi_a = self.policy.get(s)
                if pi_a is None:
                    continue
                v_old = self.V[s]
                new_v = 0.0
                a = pi_a
                transitions = self.mdp.get_transitions(s, a)
                for s_next, p in transitions:
                    r = self.mdp.get_reward(s, a, s_next)
                    new_v += p * (r + self.gamma * self.V.get(s_next, 0.0))
                self.V[s] = new_v
                delta = max(delta, abs(v_old - new_v))
            if delta < self.theta:
                break

    def policy_improvement(self) -> bool:
        policy_stable = True
        for s in self.states:
            if self.mdp.is_terminal(s):
                continue
            old_action = self.policy.get(s)
            actions = self.mdp.get_actions(s)
            if not actions:
                self.policy[s] = None
                continue
            best_a = None
            best_q = -math.inf
            for a in actions:
                q = 0.0
                transitions = self.mdp.get_transitions(s, a)
                for s_next, p in transitions:
                    r = self.mdp.get_reward(s, a, s_next)
                    q += p * (r + self.gamma * self.V.get(s_next, 0.0))
                if q > best_q:
                    best_q = q
                    best_a = a
            self.policy[s] = best_a
            if old_action != best_a:
                policy_stable = False
        return policy_stable

    def run(self, max_iterations: int = 1000) -> Tuple[Dict[State, Optional[Action]], Dict[State, float]]:
        for i in range(max_iterations):
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print(i)
                break
        return self.policy, self.V

    # -------------------
    # New: save / load and run a single episode using learned policy
    # -------------------
    def save_model(self, filepath: str) -> None:
        """Save learned policy and value function to a file using pickle."""
        with open(filepath, "wb") as f:
            pickle.dump({"policy": self.policy, "V": self.V}, f)

    def load_model(self, filepath: str) -> None:
        """Load policy and value function from file and replace current policy/V."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.policy = data.get("policy", {})
        self.V = data.get("V", {})

    def _obs_to_state(self, obs: Any) -> State:
        """Convert environment observation to MDP state tuple.
        Canonicalize the two bad cells into a deterministic order so they match
        the unordered pair used by GridWorld.get_states().
        """
        if isinstance(obs, dict):
            agent = obs["agent"]
            goal = obs["goal"]
            walls = obs["walls"]
            if isinstance(walls, np.ndarray):
                b1 = tuple(int(x) for x in walls[0])
                b2 = tuple(int(x) for x in walls[1])
            else:
                b1, b2 = walls[0], walls[1]

        ax, ay = int(agent[0]), int(agent[1])
        gx, gy = int(goal[0]), int(goal[1])
        b1x, b1y = int(b1[0]), int(b1[1])
        b2x, b2y = int(b2[0]), int(b2[1])

        # canonicalize order by flattened index so (b1,b2) is consistent
        idx1 = b1x * self.mdp.width + b1y
        idx2 = b2x * self.mdp.width + b2y
        if idx1 <= idx2:
            return (ax, ay, gx, gy, b1x, b1y, b2x, b2y)
        else:
            return (ax, ay, gx, gy, b2x, b2y, b1x, b1y)

    def run_episode(self, env: GridMazeEnv, max_steps: int = 100, render: bool = False) -> Tuple[float, int]:
        """Run one episode in the environment following the learned policy.

        Returns (total_reward, steps_taken).
        If a state encountered isn't in the stored policy mapping, we choose a random legal action.
        """
        obs, _ = env.reset()
        total_reward = 0.0
        steps = 0
        for _ in range(max_steps):
            state = self._obs_to_state(obs)
            # choose action from policy if available, otherwise random legal action
            a = self.policy.get(state)
            action = int(a)

            env.render(a)
            time.sleep(1)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if render and env.render_mode == "human":
                env.render(None)
                time.sleep(0.5)
            if terminated or truncated:
                break
        return total_reward, steps


# -------------------
# Example usage
# -------------------
if __name__ == "__main__":
    env = GridMazeEnv(grid_size=5, render_mode="human")
    mdp = GridWorld(env)

    #solver = PolicyIteration(mdp, gamma=0.99, theta=1e-6)
    #policy, V = solver.run()

    # save model
    #solver.save_model("learned_policy.pkl")

    # create a fresh solver, load model and run an episode
    solver2 = PolicyIteration(mdp, gamma=0.99, theta=1e-6)#
    solver2.load_model("learned_policy.pkl")

    total_reward, steps = solver2.run_episode(env, max_steps=100, render=True)
    print(f"Episode finished: total_reward={total_reward}, steps={steps}")
