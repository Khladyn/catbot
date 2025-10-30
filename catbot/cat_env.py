import os
import random
import time
from typing import Optional, Tuple, Any, Type
from abc import ABC, abstractmethod
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

class Cat(ABC):
    def __init__(self, grid_size: int, tile_size: int):
        self.grid_size = grid_size
        self.tile_size = tile_size
        self.pos = np.zeros(2, dtype=np.int32)
        self.visual_pos = np.zeros(2, dtype=float)
        
        self.player_pos = np.zeros(2, dtype=np.int32)
        self.prev_player_pos = np.zeros(2, dtype=np.int32)
        self.last_player_action = None
        
        self.current_distance = 0  
        self.prev_distance = 0     
        
        self._load_sprite()
    
    @abstractmethod
    def _get_sprite_path(self) -> str:
        pass
    
    def _load_sprite(self):
        img_path = self._get_sprite_path()
        if not os.path.exists(img_path):
            self.sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.sprite.fill((200, 100, 100))
            return
        try:
            self.sprite = pygame.image.load(img_path)
            self.sprite = self.sprite.convert_alpha()
            self.sprite = pygame.transform.scale(self.sprite, (self.tile_size, self.tile_size))
        except Exception as e:
            self.sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.sprite.fill((200, 100, 100))
    
    def update_player_info(self, player_pos: np.ndarray, player_action: int) -> None:
        self.prev_player_pos = self.player_pos.copy()
        self.player_pos = player_pos.copy()
        self.last_player_action = player_action

        self.prev_distance = abs(self.pos[0] - self.prev_player_pos[0]) + abs(self.pos[1] - self.prev_player_pos[1])
        self.current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
    
    def player_moved_closer(self) -> bool:
        return self.current_distance < self.prev_distance
    
    @abstractmethod
    def move(self) -> None:
        pass
    
    def reset(self, pos: np.ndarray) -> None:
        self.pos = pos.copy()
        self.visual_pos = pos.astype(float)
    
    def update_visual_pos(self, dt: float, animation_speed: float) -> None:
        for i in range(2):
            diff = self.pos[i] - self.visual_pos[i]
            if abs(diff) > 0.01:
                self.visual_pos[i] += np.clip(diff * animation_speed * dt, -1, 1)

####################################
# CAT BEHAVIOR IMPLEMENTATIONS     #
####################################

class BatmeowCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/batmeow-dp.png"
    def move(self) -> None:
        pass

class MittensCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/mittens-dp.png"
    def move(self) -> None:
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(dirs)
        d = dirs[0]
        new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
        new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
        self.pos[0] = new_r
        self.pos[1] = new_c

class PaotsinCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/paotsin-dp.png"

    def move(self) -> None:
        current_distance = abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1])
        if current_distance > 4:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            d = dirs[0]
            new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
            self.pos[0] = new_r
            self.pos[1] = new_c
            return
            
        possible_moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        random.shuffle(possible_moves)
        best_move = None
        best_distance = None
        
        for dr, dc in possible_moves:
            new_r = min(max(0, self.pos[0] + dr), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + dc), self.grid_size - 1)
            
            distance = abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1])
            
            if best_distance is None:
                best_move = (dr, dc)
                best_distance = distance
            elif self.player_moved_closer():
                if distance > best_distance:
                    best_move = (dr, dc)
                    best_distance = distance
            else:
                if distance < best_distance:
                    best_move = (dr, dc)
                    best_distance = distance
        
        if self.player_moved_closer():
            self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
            self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
        elif random.random() < 0.65:
            self.pos[0] = min(max(0, self.pos[0] + best_move[0]), self.grid_size - 1)
            self.pos[1] = min(max(0, self.pos[1] + best_move[1]), self.grid_size - 1)
        else:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            d = dirs[0]
            new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
            self.pos[0] = new_r
            self.pos[1] = new_c

class PeekabooCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/peekaboo-dp.png" 
    
    def move(self) -> None:
        is_adjacent = (
            abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1]) == 1
        )
        
        if not is_adjacent:
            return
            
        if (self.pos[0] == 0 and self.player_pos[0] == 0 and self.player_pos[1] == self.pos[1] - 1) or (self.pos[1] == 0 and self.player_pos[1] == 0 and self.player_pos[0] == self.pos[0] - 1):
            return
            
        edge_positions = []
        for i in range(self.grid_size):
            edge_positions.extend([
                (0, i), 
                (self.grid_size-1, i), 
                (i, 0),          
                (i, self.grid_size-1)   
            ])
        
        edge_positions = list(set(edge_positions))
        
        safe_positions = []
        for pos in edge_positions:
            if abs(pos[0] - self.player_pos[0]) + abs(pos[1] - self.player_pos[1]) > 1:
                safe_positions.append(pos)
        
        if safe_positions:
            new_pos = random.choice(safe_positions)
            self.pos[0] = new_pos[0]
            self.pos[1] = new_pos[1]

class SquiddyboiCat(Cat):
    def _get_sprite_path(self) -> str:
        return "images/squiddyboi-dp.png"
    
    def move(self) -> None:
        is_adjacent = (
            abs(self.pos[0] - self.player_pos[0]) + abs(self.pos[1] - self.player_pos[1]) == 1
        )
        if not is_adjacent:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            random.shuffle(dirs)
            
            for d in dirs:
                new_r = min(max(0, self.pos[0] + d[0]), self.grid_size - 1)
                new_c = min(max(0, self.pos[1] + d[1]), self.grid_size - 1)
                
                would_be_adjacent = (
                    abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1]) == 1
                )
                
                if not would_be_adjacent:
                    self.pos[0] = new_r
                    self.pos[1] = new_c
                    return
            return
            
        dr = self.player_pos[0] - self.pos[0]
        dc = self.player_pos[1] - self.pos[1]
        if dr != 0:
            dr = dr // abs(dr)
        if dc != 0:
            dc = dc // abs(dc)
            
        target_r = self.player_pos[0] + 2 * dr
        target_c = self.player_pos[1] + 2 * dc
        
        if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
            # Three-space hop is possible
            self.pos[0] = target_r
            self.pos[1] = target_c
            return
            
        target_r = self.player_pos[0] + dr
        target_c = self.player_pos[1] + dc
        
        if (0 <= target_r < self.grid_size and 0 <= target_c < self.grid_size):
            self.pos[0] = target_r
            self.pos[1] = target_c
            return
            
        new_r = min(max(0, self.pos[0] - dr), self.grid_size - 1)
        new_c = min(max(0, self.pos[1] - dc), self.grid_size - 1)
        self.pos[0] = new_r
        self.pos[1] = new_c


####################################
# HIDDEN CAT BEHAVIOR IMPLEMENTATIONS #
####################################

class SpiderCat(Cat):
    """
    Stalker: Tries to maintain a Manhattan distance of 3.
    Moves away if closer, moves toward if farther.
    """

    def _get_sprite_path(self) -> str:
        # You can create a new sprite or just re-use one
        return "images/spidercat-dp.png"

    def move(self) -> None:
        target_distance = 3
        if self.current_distance == target_distance:
            return  # Stay still

        # Check all moves, including staying still
        possible_moves = []  # (pos, dist)
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]  # U/D/L/R + Still

        for dr, dc in dirs:
            if dr == 0 and dc == 0:
                new_pos = (self.pos[0], self.pos[1])
                new_dist = self.current_distance
            else:
                new_r = min(max(0, self.pos[0] + dr), self.grid_size - 1)
                new_c = min(max(0, self.pos[1] + dc), self.grid_size - 1)
                new_pos = (new_r, new_c)
                new_dist = abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1])
            possible_moves.append((new_pos, new_dist))

        best_moves = []  # List of best positions (pos)

        # We want the move that minimizes the |distance - 3|
        best_metric = float('inf')  # Best "distance-from-3"

        for pos, dist in possible_moves:
            metric = abs(dist - target_distance)
            if metric < best_metric:
                best_metric = metric
                best_moves = [pos]  # New best, reset list
            elif metric == best_metric:
                best_moves.append(pos)  # Add to list of ties

        if not best_moves:
            self.pos[0], self.pos[1] = self.pos  # Failsafe
        else:
            # Choose one random move from all the best ones
            self.pos[0], self.pos[1] = random.choice(best_moves)


class CheddarCat(Cat):
    """
    Intelligent: Anti-Greedy Retreat. Always moves to the adjacent
    square that MAXIMIZES the Manhattan distance from the player.
    """

    def _get_sprite_path(self) -> str:
        return "images/cheddar-dp.png"  # Re-using sprite

    def move(self) -> None:
        best_moves = [self.pos]  # Start with "stay still" as a best option
        best_dist = self.current_distance

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U/D/L/R

        for dr, dc in dirs:
            new_r = min(max(0, self.pos[0] + dr), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + dc), self.grid_size - 1)
            new_dist = abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1])
            new_pos = (new_r, new_c)

            if new_dist > best_dist:
                best_dist = new_dist
                best_moves = [new_pos]  # New best, reset list
            elif new_dist == best_dist:
                best_moves.append(new_pos)  # Add to list of ties

        # Choose one random move from all the best ones
        self.pos[0], self.pos[1] = random.choice(best_moves)


class PumpkinPieCat(Cat):
    """
    Angry: Retaliatory Block. Stays still unless player moves closer.
    If player moves closer, takes a 2-space greedy move *toward* player.
    """

    def _get_sprite_path(self) -> str:
        return "images/pumpkinpie-dp.png"  # Re-using sprite

    def move(self) -> None:
        # 1. Check if player moved closer
        if not self.player_moved_closer():
            return  # Stay still

        # 2. Player moved closer, retaliate!
        dr = self.player_pos[0] - self.pos[0]
        dc = self.player_pos[1] - self.pos[1]

        new_r, new_c = self.pos.copy()

        # Move 2 steps in the direction of greatest distance
        if abs(dr) > abs(dc):
            step = 2 * np.sign(dr) if dr != 0 else 0
            new_r += step
        else:
            step = 2 * np.sign(dc) if dc != 0 else 0
            new_c += step

        # Clamp to grid
        self.pos[0] = min(max(0, new_r), self.grid_size - 1)
        self.pos[1] = min(max(0, new_c), self.grid_size - 1)


class MilkyCat(Cat):
    """
    Magical: Conditional Teleport. Moves 1 tile greedily towards player.
    If that move would make it adjacent (dist=1), teleport to a
    random non-adjacent tile instead.
    """

    def _get_sprite_path(self) -> str:
        return "images/milky-dp.png"  # Re-using sprite

    def move(self) -> None:
        # 1. Find all best greedy moves (minimizes distance)
        best_moves = [self.pos]  # Start with "stay still"
        best_dist = self.current_distance

        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U/D/L/R

        for dr, dc in dirs:
            new_r = min(max(0, self.pos[0] + dr), self.grid_size - 1)
            new_c = min(max(0, self.pos[1] + dc), self.grid_size - 1)
            new_pos = (new_r, new_c)
            new_dist = abs(new_r - self.player_pos[0]) + abs(new_c - self.player_pos[1])

            if new_dist < best_dist:
                best_dist = new_dist
                best_moves = [new_pos]  # New best, reset list
            elif new_dist == best_dist:
                best_moves.append(new_pos)  # Add to list of ties

        # 2. Check the teleport condition
        if best_dist == 1:
            # Teleport! Find all safe (non-adjacent) positions
            safe_positions = []
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    dist = abs(r - self.player_pos[0]) + abs(c - self.player_pos[1])
                    if dist > 1:
                        safe_positions.append((r, c))

            if safe_positions:
                # Pick a random safe spot
                self.pos[0], self.pos[1] = random.choice(safe_positions)
            else:
                # No safe spots (player has cat cornered), just take one of the greedy moves
                self.pos[0], self.pos[1] = random.choice(best_moves)
        else:
            # 3. No teleport, just make one of the (randomly chosen) best greedy moves
            self.pos[0], self.pos[1] = random.choice(best_moves)


class TaroCat(Cat):
    """
    Enzo/Axis Lock: Prioritizes X-axis (row) moves until blocked
    by wall or player, then switches to Y-axis (column).
    """

    def __init__(self, grid_size: int, tile_size: int):
        super().__init__(grid_size, tile_size)
        self.axis_lock = 'x'  # Start with x-axis

    def reset(self, pos: np.ndarray) -> None:
        super().reset(pos)
        self.axis_lock = 'x'  # Reset lock on new episode

    def _get_sprite_path(self) -> str:
        return "images/taro-dp.png"  # Re-using sprite

    def move(self) -> None:
        dr = self.player_pos[0] - self.pos[0]
        dc = self.player_pos[1] - self.pos[1]
        player_r, player_c = self.player_pos

        if self.axis_lock == 'x':
            if dr != 0:
                # Try to move on x-axis
                r_step = int(np.sign(dr))
                test_r = self.pos[0] + r_step
                test_r_clamped = min(max(0, test_r), self.grid_size - 1)

                if test_r != test_r_clamped:  # Blocked by wall
                    self.axis_lock = 'y'
                elif test_r == player_r and self.pos[1] == player_c:  # Blocked by player
                    self.axis_lock = 'y'
                else:
                    self.pos[0] = test_r_clamped
                    return
            else:
                # Aligned on x-axis
                self.axis_lock = 'y'

        # Fall-through to y-axis if x-axis was blocked, aligned, or lock was already 'y'
        if self.axis_lock == 'y':
            if dc != 0:
                # Try to move on y-axis
                c_step = int(np.sign(dc))
                test_c = self.pos[1] + c_step
                test_c_clamped = min(max(0, test_c), self.grid_size - 1)

                if test_c != test_c_clamped:  # Blocked by wall
                    self.axis_lock = 'x'
                elif test_c == player_c and self.pos[0] == player_r:  # Blocked by player
                    self.axis_lock = 'x'
                else:
                    self.pos[1] = test_c_clamped
                    return
            else:
                # Aligned on y-axis
                self.axis_lock = 'x'

#####################################
# TRAINER CAT IMPLEMENTATION        #
#####################################
# You can modify the behavior of    #
# this cat to test your learning    #
# algorithm. This cat will not part #
# of the grading.                   #
#####################################

class TrainerCat(Cat):
    """A customizable cat for students to implement and test their own behavior algorithms.
    
    This cat provides access to:
    - self.pos: Current cat position as [row, col]
    - self.player_pos: Current player position
    - self.prev_player_pos: Previous player position
    - self.last_player_action: Last action (0:Up, 1:Down, 2:Left, 3:Right)
    - self.current_distance: Current Manhattan distance to player
    - self.prev_distance: Previous Manhattan distance to player
    - self.grid_size: Size of the grid (e.g., 8 for 8x8 grid)
    
    Helper methods:
    - self.player_moved_closer(): Returns True if player's last move decreased distance
    """
    def _get_sprite_path(self) -> str:
        return "images/trainer-dp.png"
    
    def move(self) -> None:
        # Students can implement their own cat behavior here
        # This is a dummy implementation that stays still
        # You can:
        # 1. Access player information (position, last action)
        # 2. Check distances
        # 3. Implement your own movement strategy
        # 4. Test different learning algorithms
        return

#######################################
# END OF CAT BEHAVIOR IMPLEMENTATIONS #
#######################################

class CatChaseEnv(gym.Env):
    """Simple 8x8 grid world where an agent tries to catch a randomly moving cat.

    Observation: Dict with 'agent' and 'cat' positions as (row, col) each in [0..7].
    Action space: Discrete(4) -> 0:Up,1:Down,2:Left,3:Right
    Reward: +1 when agent catches cat (episode ends), -0.01 per step to encourage speed.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, grid_size: int = 8, tile_size: int = 64, cat_type: str = "peekaboo"):
        super().__init__()
        self.grid_size = grid_size
        self.tile_size = tile_size

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Discrete(grid_size * 1000 + grid_size * 100 + grid_size * 10 + grid_size)

        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()

        temp_surface = pygame.display.set_mode((1, 1))
        
        agent_img_path = "images/catbot.png"
        if not os.path.exists(agent_img_path):
            print(f"Warning: Agent image file not found: {agent_img_path}")
            self.agent_sprite = pygame.Surface((self.tile_size, self.tile_size))
            self.agent_sprite.fill((100, 200, 100))
        else:
            try:
                self.agent_sprite = pygame.image.load(agent_img_path)
                self.agent_sprite = self.agent_sprite.convert_alpha()
                self.agent_sprite = pygame.transform.scale(self.agent_sprite, (self.tile_size, self.tile_size))
                print(f"Successfully loaded agent sprite: {agent_img_path}")
            except Exception as e:
                print(f"Error loading agent sprite {agent_img_path}: {str(e)}")
                self.agent_sprite = pygame.Surface((self.tile_size, self.tile_size))
                self.agent_sprite.fill((100, 200, 100))

        cat_types = {
            "batmeow": BatmeowCat,
            "mittens": MittensCat,
            "paotsin": PaotsinCat,
            "peekaboo": PeekabooCat,
            "squiddyboi": SquiddyboiCat,
            "trainer": TrainerCat,
            # --- HIDDEN CATS ---
            "spidercat": SpiderCat,
            "cheddar": CheddarCat,
            "pumpkinpie": PumpkinPieCat,
            "milky": MilkyCat,
            "taro": TaroCat
        }
        if cat_type not in cat_types:
            raise ValueError(f"Unknown cat type: {cat_type}. Available types: {list(cat_types.keys())}")
        
        self.cat = cat_types[cat_type](grid_size, tile_size)
        
        pygame.display.quit()
        
        self.agent_pos = np.zeros(2, dtype=np.int32)
        self.agent_visual_pos = np.zeros(2, dtype=float)
        self.animation_speed = 48.0  
        self.last_render_time = time.time()
        
        self.agent_bump_offset = np.zeros(2, dtype=float)  
        self.cat_bump_offset = np.zeros(2, dtype=float)    
        self.bump_magnitude = 0.15  
        self.bump_spring = 8.0      
        
        self.done = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None) -> Tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agent_pos = np.array([0, 0], dtype=np.int32)
        cat_start_pos = np.array([self.grid_size-1, self.grid_size-1], dtype=np.int32)
        self.cat.reset(cat_start_pos)

        self.agent_visual_pos = self.agent_pos.astype(float)
        self.last_render_time = time.time()
        
        self.done = False
        return self._get_obs(), {}

    def _get_obs(self):
        return int(self.agent_pos[0] * 1000 + self.agent_pos[1] * 100 + self.cat.pos[0] * 10 + self.cat.pos[1])

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self.done:
            return self._get_obs(), 0.0, True, False, {}
        reward = 0

        old_pos = self.agent_pos.copy()
        
        if action == 0:  
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
            if self.agent_pos[0] == old_pos[0]:  
                self.agent_bump_offset[0] = -self.bump_magnitude
        elif action == 1:  
            self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
            if self.agent_pos[0] == old_pos[0]:  
                self.agent_bump_offset[0] = self.bump_magnitude
        elif action == 2:  
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
            if self.agent_pos[1] == old_pos[1]:  
                self.agent_bump_offset[1] = -self.bump_magnitude
        elif action == 3:  
            self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
            if self.agent_pos[1] == old_pos[1]:  
                self.agent_bump_offset[1] = self.bump_magnitude

        info = {}

        if np.array_equal(self.agent_pos, self.cat.pos):
            self.done = True
            return self._get_obs(), reward, True, False, info

        self.cat.update_player_info(self.agent_pos, action)
        self.cat.move()

        if np.array_equal(self.agent_pos, self.cat.pos):
            self.done = True

        return self._get_obs(), float(reward), bool(self.done), False, info

    def render(self) -> Optional[np.ndarray]:
        if self.screen is None:
            width = self.grid_size * self.tile_size
            height = self.grid_size * self.tile_size
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Cat Chase")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.screen.fill((25, 48, 15))

        for r in range(self.grid_size):
            for c in range(self.grid_size):
                rect = pygame.Rect(c * self.tile_size, r * self.tile_size, self.tile_size, self.tile_size)
                color = (35, 61, 20) if (r + c) % 2 == 0 else (25, 48, 15)
                pygame.draw.rect(self.screen, color, rect)

        current_time = time.time()
        dt = current_time - self.last_render_time
        self.last_render_time = current_time

        # 🛑 START: Draw the recorded path 🛑
        if hasattr(self, 'path_history'):
            # Drawing a gradient path from old (darker yellow) to new (lighter yellow)
            path_length = len(self.path_history)
            for i, (r, c) in enumerate(self.path_history):
                # Calculate color: Closer to the end of the path means Lighter Yellow
                # Use 'i / path_length' to get a factor between 0.0 and 1.0
                color_factor = i / path_length

                # Lighter yellow gradient: Starts at (200, 200, 0) and ends at (255, 255, 150)
                red = int(200 + 55 * color_factor)
                green = int(200 + 55 * color_factor)
                blue = int(0 + 150 * color_factor)
                color = (red, green, blue)

                # Calculate pixel position for the center of the tile
                center_x = int((c * self.tile_size) + (self.tile_size / 2))
                center_y = int((r * self.tile_size) + (self.tile_size / 2))

                radius = 3

                # Draw a circle on the screen
                pygame.draw.circle(self.screen, color, (center_x, center_y), radius)
        # 🛑 END: Draw the recorded path 🛑

        for i in range(2):
            diff = self.agent_pos[i] - self.agent_visual_pos[i]
            if abs(diff) > 0.01:
                self.agent_visual_pos[i] += np.clip(diff * self.animation_speed * dt, -1, 1)

            if abs(self.agent_bump_offset[i]) > 0.001:
                self.agent_bump_offset[i] *= max(0, 1 - self.bump_spring * dt)

        self.cat.update_visual_pos(dt, self.animation_speed)

        old_cat_pos = self.cat.pos.copy()
        if not np.array_equal(old_cat_pos, self.cat.pos):
            for i in range(2):
                if old_cat_pos[i] == self.cat.pos[i] and (
                        (old_cat_pos[i] == 0 and self.cat.pos[i] == 0) or
                        (old_cat_pos[i] == self.grid_size - 1 and self.cat.pos[i] == self.grid_size - 1)
                ):
                    self.cat_bump_offset[i] = self.bump_magnitude if old_cat_pos[
                                                                         i] == self.grid_size - 1 else -self.bump_magnitude

        for i in range(2):
            if abs(self.cat_bump_offset[i]) > 0.001:
                self.cat_bump_offset[i] *= max(0, 1 - self.bump_spring * dt)

        cat_x = (self.cat.visual_pos[1] + self.cat_bump_offset[1]) * self.tile_size
        cat_y = (self.cat.visual_pos[0] + self.cat_bump_offset[0]) * self.tile_size
        cat_rect = pygame.Rect(cat_x, cat_y, self.tile_size, self.tile_size)
        self.screen.blit(self.cat.sprite, cat_rect)

        ag_x = (self.agent_visual_pos[1] + self.agent_bump_offset[1]) * self.tile_size
        ag_y = (self.agent_visual_pos[0] + self.agent_bump_offset[0]) * self.tile_size
        ag_rect = pygame.Rect(ag_x, ag_y, self.tile_size, self.tile_size)
        self.screen.blit(self.agent_sprite, ag_rect)

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.screen = None
        pygame.quit()

def make_env(cat_type: str = "batmeow"):
    return CatChaseEnv(cat_type=cat_type)
