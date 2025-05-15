import argparse
import json
import os
import pygame
from pygame.locals import *
import logging
from datetime import timedelta
from environment import CleanUpEnv

# Playback settings
DEFAULT_FPS = 2.5
CELL_SIZE = 40
LEGEND_HEIGHT = 450  
LEGEND_WIDTH = 1000  
WINDOW_PADDING = 20  
APPLE_SIZE = 12    
SCROLLBAR_WIDTH = 20
SCROLL_SPEED = 20

# Colors for rendering
COLORS = {
    'WHITE': (255, 255, 255),
    'BLACK': (0, 0, 0),
    'ORCHARD': (0, 200, 0),    # 00c800
    'WATER': (0, 0, 200),      # 0000c8
    'APPLE': (255, 3, 3),      # ff0303
    'GRID': (0, 0, 0),         # 000000
    'RED': (255, 0, 0),        # ff0000
    'BROWN': (139, 69, 19),
    'DARKBROWN': (125, 100, 10),
    'POLLUTION': (138, 138, 138),  # #8a8a8a
    'GRAY': (180, 180, 180),   # Slightly darker gray
    'YELLOW': (251, 255, 20),  # fbff14
    'PURPLE': (255, 0, 255),
    'CYAN': (92, 154, 255),    # 5c9aff (used for blue agent)
    'BACKGROUND': (240, 240, 240)  # Lighter background for better contrast
}

# Agent colours - increased saturation and brightness for better contrast against tiles
AGENT_COLORS = [
    COLORS['RED'],         # Red agent: ff0000
    COLORS['CYAN'],        # Blue agent: 5c9aff
    (0, 255, 0),           # Green agent: 00ff00
    COLORS['YELLOW'],      # Yellow agent: fbff14
]

class ReplayViewer:
    """Visualize experiment logs as a replay in Pygame."""
    def __init__(self, experiment_dir: str, fps: int = DEFAULT_FPS):
        self.experiment_dir = experiment_dir
        self.fps = fps
        self.config = self._load_json('config.json')
        self.trial_logs = self._load_trial_logs()
        self.paused = True
        self.current_step = 0
        self.current_trial = 0
        self.scroll_offset = 0
        self.max_scroll = 0
        self.agent_avatar = None  # Will be loaded after display is initialized
        self.use_avatars = True  # Toggle for avatar/rectangle rendering
        # Initialize legend dimensions
        width = self.config.get('width', 10)
        height = self.config.get('height', 8)
        window_width = width * CELL_SIZE + LEGEND_WIDTH + WINDOW_PADDING + SCROLLBAR_WIDTH
        window_height = height * CELL_SIZE + LEGEND_HEIGHT
        self._update_dimensions(window_width, window_height)

    def _load_json(self, filename: str) -> dict:
        path = os.path.join(self.experiment_dir, filename)
        if not os.path.exists(path):
            logging.error("Missing config file: %s", path)
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_trial_logs(self) -> list:
        logs = []
        for fname in sorted(os.listdir(self.experiment_dir)):
            if fname.startswith('trial_') and fname.endswith('.json'):
                with open(os.path.join(self.experiment_dir, fname), 'r') as f:
                    logs.append(json.load(f))
        if not logs:
            logging.warning("No trial logs found in %s", self.experiment_dir)
        return logs

    def run(self):
        pygame.init()
        width = self.config.get('width', 10)
        height = self.config.get('height', 8)
        grid_pixel_height = height * CELL_SIZE
        screen = pygame.display.set_mode((width * CELL_SIZE + LEGEND_WIDTH + WINDOW_PADDING + SCROLLBAR_WIDTH, 
                                       height * CELL_SIZE + LEGEND_HEIGHT),
                                       pygame.RESIZABLE)
        pygame.display.set_caption("CleanUp Game Replay")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 24)
        
        # Calculate vertical offset to center the grid
        window_height = screen.get_height()
        self.grid_vertical_offset = (window_height - grid_pixel_height) // 2

        # Load agent avatar image after display is initialized
        try:
            avatar_path = os.path.join("assets", "agent_avatar.png")
            if os.path.exists(avatar_path):
                self.agent_avatar = pygame.image.load(avatar_path).convert_alpha()
                logging.info("Successfully loaded agent avatar from %s", avatar_path)
            else:
                logging.warning("Agent avatar image not found at %s. Falling back to rectangles.", avatar_path)
        except Exception as e:
            logging.error("Failed to load agent avatar: %s. Falling back to rectangles.", e)
            self.agent_avatar = None

        print("Controls:")
        print("SPACE: Pause/Resume")
        print("RIGHT: Next step")
        print("LEFT: Previous step")
        print("A: Toggle avatar/rectangle rendering")
        print("Q: Quit")
        print("Mouse Wheel: Scroll legend")
        print("Window: Resizable")
        print("\nReplay is paused. Press SPACE to start.")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        print("Replay paused" if self.paused else "Replay resumed")
                    elif event.key == pygame.K_RIGHT:
                        self.paused = True
                        self._next_step()
                    elif event.key == pygame.K_LEFT:
                        self.paused = True
                        self._prev_step()
                    elif event.key == pygame.K_a:
                        self.use_avatars = not self.use_avatars
                        print(f"Avatar rendering {'enabled' if self.use_avatars else 'disabled'}.")
                elif event.type == pygame.MOUSEWHEEL:
                    # Scroll up/down
                    self.scroll_offset = max(0, min(self.scroll_offset - event.y * SCROLL_SPEED, self.max_scroll))
                elif event.type == pygame.VIDEORESIZE:
                    # Update screen size
                    screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                    # Recalculate legend dimensions
                    self._update_dimensions(event.w, event.h)

            if not self.paused:
                self._next_step()

            # Draw current state
            screen.fill(COLORS['BACKGROUND'])
            trial = self.trial_logs[self.current_trial]
            step = trial['steps'][self.current_step]
            
            # Create and update environment state
            env = CleanUpEnv(config=self.config)
            env.reset()
            env.apples = set(tuple(a) for a in step['pre_state'].get('apples', []))
            env.pollution = set(tuple(p) for p in step['pre_state'].get('pollution', []))
            for aid, info in step['pre_state']['agents'].items():
                env.agents[int(aid)].update(info)

            self._render_state(screen, env, font)
            self._render_legend(screen, trial, step)
            self._render_scrollbar(screen)
            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

    def _next_step(self):
        trial = self.trial_logs[self.current_trial]
        if self.current_step < len(trial['steps']) - 1:
            self.current_step += 1
        elif self.current_trial < len(self.trial_logs) - 1:
            self.current_trial += 1
            self.current_step = 0
        else:
            self.paused = True

    def _prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
        elif self.current_trial > 0:
            self.current_trial -= 1
            self.current_step = len(self.trial_logs[self.current_trial]['steps']) - 1

    def _render_state(self, screen, env, font):
        # Draw grid area
        grid_height = self.config.get('height', 8) * CELL_SIZE
        grid_width = self.config.get('width', 10) * CELL_SIZE
        offset_y = getattr(self, 'grid_vertical_offset', 0)
        grid_area = pygame.Rect(0, offset_y, grid_width, grid_height)
        # Draw border (thicker, professional look)
        pygame.draw.rect(screen, COLORS['BLACK'], grid_area, border_radius=8)
        pygame.draw.rect(screen, COLORS['WHITE'], grid_area.inflate(-4, -4), border_radius=8)
        pygame.draw.rect(screen, COLORS['BLACK'], grid_area, 4, border_radius=8)  # Outer border
        
        # Draw grid cells
        for r in range(self.config.get('height', 8)):
            for c in range(self.config.get('width', 10)):
                rect = pygame.Rect(c * CELL_SIZE, offset_y + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                color = COLORS['ORCHARD'] if c in env.orchard_cols else COLORS['WATER'] if c in env.river_cols else COLORS['GRAY']
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, COLORS['GRID'], rect, 1)
                
                # Draw apple if present
                if (r, c) in env.apples:
                    # Draw apple with black border
                    pygame.draw.circle(screen, COLORS['BLACK'], rect.center, APPLE_SIZE + 2)
                    pygame.draw.circle(screen, COLORS['APPLE'], rect.center, APPLE_SIZE)
                
                # Draw pollution if present
                if (r, c) in env.pollution:
                    pygame.draw.rect(screen, COLORS['POLLUTION'], rect.inflate(-10, -10))

        # Draw agents
        for aid, info in env.agents.items():
            pos = info['pos']
            agent_r, agent_c = pos
            agent_color = AGENT_COLORS[aid % len(AGENT_COLORS)]
            
            if not info['active']: # If agent is inactive (zapped)
                # Use original color but indicate zapped status
                agent_color = COLORS['GRAY']

            if self.use_avatars and self.agent_avatar:
                # Create a copy of the avatar to tint
                tinted_avatar = self.agent_avatar.copy()
                
                if info['active']:
                    # Apply tint if active:
                    # This mode multiplies the avatar's RGB with the agent_color.
                    # Works best if avatar is white/grayscale.
                    tinted_avatar.fill(agent_color, special_flags=BLEND_RGB_MULT)
                else:
                    # For inactive (zapped) agents, we might want a grayscale version
                    # or just a desaturated look. For now, let's make it semi-transparent
                    # or use the gray color directly on a non-tinted avatar if it looks better.
                    # Let's try tinting with GRAY for zapped state as well.
                    tinted_avatar.fill(COLORS['GRAY'], special_flags=BLEND_RGB_MULT)

                # Scale the avatar (e.g., slightly smaller than the cell)
                avatar_size = (int(CELL_SIZE * 0.8), int(CELL_SIZE * 0.8))
                scaled_avatar = pygame.transform.smoothscale(tinted_avatar, avatar_size)
                
                # Calculate position to blit the avatar (centered in the cell)
                avatar_rect = scaled_avatar.get_rect(center=(agent_c * CELL_SIZE + CELL_SIZE // 2, 
                                                              offset_y + agent_r * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(scaled_avatar, avatar_rect)

            else: # Fallback to drawing a rectangle if avatar couldn't be loaded or toggled off
                rect_to_draw = pygame.Rect(agent_c * CELL_SIZE, offset_y + agent_r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, agent_color, rect_to_draw.inflate(-10, -10))

            # Render score text (black font, no background)
            smaller_font = pygame.font.SysFont(None, 20)
            score_text = smaller_font.render(f"{info['score']}", True, COLORS['BLACK'])
            if self.use_avatars and self.agent_avatar:
                # Center bottom for avatars (as before)
                text_rect = score_text.get_rect(midbottom=(agent_c * CELL_SIZE + CELL_SIZE // 2,
                                                          offset_y + agent_r * CELL_SIZE + CELL_SIZE - 2))
            else:
                # Centered for rectangles
                text_rect = score_text.get_rect(center=(agent_c * CELL_SIZE + CELL_SIZE // 2,
                                                       offset_y + agent_r * CELL_SIZE + CELL_SIZE // 2))
            screen.blit(score_text, text_rect)

    def _render_scrollbar(self, screen):
        if self.max_scroll <= 0:
            return

        # Calculate scrollbar dimensions
        grid_width = self.config.get('width', 10) * CELL_SIZE
        x = grid_width + self.legend_width + WINDOW_PADDING
        y = WINDOW_PADDING
        height = self.legend_height
        
        # Draw scrollbar track
        track_rect = pygame.Rect(x, y, SCROLLBAR_WIDTH, height)
        pygame.draw.rect(screen, COLORS['GRAY'], track_rect)
        
        # Calculate thumb size and position
        thumb_height = max(30, int(height * (height / self.max_scroll)))
        thumb_y = y + (height - thumb_height) * (self.scroll_offset / self.max_scroll)
        thumb_rect = pygame.Rect(x, thumb_y, SCROLLBAR_WIDTH, thumb_height)
        pygame.draw.rect(screen, COLORS['BLACK'], thumb_rect)

    def _render_legend(self, screen, trial, step):
        font = pygame.font.SysFont(None, 24)
        grid_width = self.config.get('width', 10) * CELL_SIZE
        y = WINDOW_PADDING - self.scroll_offset  # Apply scroll offset
        x = grid_width + WINDOW_PADDING
        
        # Draw trial and step info
        texts = [
            f"Trial: {trial.get('trial_number', 'N/A')}",
            f"Step: {step['step']}",
            f"Status: {'PAUSED' if self.paused else 'PLAYING'}",
            ""  # Empty line for spacing
        ]
        
        # Draw agent info
        for aid, info in step['pre_state']['agents'].items():
            aid = int(aid)
            provider = trial['providers'][aid]
            model_info = f"/{trial['models'][aid]}" if 'models' in trial else ""
            action = step['actions'].get(str(aid), 'stay')
            plan = step['plans'].get(str(aid), '')
            score = info['score']
            active = info['active']
            
            # Create color box for agent
            color = AGENT_COLORS[aid % len(AGENT_COLORS)]
            color_box = pygame.Surface((20, 20))
            color_box.fill(color)
            
            texts.extend([
                f"Agent {aid} ({provider}{model_info}):",
                f"  Score: {score}",
                f"  Status: {'Active' if active else 'Inactive'}",
                f"  Action: {action}",
                f"  Plan:"
            ])
            
            # Render colour box
            screen.blit(color_box, (x, y + (len(texts) - 5) * 20))
            
            # Word wrap the plan
            words = plan.split()
            current_line = "    "  # 4 spaces for indentation
            for word in words:
                test_line = current_line + word + " "
                # Check if adding this word would exceed the width
                if font.size(test_line)[0] < self.legend_width - 50:  # 50px margin
                    current_line = test_line
                else:
                    texts.append(current_line)
                    current_line = "    " + word + " "
            if current_line.strip():
                texts.append(current_line)
            
            # Add spacing between agents
            texts.append("")

        # Calculate total height of text
        total_height = len(texts) * 20
        self.max_scroll = max(0, total_height - self.legend_height)

        # Render all text
        for i, txt in enumerate(texts):
            # Only render text that's visible in the legend area
            text_y = y + i * 20
            if WINDOW_PADDING <= text_y <= self.legend_height + WINDOW_PADDING:
                surf = font.render(txt, True, COLORS['BLACK'])
                screen.blit(surf, (x + 30, text_y))  # Added 30px offset for color box

    def _update_dimensions(self, window_width, window_height):
        """Update dimensions based on window size."""
        self.legend_width = window_width - (self.config.get('width', 10) * CELL_SIZE + WINDOW_PADDING + SCROLLBAR_WIDTH)
        self.legend_height = window_height - 2 * WINDOW_PADDING

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replay experiment logs')
    parser.add_argument('experiment_dir', help='Directory with experiment logs')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS, help='Playback FPS')
    args = parser.parse_args()

    viewer = ReplayViewer(args.experiment_dir, fps=args.fps)
    viewer.run()
