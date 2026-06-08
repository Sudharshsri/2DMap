"""
Synthetic Floor Plan Dataset Generator
=======================================
Generates training data for the U-Net floor plan cleaning model.

Ground Truth Classes:
    0 = Background (outside the building)
    1 = Walkable Space (rooms, corridors)  
    2 = Wall

The generator creates realistic indoor layouts including:
    - Rectangular rooms of varying sizes
    - Long corridors connecting rooms
    - L-shaped and T-shaped junctions
    - Doors (gaps in walls between rooms)

The noisy input simulates what a real SLAM occupancy grid looks like:
    - Salt & pepper noise (sensor errors)
    - Missing regions (unexplored areas)
    - Gaussian blur (uncertainty in wall positions)
    - Random dropout patches
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2


class SyntheticFloorPlanDataset(Dataset):
    """
    Generates pairs of (noisy_occupancy_grid, clean_floor_plan_mask).
    
    Args:
        size: Image dimensions (size x size pixels)
        length: Number of samples in the dataset
        seed: Random seed for reproducibility (None for random)
    """
    
    def __init__(self, size=256, length=2000, seed=None):
        self.size = size
        self.length = length
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.length

    def _draw_room(self, mask, x, y, w, h):
        """Draw a rectangular room onto the mask."""
        x1 = max(0, min(self.size - 1, x))
        y1 = max(0, min(self.size - 1, y))
        x2 = max(0, min(self.size - 1, x + w))
        y2 = max(0, min(self.size - 1, y + h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return [x1, y1, x2, y2]

    def _draw_corridor(self, mask, start_x, start_y, end_x, end_y, width=8):
        """Draw a corridor (thick line) between two points."""
        # Draw horizontal then vertical (L-shaped path)
        min_x = min(start_x, end_x)
        max_x = max(start_x, end_x)
        min_y = min(start_y, end_y)
        max_y = max(start_y, end_y)
        
        half_w = width // 2
        
        # Horizontal segment at start_y
        hy1 = max(0, start_y - half_w)
        hy2 = min(self.size, start_y + half_w)
        hx1 = max(0, min_x)
        hx2 = min(self.size, max_x)
        if hy2 > hy1 and hx2 > hx1:
            mask[hy1:hy2, hx1:hx2] = 1
        
        # Vertical segment at end_x
        vy1 = max(0, min_y)
        vy2 = min(self.size, max_y)
        vx1 = max(0, end_x - half_w)
        vx2 = min(self.size, end_x + half_w)
        if vy2 > vy1 and vx2 > vx1:
            mask[vy1:vy2, vx1:vx2] = 1

    def _generate_floor_plan(self):
        """
        Generate a realistic floor plan with rooms and corridors.
        
        Returns:
            np.ndarray of shape (size, size) with values {0, 1, 2}
        """
        gt_mask = np.zeros((self.size, self.size), dtype=np.uint8)
        
        # Decide layout type
        layout_type = self.rng.choice(['rooms_with_corridors', 'long_corridor', 'open_plan'])
        
        if layout_type == 'rooms_with_corridors':
            # Generate 3-6 rooms connected by corridors
            num_rooms = self.rng.integers(3, 7)
            room_rects = []
            
            # First room near center
            w = self.rng.integers(35, 70)
            h = self.rng.integers(35, 70)
            x = self.size // 2 - w // 2 + self.rng.integers(-30, 30)
            y = self.size // 2 - h // 2 + self.rng.integers(-30, 30)
            rect = self._draw_room(gt_mask, x, y, w, h)
            room_rects.append(rect)
            
            for _ in range(num_rooms - 1):
                w = self.rng.integers(25, 65)
                h = self.rng.integers(25, 65)
                
                # Place relative to a random existing room
                base = room_rects[self.rng.integers(0, len(room_rects))]
                base_cx = (base[0] + base[2]) // 2
                base_cy = (base[1] + base[3]) // 2
                
                # Random offset
                dx = self.rng.integers(-120, 120)
                dy = self.rng.integers(-120, 120)
                
                new_x = base_cx + dx - w // 2
                new_y = base_cy + dy - h // 2
                
                rect = self._draw_room(gt_mask, new_x, new_y, w, h)
                room_rects.append(rect)
                
                # Connect with a corridor
                corridor_width = self.rng.integers(6, 14)
                self._draw_corridor(
                    gt_mask, 
                    base_cx, base_cy,
                    (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2,
                    width=corridor_width
                )
        
        elif layout_type == 'long_corridor':
            # A long winding corridor with rooms branching off
            corridor_width = self.rng.integers(8, 16)
            half_w = corridor_width // 2
            
            # Main corridor: vertical or horizontal
            if self.rng.random() < 0.5:
                # Horizontal main corridor
                cy = self.size // 2 + self.rng.integers(-30, 30)
                x_start = self.rng.integers(10, 40)
                x_end = self.size - self.rng.integers(10, 40)
                gt_mask[cy - half_w:cy + half_w, x_start:x_end] = 1
                
                # Branch rooms off the corridor
                num_branches = self.rng.integers(3, 7)
                for _ in range(num_branches):
                    bx = self.rng.integers(x_start + 10, x_end - 10)
                    side = self.rng.choice([-1, 1])  # above or below
                    
                    room_w = self.rng.integers(20, 50)
                    room_h = self.rng.integers(20, 50)
                    
                    if side == -1:
                        ry = cy - half_w - room_h
                    else:
                        ry = cy + half_w
                    
                    self._draw_room(gt_mask, bx - room_w // 2, ry, room_w, room_h)
                    # Short connecting corridor
                    self._draw_corridor(
                        gt_mask, bx, cy, bx, ry + room_h // 2, width=corridor_width
                    )
            else:
                # Vertical main corridor
                cx = self.size // 2 + self.rng.integers(-30, 30)
                y_start = self.rng.integers(10, 40)
                y_end = self.size - self.rng.integers(10, 40)
                gt_mask[y_start:y_end, cx - half_w:cx + half_w] = 1
                
                num_branches = self.rng.integers(3, 7)
                for _ in range(num_branches):
                    by = self.rng.integers(y_start + 10, y_end - 10)
                    side = self.rng.choice([-1, 1])
                    
                    room_w = self.rng.integers(20, 50)
                    room_h = self.rng.integers(20, 50)
                    
                    if side == -1:
                        rx = cx - half_w - room_w
                    else:
                        rx = cx + half_w
                    
                    self._draw_room(gt_mask, rx, by - room_h // 2, room_w, room_h)
                    self._draw_corridor(
                        gt_mask, cx, by, rx + room_w // 2, by, width=corridor_width
                    )
        
        else:  # open_plan
            # Large open area with internal partitions
            margin = self.rng.integers(20, 50)
            gt_mask[margin:self.size - margin, margin:self.size - margin] = 1
            
            # Add internal walls (partitions)
            num_partitions = self.rng.integers(2, 5)
            for _ in range(num_partitions):
                if self.rng.random() < 0.5:
                    # Horizontal partition
                    py = self.rng.integers(margin + 20, self.size - margin - 20)
                    px_start = self.rng.integers(margin, self.size // 2)
                    px_end = self.rng.integers(self.size // 2, self.size - margin)
                    thickness = self.rng.integers(2, 5)
                    gt_mask[py:py + thickness, px_start:px_end] = 0
                else:
                    # Vertical partition
                    px = self.rng.integers(margin + 20, self.size - margin - 20)
                    py_start = self.rng.integers(margin, self.size // 2)
                    py_end = self.rng.integers(self.size // 2, self.size - margin)
                    thickness = self.rng.integers(2, 5)
                    gt_mask[py_start:py_end, px:px + thickness] = 0
        
        # Extract walls: boundary pixels of the walkable region
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(gt_mask, kernel, iterations=1)
        eroded = cv2.erode(gt_mask, kernel, iterations=1)
        walls = dilated - eroded
        gt_mask[walls > 0] = 2
        
        return gt_mask

    def _simulate_slam_noise(self, gt_mask):
        """
        Simulate what a SLAM occupancy grid actually looks like.
        
        Input values: -1 (unknown), 0 (free space), 1 (occupied/wall)
        """
        obs_grid = np.full((self.size, self.size), -1.0, dtype=np.float32)
        
        # Free space
        free_mask = (gt_mask == 1)
        obs_grid[free_mask] = 0.0
        
        # Occupied (walls)
        occ_mask = (gt_mask == 2)
        obs_grid[occ_mask] = 1.0
        
        # --- Noise Types ---
        
        # 1. Salt & pepper noise (sensor misreadings)
        noise_level = self.rng.uniform(0.03, 0.08)
        noise = self.rng.random((self.size, self.size))
        known_mask = obs_grid != -1
        obs_grid[(noise < noise_level / 2) & known_mask] = 1.0
        obs_grid[(noise > 1 - noise_level / 2) & known_mask] = 0.0
        
        # 2. Random dropout patches (unexplored regions)
        num_patches = self.rng.integers(1, 5)
        for _ in range(num_patches):
            cx = self.rng.integers(0, self.size)
            cy = self.rng.integers(0, self.size)
            rx = self.rng.integers(10, 50)
            ry = self.rng.integers(10, 50)
            y, x = np.ogrid[:self.size, :self.size]
            patch_mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1
            obs_grid[patch_mask] = -1.0
        
        # 3. Pixel-level dropout
        dropout_rate = self.rng.uniform(0.05, 0.20)
        dropout = self.rng.random((self.size, self.size)) < dropout_rate
        obs_grid[dropout] = -1.0
        
        # 4. Gaussian blur on the known region (wall position uncertainty)
        if self.rng.random() < 0.5:
            blur_kernel = self.rng.choice([3, 5])
            blurred = cv2.GaussianBlur(obs_grid, (blur_kernel, blur_kernel), 0)
            # Only apply blur to known regions
            known = obs_grid != -1
            obs_grid[known] = blurred[known]
        
        return obs_grid

    def __getitem__(self, idx):
        gt_mask = self._generate_floor_plan()
        obs_grid = self._simulate_slam_noise(gt_mask)
        
        # Convert to tensors
        x = torch.from_numpy(obs_grid).unsqueeze(0).float()  # (1, H, W)
        y = torch.from_numpy(gt_mask).long()                  # (H, W)
        
        return x, y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    dataset = SyntheticFloorPlanDataset(length=6, seed=42)
    
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    
    for i in range(6):
        x, y = dataset[i]
        
        axes[0][i].imshow(x.squeeze().numpy(), cmap='gray', vmin=-1, vmax=1)
        axes[0][i].set_title(f'Noisy Input #{i+1}')
        axes[0][i].axis('off')
        
        axes[1][i].imshow(y.numpy(), cmap='viridis', vmin=0, vmax=2)
        axes[1][i].set_title(f'Ground Truth #{i+1}')
        axes[1][i].axis('off')
    
    plt.suptitle('Synthetic Floor Plan Dataset Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=150)
    print("Saved dataset_samples.png")
