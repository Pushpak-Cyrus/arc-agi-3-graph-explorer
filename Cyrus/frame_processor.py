"""
Frame Processor for CYRUS
Converts raw ARC-AGI-3 frames into structured states with priority tiers.

Input: 1x64x64 discrete grid (values 0-12)
Output: structured state object + hash for deduplication
"""

import hashlib
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, Tuple, List, Optional


class FrameProcessor:
    """Processes raw game frames into structured state representations."""

    def __init__(self, game_name: str):
        """
        Initialize the frame processor.

        Args:
            game_name: identifier for the game (e.g., 'ls20', 'vc33')
        """
        self.game_name = game_name
        self.status_bar_mask = None  # will be learned from first few frames
        self.frame_history = []  # keep last N frames to detect status bar
        self.max_history = 5
        self.target_history = []

    def reset_target_history(self) -> None:
        """Forget visual target persistence when a new level begins."""
        self.target_history.clear()

    def _detect_status_bar(self, frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect status bar region (typically a fixed strip, often at bottom or side).
        Returns (start_row, end_row) or None if not detected.

        Heuristic: status bar is usually a constant band with little variation.
        """
        if len(self.frame_history) < 3:
            return None

        # Stack the last N frames and look for rows that are identical across all of them
        stacked = np.array(self.frame_history[-3:])  # shape (3, 64, 64)

        # Find rows where all 3 frames are identical
        identical_rows = []
        for row_idx in range(stacked.shape[1]):
            if np.all(stacked[0, row_idx] == stacked[1, row_idx]) and \
               np.all(stacked[1, row_idx] == stacked[2, row_idx]):
                identical_rows.append(row_idx)

        # If we find a contiguous band, that's likely the status bar
        if identical_rows:
            # Find the longest contiguous sequence
            sequences = []
            current_seq = [identical_rows[0]]
            for i in range(1, len(identical_rows)):
                if identical_rows[i] == identical_rows[i-1] + 1:
                    current_seq.append(identical_rows[i])
                else:
                    sequences.append(current_seq)
                    current_seq = [identical_rows[i]]
            sequences.append(current_seq)

            longest = max(sequences, key=len)
            if len(longest) >= 4:  # must be at least 4 rows to be significant
                return (min(longest), max(longest))

        return None

    def _mask_status_bar(self, frame: np.ndarray) -> np.ndarray:
        """
        Create a masked version of the frame with status bar zeroed out.
        """
        masked = frame.copy()

        # If we haven't detected the status bar yet, try to detect it
        if self.status_bar_mask is None and len(self.frame_history) >= 3:
            bar_region = self._detect_status_bar(frame)
            if bar_region:
                self.status_bar_mask = bar_region

        # Apply the mask if we have one
        if self.status_bar_mask:
            start, end = self.status_bar_mask
            masked[0, start:end+1, :] = 0

        return masked

    def _get_blob_priority_tier(self, blob_values: np.ndarray) -> int:
        """
        Assign a priority tier (0-4, where 0 is highest priority) to a blob
        based on its properties.

        Higher priority (lower tier number):
        - Tier 0: rare/distinctive values (9-12), likely interactive
        - Tier 1: medium values (5-8), possibly interactive
        - Tier 2: common values (2-4), less likely interactive
        - Tier 3: background-like (mostly 0-1)
        - Tier 4: status bar / UI elements (if detected)

        Args:
            blob_values: array of pixel values in this blob

        Returns:
            tier (0-4)
        """
        unique_vals = np.unique(blob_values)
        primary_val = np.argmax(np.bincount(blob_values))

        # Tier 0: high distinctive values
        if primary_val >= 9:
            return 0
        # Tier 1: medium values
        elif primary_val >= 5:
            return 1
        # Tier 2: lower-medium
        elif primary_val >= 2:
            return 2
        # Tier 3: background-ish
        else:
            return 3

    def _find_connected_components(self, frame: np.ndarray) -> Dict[int, List[Tuple[int, int]]]:
        """
        Find connected components (blobs) of the same value.
        Returns a dict mapping component_id -> list of (row, col) coordinates.

        Uses 4-connectivity (up, down, left, right).

        Iterative (stack-based) implementation — a recursive version can exceed
        Python's default recursion limit on real game frames, where a single
        large background blob can cover hundreds or thousands of cells.
        """
        visited = np.zeros_like(frame, dtype=bool)
        components = {}
        component_id = 0
        height, width = frame.shape[1], frame.shape[2]

        def flood_fill_iterative(start_r: int, start_c: int, val: int) -> List[Tuple[int, int]]:
            """Iterative flood fill using an explicit stack."""
            coords = []
            stack = [(start_r, start_c)]

            while stack:
                r, c = stack.pop()

                if r < 0 or r >= height or c < 0 or c >= width:
                    continue
                if visited[0, r, c] or frame[0, r, c] != val:
                    continue

                visited[0, r, c] = True
                coords.append((r, c))

                # 4-connectivity: up, down, left, right
                stack.append((r - 1, c))
                stack.append((r + 1, c))
                stack.append((r, c - 1))
                stack.append((r, c + 1))

            return coords

        for r in range(height):
            for c in range(width):
                if not visited[0, r, c]:
                    val = frame[0, r, c]
                    coords = flood_fill_iterative(r, c, val)
                    if coords:  # only keep non-empty
                        components[component_id] = coords
                        component_id += 1

        return components

    def _extract_blob_features(self, blob_coords: List[Tuple[int, int]],
                               frame: np.ndarray) -> Dict:
        """
        Extract features from a blob for priority ranking.

        Args:
            blob_coords: list of (row, col) coordinates in the blob
            frame: the original frame

        Returns:
            dict with keys: size, primary_value, priority_tier
        """
        blob_values = np.array([frame[0, r, c] for r, c in blob_coords])
        primary_val = np.argmax(np.bincount(blob_values))
        size = len(blob_coords)
        tier = self._get_blob_priority_tier(blob_values)

        return {
            'size': size,
            'primary_value': int(primary_val),
            'priority_tier': tier,
            'coords': blob_coords,
        }

    @staticmethod
    def _group_touching_blobs(blobs: List[Dict]) -> List[Dict]:
        """Combine adjacent color components into multi-color targets.

        Same-color connected components are useful low-level observations,
        but a single outlined object can contain many colors and therefore
        become many click candidates. Merge non-dominant components that
        touch by an edge or corner. Keep the largest component separate as a
        likely board/background region and preserve it as a low-priority
        fallback target.
        """
        if len(blobs) <= 1:
            return [dict(blob, member_blob_count=1,
                         member_blob_indices=[idx],
                         is_background_candidate=(len(blobs) == 1),
                         centroid=FrameProcessor._centroid(blob['coords']),
                         persistent_frames=1)
                    for idx, blob in enumerate(blobs)]

        background_idx = max(range(len(blobs)),
                             key=lambda idx: blobs[idx]['size'])
        merge_indices = [idx for idx in range(len(blobs))
                         if idx != background_idx]
        parent = {idx: idx for idx in merge_indices}

        def find(idx):
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(left, right):
            root_left = find(left)
            root_right = find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        pixel_owner = {}
        for idx in merge_indices:
            for row, col in blobs[idx]['coords']:
                pixel_owner[(row, col)] = idx

        for idx in merge_indices:
            for row, col in blobs[idx]['coords']:
                for row_delta in (-1, 0, 1):
                    for col_delta in (-1, 0, 1):
                        if row_delta == 0 and col_delta == 0:
                            continue
                        neighbor = pixel_owner.get(
                            (row + row_delta, col + col_delta)
                        )
                        if neighbor is not None and neighbor != idx:
                            union(idx, neighbor)

        components = defaultdict(list)
        for idx in merge_indices:
            components[find(idx)].append(idx)

        targets = []
        background = dict(blobs[background_idx])
        background.update({
            'member_blob_count': 1,
            'member_blob_indices': [background_idx],
            'is_background_candidate': True,
            'centroid': FrameProcessor._centroid(background['coords']),
            'persistent_frames': 1,
        })
        targets.append(background)

        for member_indices in components.values():
            members = [blobs[idx] for idx in member_indices]
            largest_member = max(members, key=lambda item: item['size'])
            coords = [point for member in members for point in member['coords']]
            targets.append({
                'size': sum(member['size'] for member in members),
                'primary_value': largest_member['primary_value'],
                'priority_tier': min(member['priority_tier'] for member in members),
                'coords': coords,
                'member_blob_count': len(members),
                'member_blob_indices': member_indices,
                'is_background_candidate': False,
                'centroid': FrameProcessor._centroid(coords),
                'persistent_frames': 1,
            })

        return targets

    @staticmethod
    def _centroid(coords: List[Tuple[int, int]]) -> Tuple[float, float]:
        if not coords:
            return (0.0, 0.0)
        return (
            sum(row for row, _ in coords) / len(coords),
            sum(col for _, col in coords) / len(coords),
        )

    def _track_target_persistence(self, targets: List[Dict]) -> None:
        """Attach consecutive-frame persistence to nearby same-class targets."""
        previous = self.target_history
        used_previous = set()
        current = [target for target in targets
                   if not target.get('is_background_candidate')]
        current.sort(key=lambda target: target['size'], reverse=True)

        for target in current:
            row, col = target['centroid']
            best_idx = None
            best_distance = None
            for idx, old in enumerate(previous):
                if idx in used_previous:
                    continue
                if target['primary_value'] != old['primary_value']:
                    continue
                old_row, old_col = old['centroid']
                distance = (row - old_row) ** 2 + (col - old_col) ** 2
                if distance > 16:
                    continue
                size_ratio = max(target['size'], old['size']) / max(
                    1, min(target['size'], old['size'])
                )
                if size_ratio > 4:
                    continue
                if best_distance is None or distance < best_distance:
                    best_idx = idx
                    best_distance = distance

            if best_idx is not None:
                used_previous.add(best_idx)
                target['persistent_frames'] = (
                    previous[best_idx]['persistent_frames'] + 1
                )

        self.target_history = [dict(target) for target in current]

    def _state_to_hash(self, frame: np.ndarray) -> str:
        """
        Convert a frame to a unique hash.

        Args:
            frame: 1x64x64 discrete grid

        Returns:
            hex string hash
        """
        # Flatten and convert to bytes
        flat = frame.astype(np.uint8).tobytes()
        return hashlib.md5(flat).hexdigest()

    def process(self, raw_frame: np.ndarray) -> Dict:
        """
        Process a raw frame into a structured state.

        Args:
            raw_frame: 1x64x64 discrete grid (values 0-12)

        Returns:
            dict with keys:
                'hash': unique state identifier
                'blobs': list of blob dicts (size, value, tier, coords)
                'grid': the masked frame
                'masked_grid': frame with status bar zeroed out
        """
        # Handle variable input shapes robustly:
        # - (64, 64): a single 2D grid -> wrap it
        # - (N, 64, 64): a sequence of N grids (some games return multiple
        #   frames per step, e.g. showing an animation) -> take the LAST
        #   one, which is the final resulting state after the action
        # - (1, 64, 64): already in the expected shape -> use as-is
        if raw_frame.ndim == 2:
            raw_frame = raw_frame.reshape(1, 64, 64)
        elif raw_frame.ndim == 3:
            if raw_frame.shape[-2:] != (64, 64):
                raise ValueError(
                    f"Unexpected frame shape {raw_frame.shape}: "
                    f"last two dimensions must be (64, 64)"
                )
            if raw_frame.shape[0] != 1:
                raw_frame = raw_frame[-1:]  # keep only the final grid, still shape (1,64,64)
        else:
            raise ValueError(f"Unexpected frame shape {raw_frame.shape}: expected 2D or 3D array")

        # Track frame history for status bar detection
        self.frame_history.append(raw_frame)
        if len(self.frame_history) > self.max_history:
            self.frame_history.pop(0)

        # Mask the status bar
        masked_frame = self._mask_status_bar(raw_frame)

        # Find connected components
        components = self._find_connected_components(masked_frame)

        # Extract features for each blob
        blobs = []
        for comp_id, coords in components.items():
            blob_features = self._extract_blob_features(coords, masked_frame)
            blobs.append(blob_features)

        # Sort blobs by priority tier (ascending, so tier 0 comes first)
        blobs.sort(key=lambda b: (b['priority_tier'], -b['size']))

        # Build a smaller set of multi-color target candidates and attach a
        # lightweight persistence signal for the System 1 selector.
        targets = self._group_touching_blobs(blobs)
        self._track_target_persistence(targets)
        for target in targets:
            for blob_idx in target['member_blob_indices']:
                blobs[blob_idx]['group_size'] = target['size']
                blobs[blob_idx]['group_persistence'] = target['persistent_frames']
                blobs[blob_idx]['is_background_candidate'] = target[
                    'is_background_candidate'
                ]

        # Keep an object-level view alongside the low-level color blobs.
        # ``targets`` merge touching color components, which makes these
        # descriptors more stable for multi-color sprites and UI controls.
        objects = []
        for target in targets:
            if target.get('is_background_candidate'):
                continue
            coords = target.get('coords', [])
            if not coords:
                continue

            rows = [row for row, _ in coords]
            cols = [col for _, col in coords]
            row0, row1 = min(rows), max(rows)
            col0, col1 = min(cols), max(cols)
            values = Counter(int(masked_frame[0, row, col]) for row, col in coords)
            objects.append({
                'bbox': (row0, col0, row1, col1),
                'center': target['centroid'],
                'width': col1 - col0 + 1,
                'height': row1 - row0 + 1,
                'area': len(coords),
                'color_counts': dict(values),
                'color_count': len(values),
                # Rank-frequency keeps the coarse class useful when palette
                # colors change between levels or games.
                'color_profile': tuple(sorted(values.values(), reverse=True)),
                'member_blob_count': target.get('member_blob_count', 1),
                # Exact occupied cells let the interaction learner distinguish
                # a click on a sprite from a click in its empty bounding-box
                # interior.
                'cells': frozenset(coords),
            })
        # Bound pairwise relation work on highly fragmented frames.
        objects.sort(key=lambda obj: (-obj['area'], obj['bbox'][0], obj['bbox'][1]))
        objects = objects[:128]
        objects.sort(key=lambda obj: (obj['bbox'][0], obj['bbox'][1], obj['area']))

        relations = []
        for left_idx, left in enumerate(objects):
            lr0, lc0, lr1, lc1 = left['bbox']
            for right_idx in range(left_idx + 1, len(objects)):
                right = objects[right_idx]
                rr0, rc0, rr1, rc1 = right['bbox']
                row_gap = max(0, rr0 - lr1 - 1, lr0 - rr1 - 1)
                col_gap = max(0, rc0 - lc1 - 1, lc0 - rc1 - 1)
                intersects = (
                    lr0 <= rr1 and rr0 <= lr1
                    and lc0 <= rc1 and rc0 <= lc1
                )
                if intersects:
                    relation = 'overlap'
                elif row_gap <= 1 and col_gap <= 1:
                    relation = 'adjacent'
                else:
                    left_center = left['center']
                    right_center = right['center']
                    scale = max(left['width'], left['height'],
                                right['width'], right['height'])
                    if np.hypot(
                        left_center[0] - right_center[0],
                        left_center[1] - right_center[1],
                    ) <= max(8, scale * 1.5):
                        relation = 'near'
                    else:
                        relation = None

                if relation:
                    relations.append({
                        'left': left_idx,
                        'right': right_idx,
                        'kind': relation,
                    })
                if abs(left['center'][0] - right['center'][0]) <= 1:
                    relations.append({
                        'left': left_idx,
                        'right': right_idx,
                        'kind': 'aligned_row',
                    })
                if abs(left['center'][1] - right['center'][1]) <= 1:
                    relations.append({
                        'left': left_idx,
                        'right': right_idx,
                        'kind': 'aligned_column',
                    })

        # Generate hash
        state_hash = self._state_to_hash(masked_frame)

        return {
            'hash': state_hash,
            'blobs': blobs,
            'targets': targets,
            'objects': objects,
            'relations': relations,
            'grid': raw_frame,
            'masked_grid': masked_frame,
        }


# Example usage
if __name__ == "__main__":
    # Test with a dummy frame
    test_frame = np.random.randint(0, 13, size=(1, 64, 64))
    processor = FrameProcessor("test_game")

    state = processor.process(test_frame)
    print(f"State hash: {state['hash']}")
    print(f"Number of blobs: {len(state['blobs'])}")
    for i, blob in enumerate(state['blobs'][:5]):
        print(f"  Blob {i}: size={blob['size']}, value={blob['primary_value']}, tier={blob['priority_tier']}")
