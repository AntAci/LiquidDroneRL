"""
Lightweight 3D geometry definitions for the Drone simulation.

This module does NOT perform physics or rendering. It only defines:
- Drone 3D model (procedural)
- Obstacle pillars
- Wind arrow geometry
- Camera representation
- Scene container that transforms env state into a renderable geometry packet

Usage:
- The renderer (Phase C/D) should consume the JSON-style packet returned by
  Scene3D.generate_scene_packet().

Notes:
- Depends only on numpy (no OpenGL/renderer code here).
- Coordinates are in a right-handed system with z-up convention.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np


def _rotation_matrix_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Build a rotation matrix from roll (X), pitch (Y), yaw (Z) Euler angles (radians).
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    """
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cx, -sx],
                   [0, sx, cx]], dtype=np.float32)
    Ry = np.array([[cy, 0, sy],
                   [0, 1, 0],
                   [-sy, 0, cy]], dtype=np.float32)
    Rz = np.array([[cz, -sz, 0],
                   [sz, cz, 0],
                   [0, 0, 1]], dtype=np.float32)
    return (Rz @ Ry @ Rx).astype(np.float32)


class DroneModel3D:
    """
    Procedural drone model composed of:
    - central cube body
    - 4 arms (thin boxes) along +/-X and +/-Y
    - optional simple propeller discs (flat n-gons)

    Geometry is stored in local coordinates (centered at origin).
    World-space transforms are applied via scale, rotation, position.
    """
    def __init__(self,
                 body_size: float = 0.2,
                 arm_length: float = 0.4,
                 arm_width: float = 0.05,
                 arm_thickness: float = 0.02,
                 prop_radius: float = 0.09,
                 prop_segments: int = 12,
                 color: str = "#00aaff",
                 scale: float = 1.0) -> None:
        self.color = color
        self.scale = float(scale)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Build local geometry
        body_v, body_f = self._make_box(size=body_size)
        arms_v, arms_f = self._make_arms(arm_length, arm_width, arm_thickness)
        props_v, props_f = self._make_propellers(prop_radius, prop_segments, arm_length)

        # Merge components: concatenate vertices and adjust face indices
        self.vertices_local = body_v
        self.faces = body_f.copy()

        offset = self.vertices_local.shape[0]
        self.vertices_local = np.vstack([self.vertices_local, arms_v])
        self.faces += [(a + offset, b + offset, c + offset) for (a, b, c) in arms_f]

        offset = self.vertices_local.shape[0]
        self.vertices_local = np.vstack([self.vertices_local, props_v])
        self.faces += [(a + offset, b + offset, c + offset) for (a, b, c) in props_f]

        # Optionally, per-face colors (uniform for simplicity)
        self.face_colors = [self.color] * len(self.faces)

    @staticmethod
    def _make_box(size: float) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        s = float(size) * 0.5
        v = np.array([
            [-s, -s, -s],
            [ s, -s, -s],
            [ s,  s, -s],
            [-s,  s, -s],
            [-s, -s,  s],
            [ s, -s,  s],
            [ s,  s,  s],
            [-s,  s,  s],
        ], dtype=np.float32)
        f = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 4, 7), (0, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (3, 2, 6), (3, 6, 7),
            (0, 1, 5), (0, 5, 4)
        ]
        return v, f

    @staticmethod
    def _make_box_centered(dx: float, dy: float, dz: float) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        hx, hy, hz = dx / 2.0, dy / 2.0, dz / 2.0
        v = np.array([
            [-hx, -hy, -hz],
            [ hx, -hy, -hz],
            [ hx,  hy, -hz],
            [-hx,  hy, -hz],
            [-hx, -hy,  hz],
            [ hx, -hy,  hz],
            [ hx,  hy,  hz],
            [-hx,  hy,  hz],
        ], dtype=np.float32)
        f = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 4, 7), (0, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (3, 2, 6), (3, 6, 7),
            (0, 1, 5), (0, 5, 4)
        ]
        return v, f

    def _make_arms(self, length: float, width: float, thickness: float) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        # Create four thin boxes: along +X, -X, +Y, -Y
        arm_v_all: List[np.ndarray] = []
        arm_f_all: List[Tuple[int, int, int]] = []

        # Local function to translate a box vertices
        def translate(v: np.ndarray, t: Tuple[float, float, float]) -> np.ndarray:
            return (v + np.array(t, dtype=np.float32))

        # Arm along +X
        av, af = self._make_box_centered(length, width, thickness)
        av = translate(av, (length / 2.0, 0.0, 0.0))
        arm_v_all.append(av)
        arm_f_all += af
        offset = av.shape[0]

        # Arm along -X
        av2, af2 = self._make_box_centered(length, width, thickness)
        av2 = translate(av2, (-length / 2.0, 0.0, 0.0))
        arm_v_all.append(av2)
        arm_f_all += [(a + offset, b + offset, c + offset) for (a, b, c) in af2]
        offset += av2.shape[0]

        # Arm along +Y
        av3, af3 = self._make_box_centered(width, length, thickness)
        av3 = translate(av3, (0.0, length / 2.0, 0.0))
        arm_v_all.append(av3)
        arm_f_all += [(a + offset, b + offset, c + offset) for (a, b, c) in af3]
        offset += av3.shape[0]

        # Arm along -Y
        av4, af4 = self._make_box_centered(width, length, thickness)
        av4 = translate(av4, (0.0, -length / 2.0, 0.0))
        arm_v_all.append(av4)
        arm_f_all += [(a + offset, b + offset, c + offset) for (a, b, c) in af4]

        verts = np.vstack(arm_v_all).astype(np.float32)
        return verts, arm_f_all

    @staticmethod
    def _make_disc(radius: float, segments: int, z: float = 0.0) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        seg = max(3, int(segments))
        center = np.array([[0.0, 0.0, z]], dtype=np.float32)
        thetas = np.linspace(0.0, 2.0 * np.pi, seg, endpoint=False)
        ring = np.stack([radius * np.cos(thetas), radius * np.sin(thetas), np.full_like(thetas, z)], axis=1).astype(np.float32)
        verts = np.vstack([center, ring])  # 0 is center
        faces: List[Tuple[int, int, int]] = []
        for i in range(seg):
            a = 0
            b = 1 + i
            c = 1 + ((i + 1) % seg)
            faces.append((a, b, c))
        return verts, faces

    def _make_propellers(self, radius: float, segments: int, arm_length: float) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        # Four discs placed at the ends of arms (+/-X, +/-Y)
        pv_all: List[np.ndarray] = []
        pf_all: List[Tuple[int, int, int]] = []
        positions = [
            ( arm_length, 0.0, 0.0),
            (-arm_length, 0.0, 0.0),
            (0.0,  arm_length, 0.0),
            (0.0, -arm_length, 0.0),
        ]
        offset = 0
        for (tx, ty, tz) in positions:
            dv, df = self._make_disc(radius, segments, z=0.0)
            dv = dv + np.array([tx, ty, tz], dtype=np.float32)
            pv_all.append(dv)
            pf_all += [(a + offset, b + offset, c + offset) for (a, b, c) in df]
            offset += dv.shape[0]
        verts = np.vstack(pv_all).astype(np.float32)
        return verts, pf_all

    # ------------- Transforms -------------
    def set_position(self, x: float, y: float, z: float) -> None:
        self.position = np.array([x, y, z], dtype=np.float32)

    def set_rotation(self, roll: float, pitch: float, yaw: float) -> None:
        self.roll = float(roll)
        self.pitch = float(pitch)
        self.yaw = float(yaw)

    def get_transformed_vertices(self) -> np.ndarray:
        """
        Apply scale, rotation, and translation to local vertices and return world-space vertices.
        """
        s = float(self.scale)
        v = self.vertices_local * s
        R = _rotation_matrix_xyz(self.roll, self.pitch, self.yaw)
        v_rot = (v @ R.T).astype(np.float32)
        v_world = v_rot + self.position
        return v_world


class Obstacle3D:
    """
    Vertical pillar obstacle represented as a box extruded in +Z.
    """
    def __init__(self,
                 base_position: Tuple[float, float],
                 height: float = 1.0,
                 width: float = 0.1,
                 depth: float = 0.1,
                 color: str = "#444444") -> None:
        self.base_position = (float(base_position[0]), float(base_position[1]))
        self.height = float(height)
        self.width = float(width)
        self.depth = float(depth)
        self.color = color

    def get_vertices_3d(self) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Return vertices and faces for a pillar box centered at (x, y),
        from z=0 to z=height.
        """
        x, y = self.base_position
        hx = self.width / 2.0
        hy = self.depth / 2.0
        z0, z1 = 0.0, self.height
        v = np.array([
            [x - hx, y - hy, z0],
            [x + hx, y - hy, z0],
            [x + hx, y + hy, z0],
            [x - hx, y + hy, z0],
            [x - hx, y - hy, z1],
            [x + hx, y - hy, z1],
            [x + hx, y + hy, z1],
            [x - hx, y + hy, z1],
        ], dtype=np.float32)
        f = [
            (0, 1, 2), (0, 2, 3),
            (4, 5, 6), (4, 6, 7),
            (0, 4, 7), (0, 7, 3),
            (1, 5, 6), (1, 6, 2),
            (3, 2, 6), (3, 6, 7),
            (0, 1, 5), (0, 5, 4)
        ]
        return v, f


class WindArrow3D:
    """
    3D arrow representing wind direction and magnitude.
    Constructed as a shaft (thin box) plus a pyramid tip, oriented to the wind vector.
    """
    def __init__(self,
                 base_position: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 arrow_scale: float = 1.0,
                 base_color: str = "#33ff33",
                 neg_color: str = "#ff3333") -> None:
        self.base_position = np.array(base_position, dtype=np.float32)
        self.arrow_scale = float(arrow_scale)
        self.base_color = base_color
        self.neg_color = neg_color
        self.wind_vec = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def set_wind(self, wind_x: float, wind_y: float, wind_z: float = 0.0) -> None:
        self.wind_vec = np.array([wind_x, wind_y, wind_z], dtype=np.float32)

    def _arrow_local(self, length: float, shaft_w: float = 0.03, shaft_t: float = 0.03,
                     head_len: float = 0.15, head_w: float = 0.08) -> Tuple[np.ndarray, List[Tuple[int, int, int]]]:
        """
        Create an arrow along +X in local space. Return vertices and faces.
        - shaft: thin box from x=0 to x=(length - head_len)
        - head: pyramid tip from that point to x=length
        """
        shaft_len = max(0.0, length - head_len)
        # Shaft box centered around y,z = 0
        sv, sf = DroneModel3D._make_box_centered(shaft_len, shaft_w, shaft_t)
        sv = sv + np.array([shaft_len / 2.0, 0.0, 0.0], dtype=np.float32)

        # Pyramid head (base square -> tip)
        hw, ht = head_w, head_w  # base square size
        base_z = head_w / 2.0
        bx = shaft_len
        # Base square vertices (4) and tip (1)
        head_v = np.array([
            [bx,  hw/2,  ht/2],
            [bx, -hw/2,  ht/2],
            [bx, -hw/2, -ht/2],
            [bx,  hw/2, -ht/2],
            [bx + head_len, 0.0, 0.0],  # tip
        ], dtype=np.float32)
        head_f = [
            (0, 1, 4),
            (1, 2, 4),
            (2, 3, 4),
            (3, 0, 4),
            # base (optional): (0, 1, 2), (0, 2, 3)
        ]

        # Merge
        verts = np.vstack([sv, head_v]).astype(np.float32)
        faces = sf + [(a + sv.shape[0], b + sv.shape[0], c + sv.shape[0]) for (a, b, c) in head_f]
        return verts, faces

    def get_arrow_mesh(self) -> Dict[str, object]:
        """
        Build a world-space arrow mesh aligned to current wind vector.
        Returns a dict with vertices, faces, color.
        """
        magnitude = float(np.linalg.norm(self.wind_vec))
        # Color by sign of wind_x for a simple cue; renderer may decide otherwise
        color = self.base_color if self.wind_vec[0] >= 0.0 else self.neg_color
        length = max(0.05, magnitude * self.arrow_scale)
        local_v, local_f = self._arrow_local(length)

        # Orientation: align +X to the wind direction
        if magnitude > 1e-6:
            dx, dy, dz = (self.wind_vec / magnitude).tolist()
            yaw = np.arctan2(dy, dx)
            pitch = np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        else:
            yaw = 0.0
            pitch = 0.0
        R = _rotation_matrix_xyz(0.0, pitch, yaw)
        v_rot = (local_v @ R.T).astype(np.float32)
        v_world = v_rot + self.base_position

        return {
            "vertices": v_world.tolist(),
            "faces": local_f,
            "color": color,
        }


class Camera3D:
    """
    Simple orbit/follow camera representation.
    """
    def __init__(self,
                 position: Tuple[float, float, float] = (0.0, -10.0, 5.0),
                 look_at: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 up: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                 distance: float = 10.0,
                 height_offset: float = 3.0,
                 follow_smooth: float = 0.1) -> None:
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.distance = float(distance)
        self.height_offset = float(height_offset)
        self.follow_smooth = float(follow_smooth)

    def update_for_drone(self, drone_position: Tuple[float, float, float]) -> None:
        target = np.array(drone_position, dtype=np.float32)
        desired_pos = np.array([target[0], target[1] - self.distance, target[2] + self.height_offset], dtype=np.float32)
        # Exponential smoothing
        self.position = (1.0 - self.follow_smooth) * self.position + self.follow_smooth * desired_pos
        self.look_at = (1.0 - self.follow_smooth) * self.look_at + self.follow_smooth * target

    def to_packet(self) -> Dict[str, object]:
        return {
            "position": self.position.tolist(),
            "look_at": self.look_at.tolist(),
            "up": self.up.tolist(),
        }


class Scene3D:
    """
    Scene container that aggregates drone, obstacles, wind arrow, and camera.
    Provides a JSON-serialisable scene packet for a renderer.
    """
    def __init__(self,
                 world_scale: float = 10.0,
                 world_bounds: Tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0)) -> None:
        self.world_scale = float(world_scale)
        self.world_bounds = world_bounds  # (xmin, xmax, ymin, ymax) in env units
        self.drone = DroneModel3D(scale=1.5)
        self.obstacles: List[Obstacle3D] = []
        self.wind_arrow = WindArrow3D(arrow_scale=2.0)
        self.camera = Camera3D(distance=10.0, height_offset=3.0, follow_smooth=0.1)

        # Cached last env state (for reference)
        self._last_obs: Optional[np.ndarray] = None

    def add_obstacle(self, obstacle: Obstacle3D) -> None:
        self.obstacles.append(obstacle)

    def _to_world(self, p: Tuple[float, float, float]) -> np.ndarray:
        """Convert env units to scene/world units using world_scale."""
        return np.array(p, dtype=np.float32) * self.world_scale

    def update_from_env_state(self, obs_vector: np.ndarray) -> None:
        """
        Update scene elements from environment observation:
        obs = [x, y, vx, vy, z, vz, wind_x, wind_y]
        """
        obs = np.asarray(obs_vector, dtype=np.float32)
        if obs.shape[0] < 8:
            raise ValueError("Expected observation with 8 elements: [x, y, vx, vy, z, vz, wind_x, wind_y]")
        x, y, vx, vy, z, vz, wind_x, wind_y = obs.tolist()
        self._last_obs = obs

        # Update drone transform
        pos_world = self._to_world((x, y, z))
        self.drone.set_position(pos_world[0], pos_world[1], pos_world[2])
        # Optional: set small tilt from velocity for visual flair (does not affect env physics)
        roll = -0.15 * vy  # lean into lateral
        pitch = 0.15 * vx  # lean into forward motion
        self.drone.set_rotation(roll, pitch, 0.0)

        # Update wind arrow at drone position with wind vector (z-component 0 for now)
        wind_vec_world = np.array([wind_x, wind_y, 0.0], dtype=np.float32) * self.world_scale * 0.2
        self.wind_arrow.base_position = pos_world
        self.wind_arrow.set_wind(float(wind_vec_world[0]), float(wind_vec_world[1]), float(wind_vec_world[2]))

        # Update camera to follow drone
        self.camera.update_for_drone(tuple(pos_world.tolist()))

    def generate_scene_packet(self) -> Dict[str, object]:
        """
        Return a JSON-serialisable dict describing the scene geometry and transforms.
        """
        drone_vertices = self.drone.get_transformed_vertices().tolist()
        drone_packet = {
            "vertices": drone_vertices,
            "faces": self.drone.faces,
            "face_colors": self.drone.face_colors,
        }

        obstacles_packet: List[Dict[str, object]] = []
        for obs in self.obstacles:
            v, f = obs.get_vertices_3d()
            # Scale obstacle verts to world units as they are given directly in env/world units
            v_scaled = v * self.world_scale
            obstacles_packet.append({
                "vertices": v_scaled.tolist(),
                "faces": f,
                "color": obs.color,
            })

        wind_packet = self.wind_arrow.get_arrow_mesh()

        xmin, xmax, ymin, ymax = self.world_bounds
        world_packet = {
            "bounds": {
                "xmin": xmin * self.world_scale,
                "xmax": xmax * self.world_scale,
                "ymin": ymin * self.world_scale,
                "ymax": ymax * self.world_scale,
                "zmin": 0.0 * self.world_scale,
                "zmax": 1.5 * self.world_scale,
            }
        }

        packet = {
            "drone": drone_packet,
            "obstacles": obstacles_packet,
            "wind": wind_packet,
            "camera": self.camera.to_packet(),
            "world": world_packet,
            "meta": {
                "world_scale": self.world_scale,
                "schema": "scene3d.v1"
            }
        }
        return packet




