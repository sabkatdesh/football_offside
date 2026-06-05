import numpy as np
import cv2
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration


def get_attacking_direction(
    goalkeepers: sv.Detections,
    transformer,
    config: SoccerPitchConfiguration
) -> dict | None:
    """
    Determines which team is attacking and in which direction
    based on goalkeeper positions.
    """
    if transformer is None or len(goalkeepers) == 0:
        return None

    pitch_mid_x = config.length / 2

    gk_xy       = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    gk_pitch_xy = transformer.transform_points(points=gk_xy)

    gk_x          = gk_pitch_xy[:, 0]
    dist_to_left  = np.abs(gk_x - 0)
    dist_to_right = np.abs(gk_x - config.length)
    closest       = np.minimum(dist_to_left, dist_to_right)
    best_idx      = np.argmin(closest)

    gk_pos_x       = gk_pitch_xy[best_idx, 0]
    defending_team = int(goalkeepers.class_id[best_idx])
    attacking_team = 1 - defending_team

    if gk_pos_x < pitch_mid_x:
        defending_goal_x = 0.0
        attack_direction = 'right'
    else:
        defending_goal_x = float(config.length)
        attack_direction = 'left'

    return {
        'defending_team_id': defending_team,
        'attacking_team_id': attacking_team,
        'defending_goal_x':  defending_goal_x,
        'attack_direction':  attack_direction,
    }


def get_offside_line_x(
    players: sv.Detections,
    goalkeepers: sv.Detections,
    transformer,
    direction_info: dict | None,
    config: SoccerPitchConfiguration
) -> float | None:
    """
    Finds x of the 2nd last defender = offside line.
    Needs at least 2 defenders (including GK) visible.
    """
    if direction_info is None or transformer is None:
        return None

    defending_team_id = direction_info['defending_team_id']
    defending_goal_x  = direction_info['defending_goal_x']

    all_x = []

    def_mask = players.class_id == defending_team_id
    if def_mask.any():
        def_xy    = players[def_mask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        def_pitch = transformer.transform_points(points=def_xy)
        all_x.append(def_pitch[:, 0])

    gk_mask = goalkeepers.class_id == defending_team_id
    if gk_mask.any():
        gk_xy    = goalkeepers[gk_mask].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        gk_pitch = transformer.transform_points(points=gk_xy)
        all_x.append(gk_pitch[:, 0])

    if not all_x:
        return None

    all_x_flat = np.concatenate(all_x)

    if len(all_x_flat) < 2:
        return None

    # sort by distance to defending goal — closest first
    dist_to_goal = np.abs(all_x_flat - defending_goal_x)
    sorted_x     = all_x_flat[np.argsort(dist_to_goal)]

    # index 0 = last man (GK), index 1 = 2nd last = offside line
    return float(sorted_x[1])


def check_offside(
    players: sv.Detections,
    transformer,
    direction_info: dict | None,
    offside_line_x: float | None,
) -> np.ndarray:
    """
    Pure per-frame positional offside check.

    A player is in offside position if:
      dist(attacker, goal) < dist(offside_line, goal)

    No persistence, no ball check — just position every frame.

    Returns:
      - offside_mask : bool array over `players`
    """
    n            = len(players)
    offside_mask = np.zeros(n, dtype=bool)

    if direction_info is None or transformer is None or offside_line_x is None:
        return offside_mask

    attacking_team_id = direction_info['attacking_team_id']
    defending_goal_x  = direction_info['defending_goal_x']

    att_mask = players.class_id == attacking_team_id
    if not att_mask.any():
        return offside_mask

    att_players  = players[att_mask]
    att_xy       = att_players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    att_pitch_xy = transformer.transform_points(points=att_xy)
    att_x        = att_pitch_xy[:, 0]

    dist_attacker_to_goal    = np.abs(att_x        - defending_goal_x)
    dist_offsideline_to_goal = abs(offside_line_x  - defending_goal_x)

    ahead_of_defender = dist_attacker_to_goal < dist_offsideline_to_goal

    att_indices = np.where(att_mask)[0]
    for i, idx in enumerate(att_indices):
        if ahead_of_defender[i]:
            offside_mask[idx] = True

    return offside_mask


def draw_offside_lines_on_radar(
    radar_frame: np.ndarray,
    offside_line_x: float | None,
    config: SoccerPitchConfiguration,
    padding: int,
    scale: float,
    offside_detected: bool = False,
) -> np.ndarray:
    """
    Draws on the radar:
      - Red solid line   → offside line (2nd last defender x)
      - Verdict text     → OFFSIDE POSITION (yellow) or ONSIDE (green)
    """
    h, w = radar_frame.shape[:2]

    def pitch_x_to_pixel(px: float) -> int:
        return int(px * scale + padding)

    if offside_line_x is not None:
        ox = pitch_x_to_pixel(offside_line_x)
        if 0 <= ox < w:
            cv2.line(radar_frame, (ox, 0), (ox, h),
                     color=(0, 0, 255), thickness=2)

        if offside_detected:
            text  = 'OFFSIDE POSITION'
            color = (0, 215, 255)   # yellow
        else:
            text  = 'ONSIDE'
            color = (0, 200, 0)     # green

        cv2.putText(
            radar_frame, text,
            org=(padding + 4, h - padding - 4),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.55,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA
        )

    return radar_frame