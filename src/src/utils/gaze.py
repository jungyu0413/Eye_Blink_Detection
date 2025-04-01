"""Utility methods for gaze angle and error calculations."""
import cv2 as cv
import numpy as np
import torch


def angular_error_torch(predict, label):
    """Pytorch method to calculate angular loss (via cosine similarity)"""
    def angle_to_unit_vectors(y):
        sin = torch.sin(y)
        # torch.sin(angle)
        cos = torch.cos(y)
        return torch.stack([
            cos[:, 0] * sin[:, 1],
            sin[:, 0],
            cos[:, 0] * cos[:, 1],
            ], dim=1)

    a = angle_to_unit_vectors(predict)
    b = angle_to_unit_vectors(label)
    ab = torch.sum(a*b, dim=1)
    a_norm = torch.sqrt(torch.sum(torch.square(a), dim=1))
    b_norm = torch.sqrt(torch.sum(torch.square(b), dim=1))
    cos_sim = ab / (a_norm * b_norm)
    # cosine smilarity
    cos_sim = torch.clip(cos_sim, -1.0 + 1e-6, 1.0 - 1e-6)
    # clipping
    ang = torch.acos(cos_sim) * 180. / np.pi
    # angle
    return torch.mean(ang)
    # angle

def pitchyaw_to_vector(pitchyaws):
    # pitch : 눈 위아래
    # yaw : 양 옆
    # roll : 시계/반시계
    # -> gaze에서는 pich와 yaw만 필요할듯
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws (:obj:`numpy.array`): yaw and pitch angles :math:`(n\times 2)` in radians.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 3)` with 3D vectors per row.
    """
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def vector_to_pitchyaw(vectors):
    r"""Convert given gaze vectors to yaw (:math:`\theta`) and pitch (:math:`\phi`) angles.

    Args:
        vectors (:obj:`numpy.array`): gaze vectors in 3D :math:`(n\times 3)`.

    Returns:
        :obj:`numpy.array` of shape :math:`(n\times 2)` with values in radians.
    """
    n = vectors.shape[0]
    out = np.empty((n, 2))
    
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # pitch
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # yaw
    return out


radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * radians_to_degrees


def mean_angular_error(a, b):
    """Calculate mean angular error (via cosine similarity)."""
    return np.mean(angular_error(a, b))


def draw_gaze_val(eye_pos, pitchyaw, length=300.0):
    """Draw gaze angle on given image with a given eye positions."""
    dx = -length * np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])

    x = np.round(eye_pos[0] + dx).astype(int)
    y = np.round(eye_pos[1] + dy).astype(int)
    xy = [x,y]
    return xy


def draw_dot(image_in, xy, color=(100, 50, 230)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    x = np.round((xy[0][0] + xy[1][0]) / 2).astype(int) + 200
    y = np.round((xy[0][1] + xy[1][1]) / 2).astype(int) + 140
    
    cv.circle(image_out, (x, y), 1, color, 2)
    

    return image_out


def draw_gaze_map(image_in, eye_pos, pitchyaw, length=100.0, thickness=2, color=(0, 0, 230)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0]) 
    eye_pos_init = [1159, 827]
    gap =  eye_pos_init - eye_pos
    eye_pos = [eye_pos[0]+gap[0],eye_pos[1]+gap[1]]
    cv.arrowedLine(image_out, tuple(np.round(eye_pos_init).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    

    return image_out

def draw_gaze(image_in, eye_pos, pitchyaw, length=40.0, thickness=1, color=(0, 0, 230)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.cos(pitchyaw[0]) * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])

    cv.arrowedLine(image_out, tuple(np.round(eye_pos).astype(np.int32)),
                   tuple(np.round([eye_pos[0] + dx, eye_pos[1] + dy]).astype(int)), color,
                   thickness, cv.LINE_AA, tipLength=0.2)
    

    return image_out


def draw_iris(image_in, eye_pos, thickness=1, color=(255, 0, 0)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    cv.circle(image_out, tuple(np.round(eye_pos).astype(np.int32)), 1, color, -1)
    return image_out
