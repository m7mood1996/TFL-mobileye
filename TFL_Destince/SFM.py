import numpy as np
import math


def calc_TFL_dist(prev_container, curr_container, focal, pp):
    norm_prev_pts, norm_curr_pts, R, foe, tZ = prepare_3D_data(prev_container, curr_container, focal, pp)
    if (abs(tZ) < 10e-6):
        print('tz = ', tZ)
    elif (norm_prev_pts.size == 0):
        print('no prev points')
    elif (norm_curr_pts.size == 0):
        print('no curr points')
    else:
        curr_container.corresponding_ind, curr_container.traffic_lights_3d_location, curr_container.valid = calc_3D_data(
            norm_prev_pts, norm_curr_pts, R, foe, tZ)
    return curr_container


def prepare_3D_data(prev_container, curr_container, focal, pp):
    norm_prev_pts = normalize(prev_container.traffic_light, focal, pp)
    norm_curr_pts = normalize(curr_container.traffic_light, focal, pp)
    R, foe, tZ = decompose(np.array(curr_container.EM))
    return norm_prev_pts, norm_curr_pts, R, foe, tZ


def calc_3D_data(norm_prev_pts, norm_curr_pts, R, foe, tZ):
    norm_rot_pts = rotate(norm_prev_pts, R)
    pts_3D = []
    corresponding_ind = []
    validVec = []
    for p_curr in norm_curr_pts:
        corresponding_p_ind, corresponding_p_rot = find_corresponding_points(p_curr, norm_rot_pts, foe)
        Z = calc_dist(p_curr, corresponding_p_rot, foe, tZ)
        valid = (Z > 0)
        if not valid:
            Z = 0
        validVec.append(valid)
        P = Z * np.array([p_curr[0], p_curr[1], 1])
        pts_3D.append((P[0], P[1], P[2]))
        corresponding_ind.append(corresponding_p_ind)
    return corresponding_ind, np.array(pts_3D), validVec


def normalize(pts, focal, pp):
    # transform pixels into normalized pixels using the focal length and principle point
    points = []
    for point in pts:
        points.append([(point[0] - pp[0]) / focal, (point[1] - pp[1]) / focal])

    points = np.array(points)
    return points


def unnormalize(pts, focal, pp):
    # transform normalized pixels into pixels using the focal length and principle point

    return np.array([[pts[i, 0] * focal + pp[0], pts[i, 1] * focal + pp[1]] for i in range(pts.shape[0])])


def decompose(EM):
    # extract R, foe and tZ from the Ego Motion
    R = EM[:3, :3]

    tx = EM[0, 3]
    ty = EM[1, 3]
    tz = EM[2, 3]

    if 3e-10 > tz > -3e-10:
        tz = 0
        foe = []

    else:
        foe = [tx, ty] / tz

    return R, foe, tz


def rotate(pts, R):
    # rotate the points - pts using R

    rotated_points = []
    for point in pts:
        newP = R.dot(np.array([point[0], point[1], 1]))
        newP = newP[0:2] / newP[2]
        rotated_points.append(newP)
    rotated_points = np.array(rotated_points)
    return rotated_points


def find_corresponding_points(p, norm_pts_rot, foe):
    # compute the epipolar line between p and foe
    # run over all norm_pts_rot and find the one closest to the epipolar line
    # return the closest point and its index
    m = (foe[1] - p[1]) / (foe[0] - p[0])
    n = (p[1] * foe[0] - foe[1] * p[0]) / (foe[0] - p[0])
    scores = []
    for point in norm_pts_rot:
        scores.append(abs(calculate_epipolar_score(m, n, point[0]) - point[1]))

    index = get_min_index(scores)
    return index, norm_pts_rot[index]


def get_min_index(scores):
    min_score = scores[0]
    index = 0
    for i in range(len(scores)):
        if scores[i] < min_score:
            min_score = scores[i]
            index = i
        i += 1
    return index


def calculate_epipolar_score(m, n, x):
    return m * x + n


def calc_dist(p_curr, p_rot, foe, tZ):
    # calculate the distance of p_curr using x_curr, x_rot, foe_x and tZ
    # calculate the distance of p_curr using y_curr, y_rot, foe_y and tZ
    # combine the two estimations and return estimated Z

    X1 = foe[0] - p_rot[0]
    X2 = p_curr[0] - p_rot[0]
    Zx = (tZ * X1) / X2

    Y1 = foe[1] - p_rot[1]
    Y2 = p_curr[1] - p_rot[1]
    Zy = (tZ * Y1) / Y2

    curr_x = p_curr[0] + foe[0]

    curr_y = p_curr[1] + foe[1]

    if p_curr[0] == p_rot[0] and p_curr[1] == p_rot[1]:
        return 0

    else:
        return (Zy + Zx) / 2
