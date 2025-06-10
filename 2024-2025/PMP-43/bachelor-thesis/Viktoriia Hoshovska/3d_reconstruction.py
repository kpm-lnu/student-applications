import cv2 as cv
import numpy as np
import open3d as o3d


def to_homogeneous(points_1, points_2):

    if points_1.ndim == 1:
        points_1_hom = np.pad(points_1, (0, 1), "constant", constant_values=1.0)
        points_2_hom = np.pad(points_2, (0, 1), "constant", constant_values=1.0)
    else:
        points_1_hom = np.pad(points_1, [(0, 0), (0, 1)], "constant", constant_values=1.0)
        points_2_hom = np.pad(points_2, [(0, 0), (0, 1)], "constant", constant_values=1.0)

    return np.float64(points_1_hom), np.float64(points_2_hom)


def normalization_M(x: np.ndarray) -> np.ndarray:
    if x.shape[0] != 3:
        raise ValueError("Input must be 3×N homogeneous coordinates.")

    x_cart = x[:2] / x[2]
    centroid = x_cart.mean(axis=1, keepdims=True)

    dists = np.linalg.norm(x_cart - centroid, axis=0)
    mean_dist = dists.mean()

    if mean_dist == 0:
        raise ValueError("Degenerate configuration: all points coincide.")

    s = np.sqrt(2) / mean_dist

    N = np.array([[ s, 0, -s * centroid[0, 0]],
                  [ 0, s, -s * centroid[1, 0]],
                  [ 0, 0,               1. ]],
                 dtype=np.float64)
    return N


def compute_fundamental_8point(points1, points2):
    pts1_h = np.column_stack([points1, np.ones(len(points1))]).T
    pts2_h = np.column_stack([points2, np.ones(len(points2))]).T

    T1 = normalization_M(pts1_h)
    T2 = normalization_M(pts2_h)

    pts1_norm_h = T1 @ pts1_h
    pts2_norm_h = T2 @ pts2_h

    pts1_norm = (pts1_norm_h[:2] / pts1_norm_h[2]).T
    pts2_norm = (pts2_norm_h[:2] / pts2_norm_h[2]).T

    A = np.column_stack([pts1_norm[:, 0] * pts2_norm[:, 0],   # x1 x2
                         pts1_norm[:, 0] * pts2_norm[:, 1],   # x1 y2
                         pts1_norm[:, 0],                     # x1
                         pts1_norm[:, 1] * pts2_norm[:, 0],   # y1 x2
                         pts1_norm[:, 1] * pts2_norm[:, 1],   # y1 y2
                         pts1_norm[:, 1],                     # y1
                         pts2_norm[:, 0],                     # x2
                         pts2_norm[:, 1],                     # y2
                         np.ones(len(pts1_norm))])         # 1
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F_norm)
    S[2] = 0
    F_norm = U @ np.diag(S) @ Vt

    F = T2.T @ F_norm @ T1

    return F / F[2, 2]


def decompose_E(E):
    U, _, V_t = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(V_t) < 0:
        V_t *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R1 = U @ W @ V_t
    R2 = U @ W.T @ V_t
    t = U[:, 2].reshape(3, 1)

    return R1, R2, t


def resolve_rt(R1, R2, t, K0, K1, points_1, points_2):
    P0 = K0 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    options = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    best_option = None
    max_pos_depth = 0

    for R, t_option in options:
        P1_option = K1 @ np.hstack((R, t_option))

        X = triangulate_points(points_1, points_2, P0, P1_option)

        depth_1 = X[:, 2]
        depth_2 = (R @ X.T + t_option).T[:, 2]

        pos_depth = np.sum((depth_1 > 0) & (depth_2 > 0))
        if pos_depth > max_pos_depth:
            max_pos_depth = pos_depth
            best_option = (P0, P1_option)

    return best_option


def triangulate_points(points_1, points_2, P0, P1):
    points_1_hom, points_2_hom = to_homogeneous(points_1, points_2)

    num_points = points_1.shape[0]
    X_homo = np.zeros((num_points, 4))

    for i in range(num_points):
        A = np.vstack([
            points_1_hom[i, 0] * P0[2, :] - P0[0, :],
            points_1_hom[i, 1] * P0[2, :] - P0[1, :],
            points_2_hom[i, 0] * P1[2, :] - P1[0, :],
            points_2_hom[i, 1] * P1[2, :] - P1[1, :]
        ])
        _, _, Vt = np.linalg.svd(A)
        X_homo[i, :] = Vt[-1, :] / Vt[-1, -1]

    return X_homo[:, :3]


def visualize_3d_points(points3D):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    o3d.visualization.draw_geometries([pcd])


def reconstruct_8_point(points_1, points_2, K0, K1):
    F_8_point = compute_fundamental_8point(points_1, points_2)

    E = K1.T @ F_8_point @ K0

    R1, R2, t = decompose_E(E)
    P0, P1 = resolve_rt(R1, R2, t, K0, K1, points_1, points_2)

    points_3D = triangulate_points(points_1, points_2, P0, P1)

    return points_3D


def reconstruct_7_point(points_1, points_2, K0, K1):
    F, mask = cv.findFundamentalMat(points_1, points_2, cv.FM_RANSAC, 1.0, 0.99)

    inliers_1 = points_1[mask.ravel() == 1]
    inliers_2 = points_2[mask.ravel() == 1]

    E = K1.T @ F @ K0
    R1, R2, t = decompose_E(E)
    P0, P1 = resolve_rt(R1, R2, t, K0, K1, inliers_1, inliers_2)

    points_3D = triangulate_points(inliers_1, inliers_2, P0, P1)

    return points_3D


def reconstruction_3d(image_1_path, image_2_path, K0, K1):

    image_l = cv.imread(image_1_path, cv.IMREAD_GRAYSCALE)
    image_r = cv.imread(image_2_path, cv.IMREAD_GRAYSCALE)

    sift = cv.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(image_l, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image_r, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches_knn = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    good_matches = []
    ratio_thresh = 0.8
    for m, n in matches_knn:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    points_1 = np.float32([keypoints_1[m.queryIdx].pt for m in good_matches])
    points_2 = np.float32([keypoints_2[m.trainIdx].pt for m in good_matches])

    points_3D = reconstruct_7_point(points_1, points_2, K0, K1)

    visualize_3d_points(points_3D)


if __name__ == "__main__":
    image_l = "basket_l.jpg"
    image_r = "basket_r.jpg"
    # image_l = "colosseum_l.jpg"
    # image_r = "colosseum_r.jpg"
    K0 = np.array([[970.63830949, 0, 639.27638111],
                   [0, 970.1452575, 482.76599395],
                   [0, 0, 1]], dtype=np.float64)
    K1 = K0.copy()

    # for the bike
    # image_l = "bike_l.png"
    # image_r = "bike_r.png"
    # K0 = np.array([[5299.313, 0, 1263.818],
    #               [0, 5299.313, 977.763],
    #               [0, 0, 1]])
    # K1 = np.array([[5299.313, 0, 1438.004],
    #               [0, 5299.313, 977.763],
    #               [0, 0, 1]])

    points_3D = reconstruction_3d(image_l, image_r, K0, K1)
