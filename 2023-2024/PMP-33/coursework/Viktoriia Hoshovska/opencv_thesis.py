import numpy as np
import cv2

def normalize_points(points):
    mean = np.mean(points, axis=0)
    std = np.std(points) # standard deviation, sqrt(Variance)
    transform = np.array([[1/std, 0, -mean[0]/std], 
                          [0, 1/std, -mean[1]/std], 
                          [0, 0, 1]])
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    points_norm = (transform @ points_h.T).T
    return points_norm[:, :2], transform

def fundamental_matrix(points1, points2):
    # Normalizing points
    points1, T1 = normalize_points(points1)
    points2, T2 = normalize_points(points2)
        
    A = np.zeros((points1.shape[0], 9))
    for i in range(points1.shape[0]):
        x1, y1 = points1[i]
        x2, y2 = points2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    # Solving for F using SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)
    
    # Enforcing rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ Vt
    
    # Denormalizing the fundamental matrix
    F = T2.T @ F @ T1
    
    return F

def ransac_fundamental_matrix(kp1, kp2, matches, threshold=0.7):
    if len(matches) < 8:
        return [], None
    
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    best_inliers = []
    best_F = None
    iterations = 2000
    points_n = 8
    
    for _ in range(iterations):
        indices = np.random.choice(len(matches), points_n, replace=False)
        sample_points1 = points1[indices]
        sample_points2 = points2[indices]
        
        F = fundamental_matrix(sample_points1, sample_points2)
        
        points1_h = np.hstack((points1, np.ones((points1.shape[0], 1))))
        points2_h = np.hstack((points2, np.ones((points2.shape[0], 1))))
        
        lines1 = F @ points2_h.T
        lines2 = F.T @ points1_h.T
        
        errors = []
        for i in range(points1.shape[0]):
            error1 = np.abs(np.dot(lines1[:, i], points1_h[i])) / np.linalg.norm(lines1[:2, i])
            error2 = np.abs(np.dot(lines2[:, i], points2_h[i])) / np.linalg.norm(lines2[:2, i])
            errors.append((error1 + error2) / 2)
        
        inliers = [m for i, m in enumerate(matches) if errors[i] < threshold]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_F = F
            
    return best_inliers, best_F

def find_matching_points(image1, image2):
    # Resizing images to fit within a window
    max_height = 600
    max_width = 800
    scale1 = min(max_height / image1.shape[0], max_width / image1.shape[1])
    scale2 = min(max_height / image2.shape[0], max_width / image2.shape[1])
    scale = min(scale1, scale2)
    
    resized_image1 = cv2.resize(image1, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    resized_image2 = cv2.resize(image2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
    # Converting to grayscale
    gray_image1 = cv2.cvtColor(resized_image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(resized_image2, cv2.COLOR_BGR2GRAY)
    
    # Getting keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
    
    # Matching descriptors
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)
    
    # Applying Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # Computing fundamental matrix and filtering matches using RANSAC 
    inliers, F = ransac_fundamental_matrix(keypoints1, keypoints2, good_matches)
    
    # Displaying matches
    matches = cv2.drawMatches(resized_image1, keypoints1, resized_image2, keypoints2, inliers, None, matchesMask=None)
    cv2.imshow('Matched Points', matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    image1 = cv2.imread('candles_l.jpg')
    image2 = cv2.imread('candles_r.jpg')
    
    find_matching_points(image1, image2)



