import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Load images
    image1 = cv2.imread('CV2024_HW3/data/hill1.JPG')
    image2 = cv2.imread('CV2024_HW3/data/hill2.JPG')

    # Convert to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors using KNN with k=2
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply ratio test as per Lowe's ratio (0.75 is commonly used threshold)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Extract matched keypoints' coordinates for RANSAC
    matched_points_img1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
    matched_points_img2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

    # Compute homography matrix using RANSAC
    H = homomat(matched_points_img1, matched_points_img2)
    #H = compute_homography_eigen(matched_points_img1,matched_points_img2)

    # Draw matches (for visualization)
    matched_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the matched features
    plt.figure(figsize=(15, 7))
    plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matches with Ratio Test")
    plt.axis('off')
    plt.show()

    print("Computed Homography Matrix:\n", H)

    warped_image = warp(image1, image2, H)

    # Display the result
    plt.figure(figsize=(15, 10))
    plt.imshow(cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB))
    plt.title("Warped and Blended Image")
    plt.axis("off")
    plt.show()

def homomat(points_in_img1, points_in_img2, num_iterations=2000, threshold=5.0):
    """
    Use RANSAC to compute the best homography matrix between two images.
    
    Parameters:
        points_in_img1 (numpy.ndarray): Matched points from image 1, shape (N, 2).
        points_in_img2 (numpy.ndarray): Matched points from image 2, shape (N, 2).
        num_iterations (int): Number of RANSAC iterations.
        threshold (float): Distance threshold to count inliers.
        
    Returns:
        best_H (numpy.ndarray): Homography matrix with the highest number of inliers.
    """
    
    max_inliers = 0
    best_H = None
    best_inliers = None

    for _ in range(num_iterations):
        # Step I: Sample 4 random correspondences
        idx = np.random.choice(len(points_in_img1), 4, replace=False)
        sample_points_img1 = points_in_img1[idx]
        sample_points_img2 = points_in_img2[idx]

        # Step II: Compute homography matrix mapping img2 to img1
        #H, _ = cv2.findHomography(sample_points_img2, sample_points_img1, method=0)
        H = compute_homography_eigen(sample_points_img2,sample_points_img1)

        if H is None:
            continue

        # Step III: Transform all points in image 2 using H
        points_in_img2_hom = np.hstack((points_in_img2, np.ones((points_in_img2.shape[0], 1))))
        projected_points_img1 = (H @ points_in_img2_hom.T).T
        projected_points_img1 /= projected_points_img1[:, 2][:, np.newaxis]

        # Compute distances between transformed points and points in image 1
        distances = np.linalg.norm(points_in_img1 - projected_points_img1[:, :2], axis=1)

        # Count inliers based on the threshold
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # Step IV: Update the best homography matrix if current one has more inliers
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H
            best_inliers = inliers

    # Step V: Recompute H using all inliers
    if best_inliers is not None:
        points_in_img1_inliers = points_in_img1[best_inliers]
        points_in_img2_inliers = points_in_img2[best_inliers]
        best_H, _ = cv2.findHomography(points_in_img2_inliers, points_in_img1_inliers, method=0)

    return best_H

def compute_homography_eigen(points_src, points_dst):
    """
    Compute the homography matrix H using the normalized DLT method.
    
    Parameters:
        points_src (numpy.ndarray): Source points from image 1, shape (N, 2).
        points_dst (numpy.ndarray): Corresponding destination points from image 2, shape (N, 2).
    
    Returns:
        H (numpy.ndarray): Homography matrix (3x3).
    """

    N = points_src.shape[0]
    A = np.zeros((2 * N, 9))

    for i in range(N):
        xs, ys = points_src[i]
        xd, yd = points_dst[i]

        A[2 * i] = [-xs, -ys, -1, 0, 0, 0, xd * xs, xd * ys, xd]
        A[2 * i + 1] = [0, 0, 0, -xs, -ys, -1, yd * xs, yd * ys, yd]

    # Compute SVD of A
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H_norm = h.reshape((3, 3))

    # Normalize H so that H[2, 2] = 1
    H_norm = H_norm / H_norm[2, 2]

    return H_norm


def warp(img1, img2, H):
    """
    Warps img2 onto img1 using the homography H and blends them.
    
    Parameters:
        img1 (numpy.ndarray): Base image where img2 will be warped onto.
        img2 (numpy.ndarray): Image to be warped.
        H (numpy.ndarray): Homography matrix.
            
    Returns:
        result (numpy.ndarray): The blended result of img1 and the warped img2.
    """
    # Dimensions of img1
    height1, width1 = img1.shape[:2]

    # Transform corners of img2 to find the bounding box in img1 coordinates
    height2, width2 = img2.shape[:2]
    corners_img2 = np.array([[0, 0], [0, height2], [width2, height2], [width2, 0]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners_img2[None, :, :], H)[0]
    
    # Get the min and max points to determine the size of the resulting canvas
    all_corners = np.vstack((transformed_corners, [[0, 0], [width1, 0], [width1, height1], [0, height1]]))
    x_min, y_min = np.int32(all_corners.min(axis=0).flatten() - 0.5)
    x_max, y_max = np.int32(all_corners.max(axis=0).flatten() + 0.5)

    # Translation matrix to shift the result to positive coordinates
    translation = np.array([[1, 0, -x_min],
                            [0, 1, -y_min],
                            [0, 0, 1]])

    # Warp img2 with the translation and homography
    warped_img2 = cv2.warpPerspective(img2, translation @ H, (x_max - x_min, y_max - y_min))

    # Create the result canvas
    result = np.zeros((y_max - y_min, x_max - x_min, 3), dtype=np.uint8)

    # Place img1 onto the canvas
    result[-y_min:-y_min + height1, -x_min:-x_min + width1] = img1

    # Create masks for blending
    mask_img1 = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)
    mask_img1[-y_min:-y_min + height1, -x_min:-x_min + width1] = 255

    mask_img2 = cv2.warpPerspective(np.ones((height2, width2), dtype=np.uint8) * 255, translation @ H, (x_max - x_min, y_max - y_min))

    # Combine masks to find overlapping regions
    overlap_mask = cv2.bitwise_and(mask_img1, mask_img2)

    # Add non-overlapping regions from warped_img2 to result
    non_overlap_mask_img2 = cv2.subtract(mask_img2, overlap_mask)
    result[non_overlap_mask_img2 > 0] = warped_img2[non_overlap_mask_img2 > 0]

    # Blend overlapping regions
    alpha = 0.5
    indices = np.where(overlap_mask > 0)
    result[indices[0], indices[1]] = (alpha * result[indices[0], indices[1]] + (1 - alpha) * warped_img2[indices[0], indices[1]]).astype(np.uint8)

    return result

if __name__ == '__main__':
    main()
