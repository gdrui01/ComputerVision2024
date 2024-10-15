#%%

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import camera_calibration_show_extrinsics as show
from PIL import Image

#%%
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# (8,6) is for the given testing images.
# If you use the another data (e.g. pictures you take by your smartphone), 
# you need to set the corresponding numbers.
corner_x = 7
corner_y = 7
objp = np.zeros((corner_x*corner_y,3), np.float32)
objp[:,:2] = np.mgrid[0:corner_x, 0:corner_y].T.reshape(-1,2)
print(objp, objp.shape)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('data/*.jpg')

#%%
# Step through the list and search for chessboard corners
print('Start finding chessboard corners...')
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)

    #Find the chessboard corners
    print('find the chessboard corners of',fname)
    ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
        plt.imshow(img)
    else: 
        print('failed to find corners for image')

#%%

ret, corners = cv2.findChessboardCorners(gray, (corner_x,corner_y), None)
cv2.drawChessboardCorners(img, (corner_x,corner_y), corners, ret)
plt.imshow(img)
print(corners.shape)

#%%

#######################################################################################################
#                                Homework 1 Camera Calibration                                        #
#               You need to implement camera calibration(02-camera p.76-80) here.                     #
#   DO NOT use the function directly, you need to write your own calibration function from scratch.   #
#                                          H I N T                                                    #
#                        1.Use the points in each images to find Hi                                   #
#                        2.Use Hi to find out the intrinsic matrix K                                  #
#                        3.Find out the extrensics matrix of each images.                             #
#######################################################################################################

#%%
# our own code
np.set_printoptions(suppress=True)

num_images = len(objpoints)

H_list = []

# iterate over images
for i in range(num_images):
    image_points_world = objpoints[i]
    image_points_image = imgpoints[i].squeeze(axis=1)

    P_list = []

    # iterate over points i mage
    for j in range(image_points_world.shape[0]):
        point_world = image_points_world[j,:]
        point_image = image_points_image[j,:]

        U, V, W = point_world
        W = 1

        u, v = point_image

        p_rows = (
            [U, V, W, 0, 0, 0, -u*U, -u*V, -u*W],
            [0, 0, 0, U, V, W, -v*U, -v*V, -v*W],
        )
        
        P_list.append(p_rows[0])
        P_list.append(p_rows[1])

    P = np.stack(P_list, axis=0).astype(np.float32)
    
    U_P,S_P,V_P = np.linalg.svd(P)

    
    m = V_P[:, -1]
    # m = m / m[-1]
    H = V_P[-1,:].reshape(3,3)
    H /=  H[-1,-1]  
    # print(H)
    # H /= H[-1,-1]

    H_list.append(H)

    total_dist_error = 0
    for j in range(image_points_world.shape[0]):
        point_world = image_points_world[j,:]
        point_image = image_points_image[j,:]

        u_t = np.dot(H, np.array([point_world[0], point_world[1], 1]))
        u_t /= u_t[-1]

        total_dist_error += np.linalg.norm(np.array([point_image[0],point_image[1],1]) - u_t)
    print(f'Σ_i (for i in img_{i}) || u_i - HU_i || = ',total_dist_error, '💅')

V_list = []

for H in H_list:
    h_rows = (
        [
            H[0,1] * H[0,0],
            H[1,1] * H[1,0],
            H[2,1] * H[2,0],
            H[0,1] * H[1,0] + H[1,1] * H[0,0], #h1[0] * h2[0] + h2[1] * h1[0]
            H[0,1] * H[2,0] + H[2,1] * H[0,0],
            H[1,1] * H[2,0] + H[2,1] * H[1,0],
        ],
        [
            H[0,0] * H[0,0] - H[0,1] * H[0,1],
            H[1,0] * H[1,0] - H[1,1] * H[1,1],
            H[2,0] * H[2,0] - H[2,1] * H[2,1],
            2 * (H[1,0] * H[0,0] - H[1,1] * H[0,1]),
            2 * (H[2,0] * H[0,0] - H[2,1] * H[0,1]),
            2 * (H[2,1] * H[1,0] - H[2,1] * H[1,1]),
        ]
    )

    V_list.append(h_rows[0])
    V_list.append(h_rows[1])

V = np.stack(V_list, axis=0)

U_V, S_V, V_V = np.linalg.svd(V)

b = V_V[-1,:]

print('|| min_b Vb || = \n',np.linalg.norm(np.dot(V,b)))

B = -np.array([
    [b[0],b[3],b[4]],
    [b[3],b[1],b[5]],
    [b[4],b[5],b[2]],
    # [b[0],b[1],b[3]],
    # [b[1],b[2],b[4]],
    # [b[3],b[4],b[5]]
])

print('B = \n',B)
print('eigvals of B:',np.linalg.eigvals(B))

# B should be invariant to scaling by k: R \ {0}
if np.all(np.linalg.eigvals(B) < 0):
    B = -B

L = np.linalg.cholesky(B)


K_inv = L.T
K = np.linalg.inv(L.T)

print('B - (K^-1t * K^-1) = \n',(K_inv.transpose() @ K_inv) - B)

#%% extrinsic matrices

# test zeronesss
for H in H_list:
    h1  = H[:,0]
    h2  = H[:,1]

    first = np.einsum('i,ij,j->', h1,B,h2)
    left = np.einsum('i,ij,j->', h1,B,h1)
    right = np.einsum('i,ij,j->', h2,B,h2)
    print(first, left - right)

R_list = []

for H in H_list:
    h1  = H[:,0]
    h2  = H[:,1]
    h3  = H[:,2]

    lamda = 1 / np.linalg.norm(np.dot(K_inv, h1))

    r1 = np.dot(K_inv, h1) * lamda
    # r1 /= np.linalg.norm(r1)

    r2 = np.dot(K_inv, h2) * lamda
    # r2 /= np.linalg.norm(r2)

    r3 = np.cross(r1, r2)
    # r3 /= np.linalg.norm(r3)

    # print(r1, r2, r3)
    print(np.dot(r1, r2))

    
    t = np.dot(K_inv, h3) * lamda
    # t /= np.linalg.norm(h3)

    R_list.append(np.column_stack((r1,r2,r3,t)))

R = np.stack(R_list, axis=0)
#%%

img_size = img[0].shape
# You need to comment these functions and write your calibration function from scratch.
# Notice that rvecs is rotation vector, not the rotation matrix, and tvecs is translation vector.
# In practice, you'll derive extrinsics matrixes directly. The shape must be [pts_num,3,4], and use them to plot.
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
# Vr = np.array(rvecs)
# Tr = np.array(tvecs)
# extrinsics = np.concatenate((Vr, Tr), axis=1).reshape(-1,6)


# show the camera extrinsics
print('Show the camera extrinsics')
# plot setting
# You can modify it for better visualization
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection='3d')
# camera setting
# camera_matrix = mtx
# 2936 X x 3916 = 7497
cam_width = 0.064/0.1
cam_height = 0.032/0.1
scale_focal = 1600
# chess board setting
board_width = 8
board_height = 6
square_size = 1
# display
# True -> fix board, moving cameras
# False -> fix camera, moving boards
min_values, max_values = show.draw_camera_boards(ax, K, cam_width, cam_height,
                                                scale_focal, R, board_width,
                                                board_height, square_size, True)

X_min = min_values[0]
X_max =max_values[0]
Y_min =min_values[1]
Y_max =max_values[1]
Z_min =min_values[2]
Z_max = max_values[2]
max_range = np.array([X_max-X_min, Y_max-Y_min, Z_max-Z_min]).max() / ě.0

mid_x = (X_max+X_min) * 0.5
mid_y = (Y_max+Y_min) * 0.5
mid_z = (Z_max+Z_min) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, 0)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_zlabel('-y')
ax.set_title('Extrinsic Parameters Visualization')
# ax.view_init(50, 180)
# plt.show()

#animation for rotating plot

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (1500, 1500))

# for angle in range(0, 360):
#     ax.view_init(30, angle)
#     plt.draw()
#     plt.pause(.001)
    

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    
    # Convert plot to an image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Convert RGB to BGR (OpenCV uses BGR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Write the frame
    out.write(img)

out.release()
# # %%

# %%
