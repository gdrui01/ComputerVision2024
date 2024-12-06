import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageReconstruction:
    def __init__(self, img1_path, img2_path, ratio=0.65, error_threshold=2):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.ratio = ratio
        self.error_threshold = error_threshold
        
        self.K1 = np.array([[5426.566895, 0.678017, 330.096680],
                             [0.000000, 5423.133301, 648.950012],
                             [0.000000, 0.000000, 1.000000]])
        
        self.K2 = np.array([[5426.566895, 0.678017, 387.430023],
                             [0.000000, 5423.133301, 620.616699],
                             [0.000000, 0.000000, 1.000000]])
        
    def _sift_feature_matching(self):
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        
        while img2.shape[0] > 1000:
            if img1.shape == img2.shape:
                img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
            img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.xfeatures2d.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        matches = []
        for i in range(des1.shape[0]):
            des1_ = np.tile(des1[i], (des2.shape[0], 1))
            error = des1_ - des2
            SSD = np.sum((error**2), axis=1)
            idx_sort = np.argsort(SSD)
            
            if SSD[idx_sort[0]] < self.ratio * SSD[idx_sort[1]]:
                matches.append([kp1[i].pt, kp2[idx_sort[0]].pt])
        
        return np.array(matches), img1, img2
    
    def _normalize_points(self, pts_1, pts_2, img1, img2):
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        
        T1 = np.array([[2.0/w1, 0, -1], [0, 2/h1, -1], [0, 0, 1.0]])
        T2 = np.array([[2.0/w2, 0, -1], [0, 2/h2, -1], [0, 0, 1.0]])
        
        x = np.zeros(shape=(3,1))
        x[2,0] = 1
        
        for i in range(len(pts_1)):
            x[0,0] = pts_1[i,0]
            x[1,0] = pts_1[i,1]
            pts_1[i,0] = np.dot(T1, x)[0,0]
            pts_1[i,1] = np.dot(T1, x)[1,0]

            x[0,0] = pts_2[i,0]
            x[1,0] = pts_2[i,1]
            pts_2[i,0] = np.dot(T2, x)[0,0]
            pts_2[i,1] = np.dot(T2, x)[1,0]
        
        return pts_1, pts_2, T1, T2
    
    def _ransac_fundamental_matrix(self, line, img1, img2):
        kp1, kp2 = line[:, 0], line[:, 1]
        iterations = 3000
        maxinlier = 0
        kp_num = len(kp1)
        
        best_F = None
        
        for _ in range(iterations):
            rand_indices = np.random.choice(kp_num, 8, replace=False)
            kp1_rand = kp2[rand_indices]
            kp2_rand = kp1[rand_indices]
            
            normalized_pts1, normalized_pts2, T1, T2 = self._normalize_points(
                kp1_rand.copy(), kp2_rand.copy(), img1, img2
            )
            
            F = self._compute_fundamental_matrix(normalized_pts1, normalized_pts2, T1, T2)
            
            pts_tmp1, pts_tmp2, inlier = self._count_inliers(kp2, kp1, F)
            
            if inlier >= maxinlier:
                maxinlier = inlier
                best_kp1 = pts_tmp1
                best_kp2 = pts_tmp2
                best_F = F
        
        return best_kp1, best_kp2, best_F
    
    def _compute_fundamental_matrix(self, kp1, kp2, T1, T2):
        A = []
        for j in range(len(kp1)):
            x1, y1 = kp1[j][0], kp1[j][1]
            x2, y2 = kp2[j][0], kp2[j][1]
            A.append([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1])
        
        _,_,Vt = np.linalg.svd(A)
        Fi = Vt[-1].reshape(3,3)
        U,S,V = np.linalg.svd(Fi)
        
        S1 = np.zeros((3,3))
        S1[0,0], S1[1,1] = S[0], S[1]
        
        F = (U.dot(S1)).dot(V)
        F = np.dot(np.transpose(T2), np.dot(F, T1))
        F /= F[2,2]
        
        return F
    
    def _count_inliers(self, pts1, pts2, F):
        num = pts1.shape[0]
        inlier = 0
        pts_tmp1, pts_tmp2 = [], []
        
        for i in range(num):
            x1, y1 = pts1[i,0], pts1[i,1]
            x2, y2 = pts2[i,0], pts2[i,1]
            
            a = F[0,0]*x1 + F[0,1]*y1 + F[0,2]
            b = F[1,0]*x1 + F[1,1]*y1 + F[1,2]
            c = F[2,0]*x1 + F[2,1]*y1 + F[2,2]
            
            dist = abs(a*x2 + b*y2 + c) / ((a**2 + b**2)**0.5)
            
            if dist < 1:
                pts_tmp1.append([x1,y1])
                pts_tmp2.append([x2,y2])
                inlier += 1
        
        return pts_tmp1, pts_tmp2, inlier
    
    def _linear_triangulation(self, p1, p2, m1, m2):
        A = np.zeros((4,4))
        A[0,:] = p1[0]*m1[2,:] - m1[0,:]
        A[1,:] = p1[1]*m1[2,:] - m1[1,:]
        A[2,:] = p2[0]*m2[2,:] - m2[0,:]
        A[3,:] = p2[1]*m2[2,:] - m2[1,:]
        
        _, _, V = np.linalg.svd(A)
        X = V[-1] / V[-1,3]
        return X
    
    def reconstruct_3d(self):
        line, img1, img2 = self._sift_feature_matching()
        
        better_kp1, better_kp2, F = self._ransac_fundamental_matrix(line, img1, img2)
        
        E = np.dot(self.K1.T, np.dot(F, self.K2))
        
        U, S, V = np.linalg.svd(E)
        m = (S[0] + S[1]) / 2
        E = np.dot(U, np.dot(np.diag([m, m, 0]), V))
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        
        solutions = [
            np.vstack((np.dot(U, np.dot(W, V)).T, U[:,2])).T,
            np.vstack((np.dot(U, np.dot(W, V)).T, -U[:,2])).T,
            np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:,2])).T,
            np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:,2])).T
        ]
        
        P1 = np.dot(self.K2, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        
        max_pt = 0
        P2_right = None
        
        better_kp1 = np.column_stack((better_kp1, np.zeros(len(better_kp1))))
        better_kp2 = np.column_stack((better_kp2, np.zeros(len(better_kp2))))
        x1 = better_kp1[:,:2]
        x2 = better_kp2[:,:2]
        
        for i, solution in enumerate(solutions):
            P2 = np.dot(self.K1, solution)
            count = sum(
                self._linear_triangulation(x1[j], x2[j], P1, P2)[3] > 0 
                for j in range(len(x1))
            )
            
            if count > max_pt:
                max_pt = count
                P2_right = P2
        
        X = [self._linear_triangulation(x1[j], x2[j], P1, P2_right)[0:3] 
             for j in range(len(x1))]
        
        return np.array(X)
    
    def visualize_3d_points(self, X):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2])
        plt.show()
        
        
    def visualize_feature_matching(self):
        line, img1, img2 = self._sift_feature_matching()
        
        better_kp1, better_kp2, F = self._ransac_fundamental_matrix(line, img1, img2)
        
        better_kp1 = np.int32(better_kp1)
        better_kp2 = np.int32(better_kp2)
        
        lines1 = cv2.computeCorrespondEpilines(better_kp2.reshape(-1, 1, 2), 2, F)
        lines2 = cv2.computeCorrespondEpilines(better_kp1.reshape(-1, 1, 2), 1, F)
        
        lines1 = lines1.reshape(-1, 3)
        lines2 = lines2.reshape(-1, 3)
        
        img1_epipolar = img1.copy()
        img2_epipolar = img2.copy()
        
        for line, pt1, pt2 in zip(lines1, better_kp1, better_kp2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -(line[2]) / line[1]])
            x1, y1 = map(int, [img1_epipolar.shape[1], -(line[2] + line[0] * img1_epipolar.shape[1]) / line[1]])
            
            cv2.line(img2_epipolar, (x0, y0), (x1, y1), color, 1)
            cv2.circle(img1_epipolar, tuple(pt1), 5, color, -1)
            cv2.circle(img2_epipolar, tuple(pt2), 5, color, -1)
        
        combined_img = np.hstack((img1_epipolar, img2_epipolar))
        
        plt.figure(figsize=(20, 10))
        plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
        plt.title('Feature Matching and Epipolar Lines')
        plt.axis('off')
        plt.show()
        
        return better_kp1, better_kp2, F
    
    
    def main(self):
        better_kp1, better_kp2, F = self.visualize_feature_matching()
        
        X = self.reconstruct_3d()
        
        np.savetxt('3D.txt', X)
        
        self.visualize_3d_points(X)

def main():
    img1_path = './data/Mesona1.JPG'
    img2_path = './data/Mesona2.JPG'
    
    reconstructor = ImageReconstruction(img1_path, img2_path)
    reconstructor.main()

if __name__ == "__main__":
    main()