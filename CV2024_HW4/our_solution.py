import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageReconstruction:
    def __init__(self, img1_path, img2_path):
        self.K1 = np.array([[5426.566895, 0.678017, 330.096680],
                             [0.000000, 5423.133301, 648.950012],
                             [0.000000,  0.000000, 1.000000]])
        
        self.K2 = np.array([[5426.566895, 0.678017, 387.430023],
                             [0.000000, 5423.133301, 620.616699],
                             [0.000000, 0.000000, 1.000000]])
        
        self.img1_path = img1_path
        self.img2_path = img2_path

    def _sift_feature_detection(self, img1, img2, gray1, gray2):
        while(img2.shape[0] > 1000):
            if img1.shape == img2.shape:
                img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
                gray1 = cv2.resize(gray1, None, fx=0.5, fy=0.5)
            img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)
            gray2 = cv2.resize(gray2, None, fx=0.5, fy=0.5)

        sift = cv2.xfeatures2d.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)
        
        return img1, img2, kp1, kp2, des1, des2

    def feature_matching(self, ratio=0.5, show=False):
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        gray1 = cv2.imread(self.img1_path, 0)
        gray2 = cv2.imread(self.img2_path, 0)

        img1, img2, kp1, kp2, des1, des2 = self._sift_feature_detection(img1, img2, gray1, gray2)

        match = []
        for i in range(des1.shape[0]):
            des1_ = np.tile(des1[i], (des2.shape[0], 1))
            error = des1_ - des2
            SSD = np.sum((error**2), axis=1)
            idx_sort = np.argsort(SSD)
            
            if SSD[idx_sort[0]] < ratio * SSD[idx_sort[1]]:
                match.append([kp1[i].pt, kp2[idx_sort[0]].pt])
        
        line = np.array(match)
        
        if show:
            self._show_matches(img1, img2, line[:, 0], line[:, 1])
        
        return line, img1, img2

    def _show_matches(self, img1, img2, kp1, kp2):
        plt.figure(figsize=(10,10))
        
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        line = np.zeros((kp1.shape[0], 2, 2))
        
        for n in range(kp1.shape[0]):
            line[n, :, :] = np.vstack((kp1[n], kp2[n]))

        line_tran = np.transpose(line, axes=(0, 2, 1))
        
        if img1.shape[0] != img2.shape[0]:
            img2 = np.vstack([img2, np.full((np.abs(img1.shape[0] - img2.shape[0]), img2.shape[1], 3), 255)])
        
        imgStack = np.hstack([img1[:,:,::-1], img2[:,:,::-1]])
        
        for i in range(line.shape[0]):
            color = np.random.rand(3)
            plt.scatter(line[i][0][0], line[i][0][1], c='r')
            plt.scatter(line[i][1][0] + img1.shape[1], line[i][1][1], c='r')
            plt.plot(line_tran[i][0] + [0, img1.shape[1]], line_tran[i][1], color=color)
        
        plt.xlim((0, img1.shape[1] + img2.shape[1]))
        plt.ylim((img1.shape[0], 0))
        plt.imshow(imgStack)
        plt.show()

    def _normalize_points(self, pts_1, pts_2, w1, h1, w2, h2):
        T1 = np.array([[2.0/w1, 0, -1], [0, 2/h1, -1], [0, 0, 1.0]])
        T2 = np.array([[2.0/w2, 0, -1], [0, 2/h2, -1], [0, 0, 1.0]])

        x = np.zeros(shape=(3,1))
        x[2,0] = 1
        for i in range(pts_1.shape[0]):
            x[0,0] = pts_1[i,0]
            x[1,0] = pts_1[i,1]
            pts_1[i,0] = np.dot(T1, x)[0,0]
            pts_1[i,1] = np.dot(T1, x)[1,0]

            x[0,0] = pts_2[i,0]
            x[1,0] = pts_2[i,1]
            pts_2[i,0] = np.dot(T2, x)[0,0]
            pts_2[i,1] = np.dot(T2, x)[1,0]
        
        return pts_1, pts_2, T1, T2

    def ransac_fundamental_matrix(self, line, img1, img2, error_threshold=2, show=False):
        """RANSAC to find fundamental matrix"""
        kp1, kp2 = line[:, 0], line[:, 1]
        h1, w1, _ = img1.shape
        h2, w2, _ = img2.shape
        
        iterations = 3000
        maxinlier = 0
        kp_num = len(kp1)
        
        for i in range(iterations):
            rand_indices = np.random.choice(kp_num, 8, replace=False)
            kp1_rand = kp2[rand_indices]
            kp2_rand = kp1[rand_indices]
            
            pts_1, pts_2, T1, T2 = self._normalize_points(kp1_rand, kp2_rand, w1, h1, w2, h2)
            
            F = self._find_fundamental_matrix(pts_1, pts_2, T1, T2)
            
            pts_tmp1, pts_tmp2, inlier = self._count_inliers(kp2, kp1, F)
            
            if inlier >= maxinlier:
                maxinlier = inlier
                better_kp1 = pts_tmp1
                better_kp2 = pts_tmp2
                Fstore = F
        
        if show:
            self._show_matches(img1, img2, np.array(better_kp1), np.array(better_kp2))
        
        return better_kp1, better_kp2, Fstore

    def _find_fundamental_matrix(self, kp1, kp2, T1, T2):
        A = []
        for j in range(len(kp1)):
            x1, y1 = kp1[j][0], kp1[j][1]
            x2, y2 = kp2[j][0], kp2[j][1]
            A.append(np.asarray([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]))
        
        _, _, Vt = np.linalg.svd(A)
        Fi = Vt[-1].reshape(3,3)
        U, S, V = np.linalg.svd(Fi)
        
        S1 = np.zeros((3,3))
        S1[0,0], S1[1,1] = S[0], S[1]
        
        F = (U.dot(S1)).dot(V)
        F = np.dot(np.transpose(T2), np.dot(F, T1))
        F /= F[2,2]
        
        return F

    def _count_inliers(self, pts1, pts2, F):
        num = pts1.shape[0]
        inlier = 0
        pts_tmp1 = []
        pts_tmp2 = []
        
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
    
    def draw_epipolar_lines(self, img1, img2, F, pts1, pts2):
        r, c, _ = img1.shape
        
        lines = F.dot(pts1.T)
        
        for line, pt1, pt2 in zip(lines.T, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            x0, y0 = map(int, [0, -(line[2] + line[0] * 0) / line[1]])
            x1, y1 = map(int, [c, -(line[2] + line[0] * c) / line[1]])
            
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
            
            img1 = cv2.circle(img1, tuple(pt1[:2]), 5, color, -1)
            img2 = cv2.circle(img2, tuple(pt2[:2]), 5, color, -1)
        
        return img1, img2
    
    def reconstruct_3d(self, ratio=0.65):
        line, img1, img2 = self.feature_matching(ratio=ratio, show=False)
        
        better_kp1, better_kp2, F = self.ransac_fundamental_matrix(line, img1, img2, error_threshold=2, show=False)
        
        E = np.dot(self.K1.T, np.dot(F, self.K2))
        
        pts1 = np.array(better_kp1).astype(np.int32)
        pts1 = np.concatenate((pts1, np.ones(pts1.shape[0]).reshape(-1, 1)), axis=1).astype(np.int32)
        pts2 = np.array(better_kp2).astype(np.int32)
        pts2 = np.hstack((pts2, np.ones(pts2.shape[0]).reshape(-1, 1))).astype(np.int32)
        
        output1, output2 = self.draw_epipolar_lines(img1.copy(), img2.copy(), F, pts1, pts2)
        
        cv2.imshow('Epipolar Lines - Image 1', output1)
        cv2.imshow('Epipolar Lines - Image 2', output2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
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
        
        x1 = np.array(better_kp1)[:,:2]
        x2 = np.array(better_kp2)[:,:2]
        
        np.savetxt('2D.txt', x1)
        
        P1 = np.dot(self.K2, np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]))
        max_pt = 0
        P2_right = None
        
        for i, solution in enumerate(solutions):
            P2 = np.dot(self.K1, solution)
            count = sum(1 for j in range(len(x1)) if self._linear_triangulation(x1[j], x2[j], P1, P2)[2] > 0)
            
            if count > max_pt:
                max_pt = count
                P2_right = P2
        
        X = [self._linear_triangulation(x1[j], x2[j], P1, P2_right)[0:3] for j in range(len(x1))]
        X = np.array(X)
        
        np.savetxt('3D.txt', X)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2])
        plt.show()
        
        return X


if __name__ == "__main__":
    img1_path = './data/Mesona1.JPG'
    img2_path = './data/Mesona2.JPG'
    
    reconstruction = ImageReconstruction(img1_path, img2_path)
    points_3d = reconstruction.reconstruct_3d()
    
    np.savetxt('3D.txt', points_3d)