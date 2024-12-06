import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageMatcher:
    def __init__(self, img1_path, img2_path):
        self.img1_path = img1_path
        self.img2_path = img2_path
        self.K1 = np.array([[5426.566895, 0.678017, 330.096680],
                             [0.000000, 5423.133301, 648.950012],
                             [0.000000, 0.000000, 1.000000]])
        self.K2 = np.array([[5426.566895, 0.678017, 387.430023],
                             [0.000000, 5423.133301, 620.616699],
                             [0.000000, 0.000000, 1.000000]])

    def detect_and_compute_sift(self):
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)
        gray1 = cv2.imread(self.img1_path, 0)
        gray2 = cv2.imread(self.img2_path, 0)

        while img2.shape[0] > 1000:
            if img1.shape == img2.shape:
                img1 = cv2.resize(img1, None, fx=0.5, fy=0.5)
                gray1 = cv2.resize(gray1, None, fx=0.5, fy=0.5)
            img2 = cv2.resize(img2, None, fx=0.5, fy=0.5)
            gray2 = cv2.resize(gray2, None, fx=0.5, fy=0.5)

        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        return img1, img2, kp1, kp2, des1, des2

    def match_features(self, ratio=0.5, show=False):
        img1, img2, kp1, kp2, des1, des2 = self.detect_and_compute_sift()
        
        matches = []
        for i in range(des1.shape[0]):
            des1_ = np.tile(des1[i], (des2.shape[0], 1))
            error = des1_ - des2
            SSD = np.sum((error**2), axis=1)
            idx_sort = np.argsort(SSD)
            
            if SSD[idx_sort[0]] < ratio * SSD[idx_sort[1]]:
                matches.append([kp1[i].pt, kp2[idx_sort[0]].pt])
        
        line = np.array(matches)
        
        if show:
            self._visualize_matches(img1, img2, line[:, 0], line[:, 1])
        
        return line, img1, img2

    def _visualize_matches(self, img1, img2, kp1, kp2):
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        line = np.zeros((kp1.shape[0], 2, 2))
        
        for n in range(kp1.shape[0]):
            line[n, :, :] = np.vstack((kp1[n], kp2[n]))

        plt.figure(figsize=(10, 10))
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

    def normalize_points(self, pts_1, pts_2):
        h1, w1, _ = cv2.imread(self.img1_path).shape
        h2, w2, _ = cv2.imread(self.img2_path).shape
        
        T1 = np.array([[2.0/w1, 0, -1], [0, 2/h1, -1], [0, 0, 1.0]])
        T2 = np.array([[2.0/w2, 0, -1], [0, 2/h2, -1], [0, 0, 1.0]])
        
        x = np.zeros(shape=(3,1))
        x[2,0] = 1
        
        for i in range(8):
            for pts, T in [(pts_1, T1), (pts_2, T2)]:
                x[0,0] = pts[i,0]
                x[1,0] = pts[i,1]
                pts[i,0] = np.dot(T, x)[0,0]
                pts[i,1] = np.dot(T, x)[1,0]
        
        return pts_1, pts_2, T1, T2

    def compute_fundamental_matrix(self, better_kp1, better_kp2, T1, T2):
        A = []
        for j in range(len(better_kp1)):
            x1, y1 = better_kp1[j][0], better_kp1[j][1]
            x2, y2 = better_kp2[j][0], better_kp2[j][1]
            A.append(np.asarray([x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]))
        
        _, _, Vt = np.linalg.svd(A)
        Fi = Vt[-1].reshape(3,3)
        U, S, V = np.linalg.svd(Fi)
        
        S1 = np.zeros((3,3))
        S1[0,0] = S[0]
        S1[1,1] = S[1]
        
        F = (U.dot(S1)).dot(V)
        F = np.dot(np.transpose(T2), np.dot(F, T1))
        F /= F[2,2]
        
        return F

    def ransac_fundamental_matrix(self, line, error_threshold=2, show=False):
        kp1, kp2 = line[:, 0], line[:, 1]
        iterations = 3000
        max_inliers = 0
        kp_num = len(kp1)
        
        for _ in range(iterations):
            kp1_rand = np.zeros((8,2), dtype="f")
            kp2_rand = np.zeros((8,2), dtype="f")
            
            for j in range(8):
                rand = np.random.randint(0, kp_num-1)
                kp1_rand[j,0] = kp2[rand][0]
                kp1_rand[j,1] = kp2[rand][1]
                kp2_rand[j,0] = kp1[rand][0]
                kp2_rand[j,1] = kp1[rand][1]
            
            pts_1, pts_2, T1, T2 = self.normalize_points(kp1_rand, kp2_rand)
            
            F = self.compute_fundamental_matrix(pts_1, pts_2, T1, T2)
            
            pts_tmp1, pts_tmp2, inliers = self._count_inliers(kp2, kp1, F)
            
            if inliers > max_inliers:
                max_inliers = inliers
                better_kp1 = pts_tmp1
                better_kp2 = pts_tmp2
                best_F = F
        
        if show:
            self._visualize_matches(self.img1, self.img2, np.array(better_kp1), np.array(better_kp2))
        
        return better_kp1, better_kp2, best_F

    def _count_inliers(self, pts1, pts2, F):
        num = pts1.shape[0]
        inliers = 0
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
                pts_tmp1.append([x1, y1])
                pts_tmp2.append([x2, y2])
                inliers += 1
        
        return pts_tmp1, pts_tmp2, inliers

    def triangulate_points(self, x1, x2):
        P1 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
        P1 = np.dot(self.K2, P1)
        
        best_solution_idx = self._find_best_projection_matrix(x1, x2, P1)
        
        P2 = np.dot(self.K1, self.projection_solutions[best_solution_idx])
        
        X = []
        for j in range(len(x1)):
            X.append(self._linear_triangulation(x1[j], x2[j], P1, P2)[0:3])
        
        return np.array(X)

    def _find_best_projection_matrix(self, x1, x2, P1):
        U, S, V = np.linalg.svd(self._compute_essential_matrix())
        W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
        
        self.projection_solutions = []
        self.projection_solutions.append(np.vstack((np.dot(U, np.dot(W, V)).T, U[:,2])).T)
        self.projection_solutions.append(np.vstack((np.dot(U, np.dot(W, V)).T, -U[:,2])).T)
        self.projection_solutions.append(np.vstack((np.dot(U, np.dot(W.T, V)).T, U[:,2])).T)
        self.projection_solutions.append(np.vstack((np.dot(U, np.dot(W.T, V)).T, -U[:,2])).T)
        
        max_pt = 0
        best_solution_idx = 0
        
        for i in range(4):
            P2 = np.dot(self.K1, self.projection_solutions[i])
            count = self._count_positive_depth_points(x1, x2, P1, P2)
            
            if count > max_pt:
                max_pt = count
                best_solution_idx = i
        
        return best_solution_idx

    def _count_positive_depth_points(self, x1, x2, P1, P2):
        count = 0
        for j in range(len(x1)):
            x = self._linear_triangulation(x1[j], x2[j], P1, P2)
            v = np.dot(x, P2.T)
            if v[2] > 0:
                count += 1
        return count

    def _linear_triangulation(self, p1, p2, m1, m2):
        A = np.zeros((4, 4))
        A[0,:] = p1[0] * m1[2,:] - m1[0,:]
        A[1,:] = p1[1] * m1[2,:] - m1[1,:]
        A[2,:] = p2[0] * m2[2,:] - m2[0,:]
        A[3,:] = p2[1] * m2[2,:] - m2[1,:]
        
        _, _, V = np.linalg.svd(A)
        X = V[-1] / V[-1, 3]
        return X

    def _compute_essential_matrix(self):
        line, img1, img2 = self.match_features()
        better_kp1, better_kp2, F = self.ransac_fundamental_matrix(line)
        
        E = np.dot(self.K1.T, np.dot(F, self.K2))
        
        U, S, V = np.linalg.svd(E)
        m = (S[0] + S[1]) / 2
        E = np.dot(U, np.dot(np.diag([m, m, 0]), V))
        
        return E

    def visualize_3d_points(self, X):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x, y, z = X[:,0], X[:,1], X[:,2]
        ax.scatter(x, y, z)
        plt.show()
        
    def draw_epipolar_lines(self, F):
        img1 = cv2.imread(self.img1_path)
        img2 = cv2.imread(self.img2_path)

        line, _, _ = self.match_features(ratio=0.65, show=False)
        pts1 = np.array(line[:, 0]).astype(np.int32)
        pts2 = np.array(line[:, 1]).astype(np.int32)

        pts1 = np.concatenate((pts1, np.ones(pts1.shape[0]).reshape(-1, 1)), axis=1)
        pts2 = np.concatenate((pts2, np.ones(pts2.shape[0]).reshape(-1, 1)), axis=1)

        lines = F.dot(pts1.T)

        r, c, _ = img1.shape
        for line_params, pt1, pt2 in zip(lines.T, pts1, pts2):
            color = tuple(np.random.randint(0, 255, 3).tolist())

            x0, y0 = map(int, [0, -(line_params[2]) / line_params[1]])
            x1, y1 = map(int, [c, -(line_params[2] + line_params[0] * c) / line_params[1]])

            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
            
            img1 = cv2.circle(img1, tuple(map(int, pt1[:2])), 5, color, -1)
            img2 = cv2.circle(img2, tuple(map(int, pt2[:2])), 5, color, -1)

        return img1, img2

    def visualize_epipolar_geometry(self, F):
        output1, output2 = self.draw_epipolar_lines(F)

        cv2.imshow('Image 1 with Points', output1)
        cv2.imshow('Image 2 with Epipolar Lines', output2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def main_reconstruction_pipeline(self):
        line, img1, img2 = self.match_features(ratio=0.65, show=False)

        better_kp1, better_kp2, F = self.ransac_fundamental_matrix(line, error_threshold=2, show=False)

        E = np.dot(self.K1.T, np.dot(F, self.K2))

        self.visualize_epipolar_geometry(F)
        
        better_kp1 = np.array(better_kp1)
        better_kp2 = np.array(better_kp2)

        x1 = better_kp1[:, 0:2]
        x2 = better_kp2[:, 0:2]
        X = self.triangulate_points(x1, x2)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c='r', marker='o')
        ax.set_title('3D Point Cloud Reconstruction')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()

        np.savetxt('2D_points.txt', x1)
        np.savetxt('3D_points.txt', X)

def main():
    img1_path = './data/Statue1.bmp'
    img2_path = './data/Statue2.bmp'
    
    matcher = ImageMatcher(img1_path, img2_path)
    matcher.main_reconstruction_pipeline()

if __name__ == "__main__":
    main()