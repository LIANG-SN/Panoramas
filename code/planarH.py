import numpy as np
import cv2
from BRIEF import briefLite, briefMatch

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...
    
    # for-loop method
    # A = []
    # for i in range(0, p1.shape[1]):
    #     A.append([-p2[0, i], -p2[1, i], -1, 0, 0, 0, p1[0, i]*p2[0, i], p1[0, i]*p2[1, i], p1[0, i]])
    #     A.append([0, 0, 0, -p2[0, i], -p2[1, i], -1, p1[1, i]*p2[0, i], p1[1, i]*p2[1, i], p1[1, i]])
    
    # non-for-loop method
    A = np.zeros((p1.shape[1] * 2, 9))
    A[0:p1.shape[1], 0] = -p2[0, :]
    A[0:p1.shape[1], 1] = -p2[1, :]
    A[0:p1.shape[1], 2] = -1
    A[0:p1.shape[1], 3] = 0
    A[0:p1.shape[1], 4] = 0
    A[0:p1.shape[1], 5] = 0
    A[0:p1.shape[1], 6] = p1[0, :]*p2[0, :]
    A[0:p1.shape[1], 7] = p1[0, :]*p2[1, :]
    A[0:p1.shape[1], 8] = p1[0, :]

    A[p1.shape[1]: , 0] = 0
    A[p1.shape[1]: , 1] = 0
    A[p1.shape[1]: , 2] = 0
    A[p1.shape[1]: , 3] = -p2[0, :]
    A[p1.shape[1]: , 4] = -p2[1, :]
    A[p1.shape[1]: , 5] = -1
    A[p1.shape[1]: , 6] = p1[1, :]*p2[0, :]
    A[p1.shape[1]: , 7] = p1[1, :]*p2[1, :]
    A[p1.shape[1]: , 8] = p1[1, :]

    u, s, vh = np.linalg.svd(np.array(A))
    # v = vh.transpose()
    # h = vh[np.argmin(s)] # check this
    # H2to1 = np.array([h[0:3], h[3:6], h[6:9]])
    H2to1 = vh[-1].reshape(3,3)
    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...
    # append last column: all points in homogeneous
    pt1 = np.ones((matches.shape[0], 3))
    pt2 = np.ones((matches.shape[0], 3))
    pt1[:, 0:2] = locs1[matches[:,0], 0:2]
    pt2[:, 0:2] = locs2[matches[:,1], 0:2]
   
    maxInliers = -1
    bestH = np.zeros((3,3))
    bestInliers = []
    # ransac loop
    for i in range(num_iter):
        # random choose 4 pairs
        sample = np.random.randint(0, matches.shape[0], 4)
        # sample points
        p1 = pt1[sample, 0:2].transpose() # 4x2 to 2x4
        p2 = pt2[sample, 0:2].transpose()
        H = computeH(p1, p2)
        numInliers = 0
        inliers = []
        # check inliers, keep H if max
        for k in range(matches.shape[0]):
            pt2Trans = np.matmul(H, pt2[k])
            pt2Trans /= pt2Trans[-1] # normalize the point, important
            dist = np.sqrt(np.sum((pt1[k] - pt2Trans) ** 2))
            if k in sample:
                continue
            elif dist < tol:
                numInliers += 1
                inliers.append(k)
        if numInliers > maxInliers:
            maxInliers = numInliers
            bestH = H
            bestInliers = np.append(sample, inliers)
    bestInliers = np.array(bestInliers, dtype=int)
    # recompute best H with inliers
    p1 = pt1[bestInliers, 0:2].transpose()
    p2 = pt2[bestInliers, 0:2].transpose()
    bestH = computeH(p1, p2)

    return bestH

if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
