import numpy as np
from planarH import computeH
import matplotlib.pyplot as plt
def compute_extrinsics(K, H):
    
    H_ = np.matmul(np.linalg.inv(K), H)
    U, S, Vh = np.linalg.svd(H_[:, 0:2])
    I = np.zeros((3, 2))
    I[0, 0] = 1
    I[1, 1] = 1
    R12 = np.matmul(np.matmul(U, I), Vh)
    R3 = np.cross(R12[:, 0], R12[:, 1])
    R = np.zeros((3,3))
    R[:, 0:2] = R12
    R[:, 2] = R3.T # need T?
    if np.linalg.det(R) == -1:
        R[:, 2] *= -1
    scale = np.divide(H_[:, 0:2], R12, out=np.zeros_like(H_[:, 0:2]), where=R12!=0).sum() / 6
    t = H_[:, 2] / scale
    
    return R, t

def project_extrinsics(K, W, R, t):
    with open('../data/sphere.txt', 'r') as f:
        lines = f.readlines()
    pts = []
    for line in lines:
      line = line.strip().split('  ')
    #   results = [float(p) for p in i]
      coord_list = np.array(line, dtype='float')
      pts.append(coord_list)

    pts.append(np.ones(coord_list.shape[0])) # no scope?
    pts = np.array(pts) # 4 x N

    E = np.zeros((3,4))
    E[:, 0:3] = R
    E[:, 3] = t
    P = np.matmul(K, E)
    
    projected = np.matmul(P, pts)
    projected[:, :] /= projected[2, :]

    # translate the ball to O
    projected[0, :] += 325
    projected[1, :] += 600

    im = plt.imread('../data/prince_book.jpeg')
    implot = plt.imshow(im)
    plt.scatter(x=projected[0, :], y=projected[1, :], c='y', s=2)
    plt.show()
    plt.close()
    
if __name__ == '__main__':
    K = np.array([[3043.72,0.0,1196.00],[0.0,3043.72,1604.00],[0.0,0.0,1]])
    W = np.array([[0.0, 18.2, 18.2, 0.0],[0.0, 0.0, 26.0, 26.0]])
    D = np.array([[483, 1704, 2175, 67],[810, 781, 2217, 2286]])
    H = computeH(D, W)

    R, t = compute_extrinsics(K, H)

    project_extrinsics(K, W, R, t)

    