import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches


def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...
    warp_im = cv2.warpPerspective(im2, H2to1, (im1.shape[1]+800, im1.shape[0]))
    
    # save file
    # cv2.imwrite('../results/6_1.jpg', warp_im)
    # np.save('../results/q6_1.npy', H2to1)
    
    cv2.imshow('img',warp_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # blend
    # pano_im  = cv2.addWeighted(im1, 0.5, warp_im, 0.5, 0.0)
    return None


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
   
    im1TL = np.array([0, 0, 1])
    im1TR = np.array([im1.shape[1], 0, 1])
    im1BL = np.array([0, im1.shape[0], 1])
    im1BR = np.array([im1.shape[1], im1.shape[0], 1])

    im2TL = np.array([0, 0, 1], dtype='int')
    im2TR = np.array([im2.shape[1], 0, 1], dtype='int')
    im2BL = np.array([0, im2.shape[0], 1], dtype='int')
    im2BR = np.array([im2.shape[1], im2.shape[0], 1], dtype='int')
    
    im2TL = (np.matmul(H2to1, im2TL))
    im2TR = (np.matmul(H2to1, im2TR))
    im2BL = (np.matmul(H2to1, im2BL))
    im2BR = (np.matmul(H2to1, im2BR))
    
    im2TL = (im2TL / im2TL[2]).astype(int)
    im2TR = (im2TR / im2TR[2]).astype(int)
    im2BL = (im2BL / im2BL[2]).astype(int)
    im2BR = (im2BR / im2BR[2]).astype(int)

    L = min(im1TL[0], im1TR[0], im1BL[0], im1BR[0], \
        im2TL[0],im2TR[0],im2BL[0],im2BR[0])
    R = max(im1TL[0], im1TR[0], im1BL[0], im1BR[0], \
        im2TL[0],im2TR[0],im2BL[0],im2BR[0])
    T = min(im1TL[1], im1TR[1], im1BL[1], im1BR[1], \
        im2TL[1],im2TR[1],im2BL[1],im2BR[1])
    B = max(im1TL[1], im1TR[1], im1BL[1], im1BR[1], \
        im2TL[1],im2TR[1],im2BL[1],im2BR[1])
    
    TL = [L, T]
    TR = [R, L]
    BL = [L, B]
    BR = [R, B]

    wRaw = TR[0] - TL[0]
    hRaw = BL[1] - TL[1]
    
    w = R - L
    h = (int)(w / wRaw * hRaw)

    M = np.identity(3)
    M[1, 2] = -T
    M[0, 2] = -L
    
    im1 = cv2.warpPerspective(im1, M, (w, h))
    warp_im = cv2.warpPerspective(im2, np.matmul(M, H2to1), (w, h))
    #  blend
    pano_im  = np.maximum(im1, warp_im)
    return pano_im

def generatePanorama(im1, im2):
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # H2to1 = np.load('../results/q6_1.npy')
    return imageStitching_noClip(im1, im2, H2to1)

if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')
    
    # scale to debug
    # im1 = cv2.resize(im1, ((int)(im1.shape[1]/5), (int)(im1.shape[0]/5)))
    # im2 = cv2.resize(im2, ((int)(im2.shape[1]/5), (int)(im2.shape[0]/5)))
    
    # uncomment this for new img
    # locs1, desc1 = briefLite(im1)
    # locs2, desc2 = briefLite(im2)
    # matches = briefMatch(desc1, desc2)
    # plotMatches(im1,im2,matches,locs1,locs2)
    

    # for new image
    # H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    # for pre-calculated H2to1
    # H2to1 = np.load('../results/q6_1.npy')
    
    # pano_im = imageStitching_noClip(im1, im2, H2to1)
    # print(H2to1)

    pano_im = generatePanorama(im1, im2)
    
    cv2.imwrite('../results/panoImg.png', pano_im)
    cv2.imshow('panoramas', pano_im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()