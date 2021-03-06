import numpy as np
import cv2

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    # compute DoG_pyramid here
    for i in range(0, len(levels) - 1):
        # why not [i+1] - [i] ?
        DoG_pyramid.append(gaussian_pyramid[:, :, i + 1] - gaussian_pyramid[:, :, i])
    DoG_pyramid = np.stack(DoG_pyramid, axis= -1)
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    ##################
    # TO DO ...
    # Compute principal curvature here
    principal_curvature = []
    h = DoG_pyramid.shape[0]
    w = DoG_pyramid.shape[1]
    for k in range(0, DoG_pyramid.shape[2]):
      Dxx = cv2.Sobel(DoG_pyramid[:, :, k], cv2.CV_32F, 2, 0)
      Dxy = cv2.Sobel(DoG_pyramid[:, :, k], cv2.CV_32F, 1, 1)
      Dyy = cv2.Sobel(DoG_pyramid[:, :, k], cv2.CV_32F, 0, 2)
      curvature_level = np.zeros((h, w))
      
      # non-for-loop method
      H = np.zeros((h, w, 2, 2))
      H[:, :, 0, 0] = Dxx[:, :]
      H[:, :, 0, 1] = Dxy[:, :]
      H[:, :, 1, 0] = Dxy[:, :]
      H[:, :, 1, 1] = Dyy[:, :]
      eig = np.linalg.eigvals(H[:, :])
      curvature_level[:, :] = ((eig[:, :, 0] + eig[:, :, 1]) ** 2) / np.linalg.det(H[:, :])
      
      # for loop method
    #   for i in range(0, h):
    #       for j in range(0, w):
    #         H = np.array([[Dxx[i, j], Dxy[i, j]], [Dxy[i, j], Dyy[i, j]]])
    #         eig = np.linalg.eigvals(H)
    #         if np.linalg.det(H) != 0:
    #             curvature_level[i, j] = (H.trace() ** 2) / np.linalg.det(H)
    #         # test points bigger than thrshold
    #         # if curvature_level[i, j] > 12:
    #         #     print(i, j)
      principal_curvature.append(curvature_level)
    principal_curvature = np.stack(principal_curvature, axis=-1) # change the shape, let L be the last dim
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    locsDoG = None
    ##############
    #  TO DO ...
    # Compute locsDoG here
    padded = np.zeros(np.array(DoG_pyramid.shape) + 2)
    padded[1 : -1 , 1 : -1, 1 : -1] = DoG_pyramid
    compare_max = np.zeros([10, *DoG_pyramid.shape])
    compare_min = np.zeros([10, *DoG_pyramid.shape])
    for c, p, pad in [(compare_max, DoG_pyramid, padded), (compare_min, -DoG_pyramid, -padded)]:
        c[0] = p > pad[:-2,  :-2,  1 : -1] # (x, y) > (x-1, y-1)
        c[1] = p > pad[:-2,  1:-1, 1:-1]
        c[2] = p > pad[:-2,  2:,   1:-1]
        c[3] = p > pad[1:-1, :-2,  1:-1]
        c[4] = p > pad[1:-1, 2:,   1:-1]
        c[5] = p > pad[2:,   :-2,  1:-1]
        c[6] = p > pad[2:,   1:-1, 1:-1]
        c[7] = p > pad[2:,   2:,   1:-1]
        c[8] = p > pad[1:-1, 1:-1, :-2]
        c[9]= p > pad[1:-1, 1:-1, 2:]
    max = np.ones(DoG_pyramid.shape)
    min = np.ones(DoG_pyramid.shape)
    for i in range(0, 10):
        max = np.logical_and(max, compare_max[i])
        min = np.logical_and(min, compare_min[i])
    extrema = np.logical_or(max, min)
    
    magnitude_check = np.zeros(DoG_pyramid.shape)
    curvature_check = np.zeros(DoG_pyramid.shape)
    final = extrema
    
    magnitude_check = DoG_pyramid > th_contrast
    curvature_check = principal_curvature < th_r
    final = np.logical_and(final, magnitude_check)
    final = np.logical_and(final, curvature_check)

    locsDoG = np.stack(np.nonzero(final), axis=-1) # change 3xN to Nx3
    locsDoG[:, [0,1]] = locsDoG[:, [1,0]] # swap the y, x column to x, y
    #print(locsDoG.shape)
    return locsDoG
    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid = createGaussianPyramid(im, sigma0, k, levels)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    
    return locsDoG, gauss_pyramid




if __name__ == '__main__':
    # test gaussian pyramid
    #levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')
    #im_pyr = createGaussianPyramid(im)
    #displayPyramid(im_pyr)
    
    # test DoG pyramid
    # DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    # pc_curvature = computePrincipalCurvature(DoG_pyr)
    #displayPyramid(pc_curvature)
    
    # test get local extrema
    # th_contrast = 0.03
    # th_r = 12
    # locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    
    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)

    for i in range(0, locsDoG.shape[0]):
        coord = (locsDoG[i, 0], locsDoG[i, 1]) # check the coordinate
        im = cv2.circle(im, coord, 1, color=(255,0,0), thickness=-1)
    cv2.imshow('Discriptors', im)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()


