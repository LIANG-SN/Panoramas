import BRIEF as brief
import numpy as np
import cv2
import matplotlib.pyplot as plt
if __name__ == '__main__':

    im = cv2.imread('../data/model_chickenbroth.jpg')
    rows,cols = im.shape[0], im.shape[1]
    
    numMatch = []
    x = range(0, 360, 10)
    for deg in x:
        M = cv2.getRotationMatrix2D((cols/2,rows/2), deg, 1)
        dst = cv2.warpAffine(im,M,(cols,rows))
        
        locs1, desc1 = brief.briefLite(im)
        locs2, desc2 = brief.briefLite(dst)
        matches = brief.briefMatch(desc1, desc2)
        numMatch.append(matches.shape[0])
        
        # print(matches.shape[0])
        # brief.plotMatches(im,dst,matches,locs1,locs2)
    
    plt.bar(x, numMatch)
    plt.show()
