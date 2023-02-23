# Panorama Stitcher

## Demostration

- BRIEF feature matching:

  <img src="results/textbookMatching.png" width="600">
- Panorama stitching:
  - Original images

    <img src="data/incline_L.png" width="300"> <img src="data/incline_R.png" width="300">
  - Stitched image
  
    <img src="results/q6_3.jpg" width="600">

## Algorithms

- Keypoint detection
  - Difference of Gaussian (DoG) detector
  - SIFT detector
- Feature matching
  - BRIEF feature matching
  - RANSAC