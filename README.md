# Panorama Stitcher

## Demostration

- BRIEF feature matching:
  ![img](results/textbookMatching.jpg)
- Panorama stitching:
  - Original images
    <img src="data/incline_L.png" width="180"> <img src="data/incline_R.png" width="180">
  - Stitched image
    <img src="results/q6_3.jpg" width="360">

## Algorithms

- Keypoint detection
  - Difference of Gaussian (DoG) detector
  - SIFT detector
- Feature matching
  - BRIEF feature matching
  - RANSAC