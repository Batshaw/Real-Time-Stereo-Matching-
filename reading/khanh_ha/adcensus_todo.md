# Cost Initialization Improvement
- Modified Color Census Transform in  Gaussian color model space due because RGB is sensitive to intensity change
- Add gradient costs
- Combine 3 costs: census, SAD, gradient

# Region Voting optimization
- approximate region voting by separate vertical and horizontal aggregation.(performance could degrade)(Stereo Matching and Viewpoint Synthesis FPGA Implementation - liao2012)

# Small Hole Filling (Local Stereo Matching with Improved Matching Cost and Disparity Refinement)

# [Adaptive cross-trilateral depth map filtering](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5506336)

# Uniqueness of minimum/maximum [section 4.2.1 overview of stereo matching]
it is possible to have a double minimum but not a triple minimum or more. Based on that assumption,
by investigating the difference between the best and the worst of these three scores, it is possible
to decide if a match is of high or low confidence

# Cross-Scale Cost aggregration
- build Gaussian Pyramid
- generate cost volume at each scale
- aggregate cost volume at eac scale
- aggregate cost volumes across multiple scales.