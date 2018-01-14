## Detection format

The detections are stored in the following order (similar to the detection
format from Pascal VOC):

For each folder corresponding to a discretization angle, each file contains the
detection scores for a specific class.

Each line in a file corresponds to one bounding box, where the first entry is
the image name, followed by the 4 coordinates of the bounding boxes, followed
by the detection score and finaly the discretized viewpoint.

For example

2008_000021 26 174 482 290 0.997655 1

correspond to a detection in image 2008_000021,
with bounding box `[26 174 482 290]` in `x1, y1, x2, y2` order,
with detection score 0.997655 and it was attributed to the discretized viewpoint
1.

### Discretization of the viewpoints

For a given number of discretized elements (for example for AVP4), the angles
were discretized using the following function

```
function [discrete_ang] = discretize(ang, nPoses)
  divAng = 360/nPoses;
  discrete_ang = mod(ceil((ang-divAng/2)/divAng),nPoses)+1;
end
```
