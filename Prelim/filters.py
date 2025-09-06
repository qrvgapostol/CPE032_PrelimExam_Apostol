import cv2
import numpy as np

#StrokeEdge
def strokeEdges(src, dst, blurKsize = 7, edgesKsize = 5):
    if blurKsize >= 3:
        blurredSrc = cv2.medianBlur(src, blurKsize)
        graySrc = cv2.cvtColor(blurredSrc, cv2.COLOR_BGR2GRAY)
    else:
        graySrc = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.Laplacian(graySrc, cv2.CV_8U, graySrc, ksize = edgesKsize)
    normalizedInverseAlpha = (1.0/255) * (255 - graySrc)

    channels = cv2.split(src)
    for channel in channels:
        channel[:] = channel * normalizedInverseAlpha
    cv2.merge(channels, dst)

# Canny Edge Detect
def cannyEdges(src, dst, lowThreshold=100, highThreshold=200):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, lowThreshold, highThreshold)
    dst[:] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Contour Detect
def findContours(src, dst):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dst[:] = src.copy()

    for c in contours:
        # Find the bounding box coordinates
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(dst, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find minimum area
        rect = cv2.minAreaRect(c)

        # Calculate coordinates of the minimum area rectangle
        box = cv2.boxPoints(rect)

        # Normalize coordinates to integers
        box = np.int64(box)

        # Draw contours
        cv2.drawContours(dst, [box], 0, (0, 0, 255), 2)

        # Calculate center and radius of minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(c)

        # Cast to integers
        center = (int(x), int(y))
        radius = int(radius)

        # Draw the circle
        cv2.circle(dst, center, radius, (255, 0, 0), 2)

    # Draw all contours thin
    cv2.drawContours(dst, contours, -1, (255, 255, 0), 1)

class VFuncFilter(object):
    def __init__(self, vFunc=None, dtype=np.uint8):
        self._vFunc = vFunc
        self._dtype = dtype

    def apply(self, src, dst):
        srcFlatView = src.reshape((-1, src.shape[2]))
        dstFlatView = dst.reshape((-1, dst.shape[2]))
        dstFlatView[:] = self._vFunc(srcFlatView).astype(self._dtype)

#CurveFilter
class BGRPortaCurveFilter(VFuncFilter):
    def __init__(self, dtype=np.uint8):
        def curve(v):
            v = np.clip(v, 0, 255)
            return np.where(v < 128,
                            v * 0.9,
                            255 - (255 - v) * 0.7)
        super().__init__(curve, dtype)

#CircleDetect
def findCircles(src, dst, blur_ksize=15, dp=1, minDist=30, 
                param1=100, param2=40, minRadius=0, maxRadius=0):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_ksize)

    # Detect circles
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp,
        minDist,
        param1=param1,
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(dst, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(dst, (i[0], i[1]), 2, (0, 0, 255), 3)