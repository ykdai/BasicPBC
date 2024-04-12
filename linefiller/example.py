import numpy as np
import cv2
from linefiller.trappedball_fill import trapped_ball_fill_multi, flood_fill_multi, mark_fill, build_fill_map, merge_fill, \
    show_fill_map
from linefiller.thinning import thinning

im = cv2.imread('example.png', cv2.IMREAD_GRAYSCALE)
ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

fills = []
result = binary

fill = trapped_ball_fill_multi(result, 3, method='max')
fills += fill
result = mark_fill(result, fill)

fill = trapped_ball_fill_multi(result, 2, method=None)
fills += fill
result = mark_fill(result, fill)

fill = trapped_ball_fill_multi(result, 1, method=None)
fills += fill
result = mark_fill(result, fill)

fill = flood_fill_multi(result)
fills += fill

fillmap = build_fill_map(result, fills)
cv2.imwrite('fills.png', show_fill_map(fillmap))

fillmap = merge_fill(fillmap)

cv2.imwrite('fills_merged.png', show_fill_map(fillmap))

cv2.imwrite('fills_merged_no_contour.png', show_fill_map(thinning(fillmap)))
