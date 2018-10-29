from PIL import Image, ImageDraw
#from global_config import config
import os, sys
pathname = os.path.dirname(sys.argv[0])
abs_path = pathname.split('/')[:-1]
sys.path.append('/'.join(abs_path))
from tools.color_map import range2color
#from color_map import range2color
import numpy as np

def draw_keypoint(source_img, points):
    draw = ImageDraw.Draw(source_img)

    for i, s in enumerate(points):
        if len(s) == 0:
            continue
        for j in s:

            color = range2color(i,
                                len(points))

            draw.ellipse((j[0]-3,
                          j[1]-3,
                          j[0]+3,
                          j[1]+3),
                         fill=color)#upper left and lower right.

def draw_lines(source_img, points):
    draw = ImageDraw.Draw(source_img)

    for i, s in enumerate(points):
        for j in s:
            color = range2color(i,
                                len(points))

            draw.line((j[0][0],
                       j[0][1],
                       j[1][0],
                       j[1][1]),
                    fill=color)#upper left and lower right.

def draw_points(img, points):
    img = np.transpose(img, [1, 0, 2])*255
    source_img = Image.fromarray(np.uint8(img))

    points = [list(j * 4) for j in points]
    draw_keypoint(source_img, [points])
    source_img.show()
    
def draw_graph(img, graph):
    img = np.transpose(img, [1, 0, 2])*255
    graph *= 255
    graph = np.transpose(graph, [1, 0])
    graph = Image.fromarray(np.uint8(graph)).resize([img.shape[1], img.shape[0]])
    img[:, :, 2] += np.array(graph)
    #img = np.array(graph)
    img[img>255] = 255
    source_img = Image.fromarray(np.uint8(img))
    return source_img
