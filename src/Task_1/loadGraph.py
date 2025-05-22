import cv2
import numpy as np
from PathGenerator import PathHandler
from Graph import *


        
def on_mouse(event, x, y, flags, user_param):
    """
    event: one of cv2.EVENT_*
    x,y:   coordinates within the window
    user_param: whatever you passed in when setting the callback
    """
    global dispMap, baseMap, vehicleX, vehicleY, racetrack
    if event == cv2.EVENT_LBUTTONDOWN:
        dispMap = baseMap.copy()
        vehicleX = x; vehicleY = y

        edge = graph.currentEdge(
            np.array([vehicleX, vehicleY]),
            numSearchNodes=5
        )
        dispMap = racetrack.drawSingleEdge(graph, edge, dispMap, thickness = 1, head_width_px = 7, head_length_px = 7, color = ((100, 0, 0), (255, 100, 0)))
        cv2.circle(dispMap, (x, y), 3, (255, 100), cv2.FILLED, cv2.LINE_AA)

        
if __name__ == "__main__":

    graph = Grapher("./Competition_track_graph.graphml", scale = 72.5, shift = [0, 0], eps = 2)
    path = randomPath(graph)
    points = graph['nodeOrig', path]
    handler = PathHandler(points)
   
    if handler is not None:
        dispPath = handler.definedPath.copy()
        yMin, yMax = graph.min[1], graph.max[1]
        dispPath[1] = yMax - (dispPath[1] - yMin)

    

    img = cv2.imread("./Track.png")
    racetrack = (
        MapVisualizer(img)
            .resize(ratio = 9766 / 1500)
            .drawNodes(graph, radius = 2)
            # .drawEdges(graph, thickness = 1, head_width_px = 7, head_length_px = 7)
            # .drawLabels(graph, scale = .4, line_type = cv2.LINE_AA, thickness = 1, color = (25, 163, 233), font = cv2.FONT_HERSHEY_TRIPLEX)
            .drawPath(graph, path, thickness = 1, head_width_px = 7, head_length_px = 7, color = ((0, 255, 100), (0, 255, 100)))
            .drawClothoidPath(dispPath, thickness = 1)
    )

    vehicleX = 0; vehicleY = 0
    edge = graph.currentEdge(np.array([vehicleX, vehicleY]), numSearchNodes = 5)
    
    winname = "Map"
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(winname, on_mouse, param = winname)
    
    baseMap = racetrack.img
    dispMap = baseMap.copy()
    
    while True:
        
        cv2.imshow(winname, dispMap)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    