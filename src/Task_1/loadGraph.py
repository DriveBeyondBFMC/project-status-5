import cv2
import numpy as np
from PathGenerator import PathHandler
from Graph import *
import socket

def openFile(path: str):
    nodes = []
    with open(path, "r") as f:
        rawNodes = f.readlines()
        for rawnode in rawNodes:
            nodes += [int(rawnode.strip("\n"))]

    return nodes
        
if __name__ == "__main__":
    
    coorSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    coorSocket.bind(("0.0.0.0", 1234))

    graph = Grapher("./track/Competition_track_graph.graphml", scale = 72.5, shift = [0, 0], eps = 2)
    path = openFile("./track/nodes.txt")
    checkpoints = openFile("./track/checkpoints.txt")
    points = graph['nodeOrig', path]
    handler = PathHandler(points)
   
    if handler is not None:
        dispPath = handler.definedPath.copy()
        yMin, yMax = graph.min[1], graph.max[1]
        dispPath[1] = yMax - (dispPath[1] - yMin)

    

    img = cv2.imread("./track/Track.png")
    racetrack = (
        MapVisualizer(img)
            .resize(ratio = 9766 / 1500)
            .drawNodes(graph, radius = 2)
            .drawCheckpoints(graph, checkpoints, radius = 5, color = (0, 255, 255))
            # .drawEdges(graph, thickness = 1, head_width_px = 7, head_length_px = 7)
            # .drawLabels(graph, scale = .4, line_type = cv2.LINE_AA, thickness = 1, color = (25, 163, 233), font = cv2.FONT_HERSHEY_TRIPLEX)
            .drawPath(graph, path, thickness = 1, head_width_px = 7, head_length_px = 7, color = ((0, 255, 100), (0, 255, 100)))
            .drawClothoidPath(dispPath, thickness = 1)
    )

    vehicleX = 0; vehicleY = 0
    edge = graph.currentEdge(np.array([vehicleX, vehicleY]), numSearchNodes = 5)
    
    winname = "Map"
    cv2.namedWindow(winname, cv2.WINDOW_AUTOSIZE)
    
    baseMap = racetrack.img
    dispMap = baseMap.copy()
    print(dispMap.shape)
    
    while True:
        coordinate = [0, 0]
        data, addr = coorSocket.recvfrom(1024)    
        message = data.decode("utf-8")
        splitCmd = message.split(" | ")
        for idx, unprocessed in enumerate(splitCmd):
            coordinate[idx] = int(float(unprocessed.split(": ")[1]) * 100)
            
        yMin, yMax = graph.min[1], graph.max[1]
        coordinate[1] = int(yMax - (coordinate[1] - yMin))
        
        
        dispMap = baseMap.copy()
        vehicleX = coordinate[0]; vehicleY = coordinate[1]

        edge = graph.currentEdge(
            np.array([vehicleX, vehicleY]),
            numSearchNodes=5
        )
        dispMap = racetrack.drawSingleEdge(graph, edge, dispMap, thickness = 1, head_width_px = 7, head_length_px = 7, color = ((255, 0, 0), (255, 100, 0)))
        cv2.circle(dispMap, (vehicleX, vehicleY), 3, (255, 100), cv2.FILLED, cv2.LINE_AA)
        
        
        
        cv2.imshow(winname, dispMap)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    