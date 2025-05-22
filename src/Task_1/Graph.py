
import cv2
import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import random

from typing import Tuple

class Grapher:
    def __init__(self, path: str, scale: float = 1.0, shift: np.ndarray = np.array([0, 0]), eps: float = .5):
        G = nx.read_graphml(path)
        print(f"\nGraph Summary:\n"
              f"- Nodes:    {G.number_of_nodes()}\n"
              f"- Edges:    {G.number_of_edges()}\n"
              f"- Directed: {G.is_directed()}\n")

        # ==================== STRUCTURED NODES ===================
        ys = [float(d.get('y',0.0)) * scale for _, d in G.nodes(data=True)]
        yMin, yMax = min(ys), max(ys)
        self.nodeMap = {}; self.nodeMapOrig = {}
        for node_id, data in G.nodes(data=True):
            nid = int(node_id)
            x = float(data.get('x', 0.0)) * scale - shift[0]
            y = float(data.get('y', 0.0)) * scale
            yOrig = y
            y = (yMax - (y - yMin)) - shift[1]
            self.nodeMap[nid] = (x, y)
            self.nodeMapOrig[nid] = (x, yOrig)

        self.yMin = yMin; self.yMax = yMax

        self.nodeIds = sorted(self.nodeMap.keys())

        # ==================== STRUCTURED EDGE =====================
        edge_list = []
        for u, v, data in G.edges(data=True):
            edge_list.append((int(u), int(v), bool(data['dotted'])))
        self.edgeRecords = np.array(
            edge_list,
            dtype=[('u','i4'), ('v','i4'), ('dotted','?')]
        )
        
        self.groupDup = self.duplicateGroupFind(eps)
        self._dupMap = {}
        for cluster in self.groupDup:
            for nid in cluster:
                self._dupMap[nid] = cluster

    def __getitem__(self, key):
        """
        g['node']           → array of all node coords in ID order
        g['node', i]        → (x,y) of node with ID == i
        g['node', start:stop:step']
                            → list of coords for those IDs
        
        Same for 'edge', returning tuples (u, v, dotted_flag).
        """
        if isinstance(key, tuple) and len(key) == 2:
            kind, idx = key
        else:
            kind, idx = key, slice(None)

        def is_int_sequence(x):
            return (
                isinstance(x, (list, tuple)) or
                (isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.integer))
            )

        if kind == 'node':
            if isinstance(idx, slice):
                ids = self.nodeIds[idx]
                return np.array([self.nodeMap[i] for i in ids])
            if is_int_sequence(idx):
                return np.array([self.nodeMap[int(i)] for i in idx])
            if isinstance(idx, (int, np.integer)):
                if idx not in self.nodeMap:
                    raise KeyError(f"No such node ID: {idx}")
                return np.array(self.nodeMap[idx])
            raise KeyError(f"Invalid node lookup: {idx!r}")
        
        if kind == 'nodeOrig':
            if isinstance(idx, slice):
                ids = self.nodeIds[idx]
                return np.array([self.nodeMapOrig[i] for i in ids])
            if is_int_sequence(idx):
                return np.array([self.nodeMapOrig[int(i)] for i in idx])
            if isinstance(idx, (int, np.integer)):
                if idx not in self.nodeMapOrig:
                    raise KeyError(f"No such node ID: {idx}")
                return np.array(self.nodeMapOrig[idx])
            raise KeyError(f"Invalid node lookup: {idx!r}")


        elif kind == 'edge':
            if isinstance(idx, slice):
                return self.edgeRecords[idx]
            if is_int_sequence(idx):
                rows = []
                for i in idx:
                    i = int(i)
                    if i in self.nodeIds:
                        mask = (self.edgeRecords['u'] == i)
                        subset = self.edgeRecords[mask]
                        if subset.size:
                            u_c = subset['u']
                            v_c = subset['v']
                            d_c = subset['dotted'].astype(np.int32)
                            rows.append(np.column_stack([u_c, v_c, d_c]))
                    elif 1 <= i <= len(self.edgeRecords):
                        rec = self.edgeRecords[i-1]
                        rows.append(np.array([[rec['u'], rec['v'], int(rec['dotted'])]]))
                    else:
                        raise KeyError(f"No such edge or node ID: {i}")
                if rows:
                    return np.vstack(rows)
                else:
                    return np.empty((0,3), dtype=int)
            if isinstance(idx, (int, np.integer)):
                if idx in self.nodeIds:
                    mask = (self.edgeRecords['u'] == idx)
                    subset = self.edgeRecords[mask]
                    u_c = subset['u']
                    v_c = subset['v']
                    d_c = subset['dotted'].astype(np.int32)
                    return np.column_stack([u_c, v_c, d_c])

                if 1 <= idx <= len(self.edgeRecords):
                    rec = self.edgeRecords[idx-1]
                    return np.array([[rec['u'], rec['v'], int(rec['dotted'])]])

            raise KeyError(f"Invalid edge lookup: {idx!r}")

        else:
            raise KeyError(f"Unknown kind {kind!r}; use 'node' or 'edge' or 'nodeOrig")

    @property
    def max(self):
        arr = np.array(list(self.nodeMap.values()))
        return arr.max(axis=0)

    @property
    def min(self):
        arr = np.array(list(self.nodeMap.values()))
        return arr.min(axis=0)
    
    def nearestNodes(self, point: np.ndarray, numNodes: int = 2 ):
        coords = self['node']                        # (N×2) array
        tree = cKDTree(coords)                        # build KD-tree
        allDists, allIndices = tree.query(point, k=len(coords))
        
        allDists   = np.atleast_1d(allDists)
        allIndices = np.atleast_1d(allIndices)

        seenClusters = set()
        nodeIdList   = []
        distList     = []
        clusterList  = []

        for dist, idx in zip(allDists, allIndices):
            nodeId  = self.nodeIds[idx]
            cluster = tuple(sorted(self._dupMap.get(nodeId, [nodeId])))

            if cluster in seenClusters:
                continue

            seenClusters.add(cluster)
            nodeIdList.append(nodeId)
            distList.append(dist)
            clusterList.extend(cluster)

            if len(nodeIdList) >= numNodes:
                break

        return clusterList
    
    def currentEdge(self, point, numSearchNodes: int = 2):
        clusterList = self.nearestNodes(point, numSearchNodes)
        candidateEdge = self['edge', clusterList]
        mask = np.isin(candidateEdge[:, 1], clusterList)
        candidateEdge = candidateEdge[mask]

        dist2Candidate = []
        for edge in candidateEdge:
            dist = self.pointLineDistanceProj(point, *self['node', edge[:2]])
            edge = self.edgeOppositeTest(point, *self['node', edge[:2]])
            dist2Candidate.append(dist if edge == True else np.inf)

        
        return candidateEdge[np.argmin(dist2Candidate)]    
        
    @staticmethod
    def edgeOppositeTest(pt, A, B):
        P = np.asarray(pt, dtype=float)
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        AB = B - A
        denom = np.dot(AB, AB)
        if denom == 0:
            return False   # edge of length zero
        
        t = np.dot(P - A, AB) / denom
        return (0.0 < t) and (t < 1.0)

    @staticmethod
    def pointLineDistanceProj(pt, A, B):
        P = np.array(pt, dtype=float)
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        AB = B - A
        t  = np.dot(P - A, AB) / np.dot(AB, AB)
        closest = A + t * AB
        return np.linalg.norm(P - closest) 
    
    def duplicateGroupFind(self, epsilon: float):
        coords = self['node']
        ids    = self.nodeIds               
        N      = len(coords)

        tree  = cKDTree(coords)
        pairs = tree.query_pairs(r=epsilon)

        parent = list(range(N))
        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i,j in pairs:
            union(i, j)

        clusters = {}
        for idx in range(N):
            root = find(idx)
            clusters.setdefault(root, []).append(ids[idx])

        return [grp for grp in clusters.values() if len(grp) > 1]
    
    def savePath(self, path: list[int], fname: str):
        np.savetxt(fname, path, fmt="%d", delimiter=",")

    def loadPath(self, fname: str) -> list[int]:
        return np.loadtxt(fname, dtype=int, delimiter=",").tolist()
    
   
class MapVisualizer:
    def __init__(self, img: np.ndarray):
        self.original = img.copy()
        self.img = img.copy()
        self.h, self.w = img.shape[:2]
        self.ratio = 1

    def drawNodes(self, graph: Grapher, radius=20, color=(0,255,0)):
        for nid in graph.nodeIds:
            x,y = graph['node', nid].astype(np.int32)
            cv2.circle(self.img, (x,y), radius, color, cv2.FILLED)
        return self

    def resize(self, ratio: float, interpolation=cv2.INTER_AREA):
        new_w = int(self.w // ratio)
        new_h = int(self.h // ratio)
        self.img = cv2.resize(self.img, (new_w, new_h), interpolation=interpolation)
        self.ratio = ratio
        return self

    def drawLabels(self, graph: Grapher,
                   font=cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                   scale=0.35, color=(0,0,255),
                   thickness=1, line_type=cv2.LINE_AA):
        for nid in graph.nodeIds:
            x,y = (graph['node', nid]).astype(np.int32)
            cv2.putText(self.img, str(nid), (x,y),
                        font, scale, color,
                        thickness=thickness, lineType=line_type)
        return self

    def drawSingleEdge(self, graph: Grapher, edge: np.ndarray, 
                       sourceImg: cv2.Mat = None,
                       color=((0,255,100),(0,100,255)),
                       thickness=2,
                       head_length_px=20,
                       head_width_px=10,
                       dash_len=30,
                       gap_len=15):

        isDot = edge[2]
        p1 = np.array((graph['node', edge[0]]).astype(int))
        p2 = np.array((graph['node', edge[1]]).astype(int))
        vec = p2 - p1
        L = np.linalg.norm(vec)
        unit = vec / L

        shaft_end = p2 - unit * head_length_px

        if not isDot:
            cv2.line(
                self.img if sourceImg is None else sourceImg,
                tuple(p1.astype(int)),
                tuple(shaft_end.astype(int)),
                color=color[0],
                thickness=thickness,
                lineType=cv2.LINE_AA
            )
        else:
            shaft_L = np.linalg.norm(shaft_end - p1)
            pos = 0.0
            on = dash_len
            off = gap_len
            while pos < shaft_L:
                start = p1 + unit * pos
                end   = p1 + unit * min(pos + on, shaft_L)
                cv2.line(
                    self.img if sourceImg is None else sourceImg,
                    tuple(start.astype(int)),
                    tuple(end.astype(int)),
                    color=color[1],
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
                pos += on + off

        perp = np.array([-unit[1], unit[0]])
        tip   = p2
        left  = shaft_end + perp * (head_width_px/2)
        right = shaft_end - perp * (head_width_px/2)
        pts   = np.array([
            tip.astype(int),
            left.astype(int),
            right.astype(int)
        ])
        cv2.fillConvexPoly(self.img if sourceImg is None else sourceImg, pts, color[0] if not isDot else color[1])
    
        return self.img if sourceImg is None else sourceImg

    def drawEdges(self, graph: Grapher,
                  color=((0,255,100),(0,100,255)),
                  thickness=2,
                  head_length_px=20,
                  head_width_px=10,
                  dash_len=30,
                  gap_len=15):
        for u, v, isDot in graph['edge']:
            p1 = np.array((graph['node', u]).astype(int))
            p2 = np.array((graph['node', v]).astype(int))
            vec = p2 - p1
            L = np.linalg.norm(vec)
            unit = vec / L

            shaft_end = p2 - unit * head_length_px

            if not isDot:
                cv2.line(
                    self.img,
                    tuple(p1.astype(int)),
                    tuple(shaft_end.astype(int)),
                    color=color[0],
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )
            else:
                shaft_L = np.linalg.norm(shaft_end - p1)
                pos = 0.0
                on = dash_len
                off = gap_len
                while pos < shaft_L:
                    start = p1 + unit * pos
                    end   = p1 + unit * min(pos + on, shaft_L)
                    cv2.line(
                        self.img,
                        tuple(start.astype(int)),
                        tuple(end.astype(int)),
                        color=color[1],
                        thickness=thickness,
                        lineType=cv2.LINE_AA
                    )
                    pos += on + off

            perp = np.array([-unit[1], unit[0]])
            tip   = p2
            left  = shaft_end + perp * (head_width_px/2)
            right = shaft_end - perp * (head_width_px/2)
            pts   = np.array([
                tip.astype(int),
                left.astype(int),
                right.astype(int)
            ])
            cv2.fillConvexPoly(self.img, pts, color[0] if not isDot else color[1])

        return self

    def show(self, window_name="Map View", wait_key=ord('q')):
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        while True:
            cv2.imshow(window_name, self.img)
            if (cv2.waitKey(1) & 0xFF) == wait_key:
                break
        cv2.destroyAllWindows()

    def reset(self):
        self.img = self.original.copy()
        self.ratio = 1
        return self
    
    def drawPath(self, graph: Grapher, path: list[int],
                 color=((0,255,100),(0,100,255)),
                 thickness=2,
                 head_length_px=20,
                 head_width_px=10,
                 dash_len=30,
                 gap_len=15):
        for u, v in zip(path[:-1], path[1:]):
            mask = (graph.edgeRecords['u'] == u) & (graph.edgeRecords['v'] == v)
            if np.any(mask):
                dotted = bool(graph.edgeRecords['dotted'][mask][0])
            else:
                dotted = False

            edge = np.array([u, v, dotted])
            # draw one arrow segment
            self.drawSingleEdge(
                graph, edge, sourceImg=None,
                color=color,
                thickness=thickness,
                head_length_px=head_length_px,
                head_width_px=head_width_px,
                dash_len=dash_len,
                gap_len=gap_len
            )
        return self
    
    def drawClothoidPath(self,
                        definedPath: np.ndarray,   # shape (3, N): x, y, heading
                        color: Tuple[int,int,int] = (0,0,255),
                        thickness: int = 2
                       ) -> "MapVisualizer":
        """
        Draws the continuous path (only x,y) on self.img.
        Assumes definedPath is [x_vals, y_vals, heading_vals].
        """
        # grab only x,y, transpose to (N,2) and int32
        pts = definedPath[:2].T.astype(np.int32)
        # draw as one open polyline
        cv2.polylines(self.img,
                      [pts],
                      isClosed=False,
                      color=color,
                      thickness=thickness,
                      lineType=cv2.LINE_AA)
        return self

        

def randomPath(graph, maxLen=None):
    """
    graph: an instance of your Grapher class
    maxLen: optional hard‐cap on how many steps to take
    """
    path = []
    visited = set()

    current = random.choice(graph.nodeIds)
    path.append(current)
    visited.add(current)

    while True:
        outs = graph['edge', current]  # Nx3 array of (u,v,dotted)
        nextNodes = [int(v) for u,v,_ in outs]

        if not nextNodes:
            break

        nxt = random.choice(nextNodes)

        if nxt in visited:
            path.append(nxt)
            break

        path.append(nxt)
        visited.add(nxt)
        current = nxt

        if maxLen is not None and len(path) >= maxLen:
            break

    return path