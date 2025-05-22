import re
import numpy as np
import pyclothoids
from scipy.interpolate import interp1d

class HeadFit:
    def __init__(self, nodes: np.ndarray, thetaEps: float = np.pi / 2):
        self.points = nodes
        self.nodesID = np.arange(0, nodes.shape[1])
        self.thetaEps = thetaEps
        
        self.tripletSegments = np.stack([
            self.nodesID[:-2],
            self.nodesID[1:-1],
            self.nodesID[2:]
        ], axis=1)

    def makePseudoPoints(self, pt):
        """This math works if AB = BC (glorified assumption)"""
        
        # FIND AN ASSUME CENTER
        A, B, C = pt
        v = A - C
        perp = np.array([-v[1], v[0]])
        u = perp / np.linalg.norm(perp)
        
        E = B - A
        F = B - C

        a = np.dot(perp, perp)              
        b = np.dot(E + F, perp)             
        c = np.dot(E, F)

        disc = b*b - 4*a*c
        if disc < 0:
            raise ValueError("No real solution: the line misses the circle.")

        sqrt_disc = np.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2*a)
        t2 = (-b - sqrt_disc) / (2*a)

        P1 = B + t1 * perp
        P2 = B + t2 * perp
        
        if self.edgeOppositeTest(A, B, P1) == True:
            P = P1
        else: 
            P = P2
            
        # FIND AN PSEUDO POINT
        u = (C - P) / np.linalg.norm(C - P)
        R = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                                    [np.sin(theta), np.cos(theta)]])
        
        pseudo1 = P + np.linalg.norm(P - C) * R(-np.pi / 4) @ u
        pseudo2 = P + np.linalg.norm(P - C) * R(np.pi / 4) @ u
        
        
        if self.edgeOppositeTest(pseudo1, A, C) == True: return pseudo1
        else: return pseudo2
        
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

    def angleValidation(self, pt): 
        """Validate if the angle formed from start, end and middle forms an obtuse angle
        If there are only 3 points and mid point is not on the circle then create a pseudo point
        """
        start, end = pt[0], pt[-1]
        validPoints = []
        change = False
        for idx in range(1, len(pt) - 1):
            # cos(beta) = (AB^2 + BC^2 - AC^2) / (2 * AB * BC)
            AC = np.linalg.norm(start - end)
            AB = np.linalg.norm(start - pt[idx])
            BC = np.linalg.norm(pt[idx] - end)
            
            cosBeta = (AB**2 + BC ** 2 - AC ** 2) / (2 * AB * BC)
            cosBeta = np.clip(cosBeta, -1.0, 1.0)
            beta = np.arccos(cosBeta)
            if beta >= self.thetaEps:
                validPoints.append(pt[idx])
            else:
                change = True
        
        if len(validPoints) == 0:
            validPoints.append(self.makePseudoPoints(pt))
        
            
        validPoints.insert(0, start)
        validPoints.append(end)
        
        return np.array(validPoints), change

    def valCircleParam(self):
        
        centers = np.empty((0, 2)); radiuses = np.empty((0))
        for triplet in self.tripletSegments:
            A = np.c_[self.points[:, triplet].T, np.ones(self.points[:, triplet].T.shape[0])[..., None]]
            b = -np.sum(self.points[:, triplet].T ** 2, axis = 1)[..., None]
            (D, E, F) = np.linalg.lstsq(A, b, rcond = None)[0]
            h, k = -D / 2, -E / 2
            r = (h ** 2 + k ** 2 - F) ** .5
            
            radiuses = np.r_[radiuses, r[0]]
            centers = np.r_[centers, np.array([[h[0], k[0]]])]
        return radiuses, centers 

    def circleHeadings(self, centers):
        """
        For each curved run in curveSet[j] (list of indices in path order)
        returns an array of heading angles (radians) at every index.
        """
        all_headings = []
        for j,run in enumerate(self.tripletSegments):
            
            run_pts = self.points[:, run].T        # shape (M,2)
            c       = centers[j]         # (h,k) for this circle

            radials = run_pts - c[None,:]              # (M,2)
            t1 = np.stack([-radials[:,1], radials[:,0]], axis=1)
            t2 = np.stack([ radials[:,1],-radials[:,0]], axis=1)
            t1 /= np.linalg.norm(t1,axis=1,keepdims=True)
            t2 /= np.linalg.norm(t2,axis=1,keepdims=True)

            headings = np.zeros(len(run))
            for i in range(len(run_pts)):
                if i < len(run_pts)-1:
                    seg = run_pts[i+1] - run_pts[i]
                else:
                    seg = run_pts[i] - run_pts[i-1]
                seg = seg/np.linalg.norm(seg)
                if np.dot(seg, t1[i]) > np.dot(seg, t2[i]):
                    tang = t1[i]
                else:
                    tang = t2[i]
                
                headings[i] = np.arctan2(tang[1], tang[0])

            all_headings.append(headings)
        return np.array(all_headings)

    def checkIntegrity(self):
        nodeChanges = []
        pointsPermute = []
        for triplet in self.tripletSegments:
            validPoints, change = self.angleValidation(self.points[:, triplet].T)
            nodeChanges += [change]
            pointsPermute += [validPoints]
            
        nodeChanges = np.array(nodeChanges, dtype = bool)
        pointsPermute = np.array(pointsPermute)

        
        self.points[:, self.tripletSegments[nodeChanges]] = pointsPermute[nodeChanges].transpose(2, 0, 1)

class PathHandler(HeadFit):
    def __init__(self, nodes: np.ndarray, headings: np.ndarray = None, cameraSwitch: np.ndarray = None, curveEps: float = np.pi / 2):
        super(PathHandler, self).__init__(nodes.T, thetaEps = curveEps)

        if isinstance(headings, type(None)):
            self.checkIntegrity()
            radiuses, centers = self.valCircleParam()
            rawHeadings = self.circleHeadings(centers)
            
            cosSum = np.zeros(len(self.nodesID))
            sinSum = np.zeros(len(self.nodesID))
            headingCount = np.zeros(len(self.nodesID))

            for triplet, heading in zip(self.tripletSegments, rawHeadings):
                for i, nodeId in enumerate(triplet):
                    cosSum[nodeId] += np.cos(heading[i])
                    sinSum[nodeId] += np.sin(heading[i])
                    headingCount[nodeId] += 1

            self.headings = np.arctan2(sinSum, cosSum)
        else: self.headings = headings
        
        self.definedPath = self.__generator__(self.points[0], self.points[1], self.headings)
        self.cameraSwitch = cameraSwitch
        self.nodes = self.points.T    
        self.pathInterpolation

        self.definedPathCopy = self.definedPath.copy()
    
    @classmethod
    def clothoidsGenFile(cls, pathName: str, curveEps: float = np.pi / 2) -> "PathHandler":
        dataArr = {}
        with open(pathName, "r") as file:
            for line in file:
                key, values = line.split(":")
                # Remove brackets and commas, then split into a list
                cleaned_values = re.sub(r"[\[\],]", "", values).strip()
                values_list = np.array([float(v) for v in cleaned_values.split()])
                dataArr[key.strip()] = values_list

        # Accessing the arrays
        yPath = dataArr["Array_y"] 
        xPath = dataArr["Array_x"] 
        try:
            theta = np.pi * 2 - (dataArr["Array_z"] + np.pi / 2)
        except:
            theta = None
        cameraSwitch = dataArr["Array_t"]
        
        mask = np.insert((np.diff(xPath) != 0) | (np.diff(yPath) != 0), 0, True)

        xPath = xPath[mask]
        yPath = yPath[mask]
        cameraSwitch = cameraSwitch[mask]
        points = np.array([xPath, yPath])
        return cls(points, theta, cameraSwitch, curveEps)
    
    @staticmethod
    def __generator__(x, y, theta):
        xVals, yVals, headings = [], [], []
        ds = 1e-4

        for i in range(min(len(theta), len(x), len(y)) - 1):
            clothoid = pyclothoids.Clothoid.G1Hermite(x[i], y[i], theta[i], x[i+1], y[i+1], theta[i+1])
            pointDist = np.linalg.norm([x[i+1] - x[i], y[i+1] - y[i]])

            sVals = np.linspace(0, clothoid.length - ds, int(pointDist))
            
            for s in sVals:
                px = clothoid.X(s)
                py = clothoid.Y(s)
                dx = clothoid.X(s + ds) - px
                dy = clothoid.Y(s + ds) - py
                angle = np.arctan2(dy, dx)

                xVals.append(px)
                yVals.append(py)
                headings.append(angle)
        return np.array([xVals, yVals, headings])

    def generateOffsetPath(self, offsetDist: float):
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                        [np.sin(base[2]), np.cos(base[2]), base[1]],
                                        [0, 0, 1]])
        offsetPoints = []; self.offsetDist = offsetDist
        for nodes in self.nodes.T:
            
            offsetMajorPoint = (rotmat(nodes) @ np.array([[0, offsetDist, 1]]).T).T[0]
            offsetMajorPoint[2] = nodes[2]
            offsetPoints += [offsetMajorPoint]
        offsetPoints = np.array(offsetPoints).T
        
        self.offsetPath = self.__generator__(*offsetPoints)
    
    @property
    def pathInterpolation(self):
        pathDiff = np.diff(self.definedPath, axis = 1)
        ds = np.sqrt(np.sum(pathDiff ** 2, axis = 0))
        s = np.concat(([0], np.cumsum(ds)))

        xInterp = interp1d(s, self.definedPath[0], kind = 'linear')
        yInterp = interp1d(s, self.definedPath[1], kind = 'linear')
        psiInterp = interp1d(s, self.definedPath[2], kind = 'linear')
        
        self.pathInterp = (xInterp, yInterp, psiInterp)
        self.s = s

    def generateReference(self, vRef: float, dt: float, startIdx: int, N: int) -> np.ndarray:
        refTraj = []
        for i in range(N):
            s_i = min(self.s[startIdx] + vRef * i * dt, self.s[-1])
            refTraj.append([
                self.pathInterp[0](s_i),
                self.pathInterp[1](s_i),
                self.pathInterp[2](s_i),
            ])
            
        return np.array(refTraj)
    
    def pathAvoidance(self, positionIdx: np.ndarray, offsetOnPath: float, offsetDist: np.ndarray, baseLength: float):
        startIdx = positionIdx
        
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                         [np.sin(base[2]), np.cos(base[2]), base[1]],
                                         [0, 0, 1]])

        # ========  Define 4 points for avoidance path =======
        start = np.array([self.pathInterp[i](self.s[startIdx]) for i in range(3)])
        
        offset1 = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath) for i in range(3)])

        offset2Base = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist[0]) for i in range(3)])
        offset2 = (rotmat(offset2Base) @ np.array([[0, offsetDist[1], 1]]).T).T[0]
        
        
        offset3Base = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist[0] + baseLength) for i in range(3)])
        offset3 = (rotmat(offset3Base) @ np.array([[0, offsetDist[1], 1]]).T).T[0]

        end = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist[0] * 2 + baseLength) for i in range(3)])
        # ========  Define 4 points for avoidance path =======

        targetS = self.s[startIdx] + offsetDist[0] * 2 + baseLength + offsetOnPath
        endIdx = np.searchsorted(self.s, targetS)
        
        offset2[2] = offset2Base[2]
        offset3[2] = offset3Base[2]

        offsetPath = self.__generator__(
            *zip(*[start, offset1, offset2, offset3, end])
        )

        self.definedPath = np.concatenate(
            (self.definedPath[:, :startIdx], offsetPath, self.definedPath[:, endIdx + 20:]),
            axis = 1
        ) # add new path

        return offsetPath

    def entryRamp(self, positionIdx: int, 
                  offsetPositionIdx: int, 
                  offsetOnPath: float, 
                  offsetDist: float, 
                  possibleEnd: float = 0, 
                  epsilon: float = 1) -> np.ndarray:
        startIdx = positionIdx
        
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                         [np.sin(base[2]), np.cos(base[2]), base[1]],
                                         [0, 0, 1]])

        start = np.array([self.pathInterp[i](self.s[startIdx]) for i in range(3)])
        
        mid1 = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath) for i in range(3)])

        mid2Offset = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist) for i in range(3)])
        mid2 = (rotmat(mid2Offset) @ np.array([[0, self.offsetDist, 1]]).T).T[0]
        mid2[2] = mid2Offset[2]
        
        endOffset = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist + possibleEnd) for i in range(3)])
        end = (rotmat(endOffset) @ np.array([[0, self.offsetDist, 1]]).T).T[0]
        end[2] = endOffset[2]

        def lastIndex(idxState):
            dist = np.linalg.norm(idxState[:2][..., None] - self.offsetPath[:2], axis = 0)
            minIndices = np.where(dist <= dist.min() + epsilon)[0]
            if len(minIndices) > 1:
                lastIdx = min(minIndices, key=lambda x: abs(x - offsetPositionIdx))
            else:
                lastIdx = minIndices[0]
            
            return lastIdx
        
        lastIdxRamp = lastIndex(mid2)
        lastIdxPossibleLength = lastIndex(end)
        
        
        enterRamp = self.__generator__(*zip(*[start, mid1, mid2]))
        self.offsetPath = np.concatenate([self.offsetPath[:, :offsetPositionIdx], enterRamp, self.offsetPath[:, lastIdxRamp:]], axis = 1)

        return lastIdxPossibleLength    
    
    def exitRamp(self, positionIdx: int, offsetOnPath: float, offsetDist: float) -> np.ndarray:
        startIdx = positionIdx
        
        rotmat = lambda base: np.array([[np.cos(base[2]), -np.sin(base[2]), base[0]],
                                         [np.sin(base[2]), np.cos(base[2]), base[1]],
                                         [0, 0, 1]])
        
        startOffset = np.array([self.pathInterp[i](self.s[startIdx]) for i in range(3)])
        start = (rotmat(startOffset) @ np.array([[0, self.offsetDist, 1]]).T).T[0]
        start[2] = startOffset[2]

        mid = np.array([self.pathInterp[i](self.s[startIdx] + offsetDist) for i in range(3)])
        
        end = np.array([self.pathInterp[i](self.s[startIdx] + offsetOnPath + offsetDist) for i in range(3)])
        
        targetS = self.s[startIdx] + offsetDist + offsetOnPath
        endIdx = np.searchsorted(self.s, targetS)
        
        exitRamp = self.__generator__(*zip(*[start, mid, end]))
        self.definedPath = np.concatenate([self.definedPath[:, :startIdx], exitRamp, self.definedPath[:, endIdx:]], axis = 1)
        self.pathInterpolation
    
        return endIdx
        
    def distance2Path(self, position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(position - self.definedPath[:2], axis = 0)
    
    def distance2MajorPoint(self, position: np.ndarray) -> np.ndarray:
        return np.linalg.norm(position - self.nodes[:2], axis = 0)