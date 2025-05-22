
# Task 1: Map Visualization
## 📂 Project Directory Structure
```bash
~/project-status-5/src/Task_1
├── loadGraph.py
└── Graph.py
```

## 📝 Description
   Focuses on parsing a GraphML track, generating node‐based and clothoid paths, and visualizing them on an overhead map for lane‐changing scenarios.

**Graph Processing Module**  
   - File: `Graph.py`
   - Functionality: Load and normalize a GraphML track; build a KD-tree for nearest-node/edge lookups; draw nodes, directed (solid/dotted) edges, arrowed paths and clothoid splines onto an OpenCV image;

**Graph Loader & Visualization**
   - File: `loadGraph.py`
   - Functionality: nstantiate Grapher on your .graphml; fit a smooth (clothoid) trajectory via PathHandler; overlay nodes, discrete path and clothoid on the track image; show interactively with OpenCV.


# Task 3: Auto heading calculation
## 📂 Project Directory Structure
```bash
~/project-status-5/src/Task_3
└── PathGenerator.py
```
## 📝 Description
   Automatically compute a smooth, reliable heading for each 2D waypoint

**MPC Controller**
   - File: `PathGenerator.py`
   - Functionality: Generate headings of each point for PyClothoids by fitting 3 consecutive points onto an arc.