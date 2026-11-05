import glob
import os

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Extend.TopologyUtils import TopologyExplorer

# load STEP file
cad_paths = glob.glob(os.path.join("..", "..", "data", "CADs", "bar.STEP"))
if not cad_paths:
    raise FileNotFoundError("Could not find 'bar.STEP' in '../../data/CADs/'")

cad_loc = cad_paths[0]
step_reader = STEPControl_Reader()
if step_reader.ReadFile(cad_loc) != IFSelect_RetDone:
    raise ValueError(f"Failed to read STEP file at '{cad_loc}'")

step_reader.TransferRoot()
cad_shape = step_reader.Shape()

#Initialize Topology Explorer
topo_exp = TopologyExplorer(cad_shape)

# Create unique ID maps for faces and edges
faces = list(topo_exp.faces())
edges = list(topo_exp.edges())
face_to_id = {face: idx + 1 for idx, face in enumerate(faces)}
edge_to_id = {edge: idx + 1 for idx, edge in enumerate(edges)}

# BUILD ADJACENCY INFORMATION


# Face-to-Face Adjacency (Neighboring faces sharing an edge)
face_adjacency = {f_id: set() for f_id in face_to_id.values()}

# Edge-to-Face Adjacency (Which faces meet at each edge)
edge_to_faces = {}

for edge in edges:
    e_id = edge_to_id[edge]
    sharing_faces = list(topo_exp.faces_from_edge(edge))
    sharing_face_ids = [face_to_id[f] for f in sharing_faces if f in face_to_id]

    edge_to_faces[e_id] = sharing_face_ids

    # Populate Face-to-Face connections across shared edges
    for f1_id in sharing_face_ids:
        for f2_id in sharing_face_ids:
            if f1_id != f2_id:
                face_adjacency[f1_id].add(f2_id)

# DISPLAY RESULTS
print(f"File: {cad_loc}")
print(f"Total Faces: {len(faces)} | Total Edges: {len(edges)}\n")

print("--- Face-to-Face Adjacency (Neighboring Faces) ---")
for f_id, neighbors in sorted(face_adjacency.items()):
    print(f"Face {f_id} is adjacent to Face(s): {sorted(list(neighbors))}")

print("\n--- Edge-to-Face Adjacency (Faces sharing an Edge) ---")
for e_id, f_ids in list(sorted(edge_to_faces.items()))[:10]:  # Show first 10
    print(f"Edge {e_id} connects Face(s): {f_ids}")