from pathlib import Path
import argparse #parse command-line flags like --input and --output passed when running the script
import json
import math
import numpy as np
import cadquery as cq

SURFACE_TYPES = ["PLANE", 
                 "CYLINDER", 
                 "CONE", 
                 "SPHERE", 
                 "TORUS", 
                 "BSPLINE", 
                 "OTHER"]
CURVE_TYPES = ["LINE", 
               "CIRCLE",
                 "ELLIPSE", 
                 "BSPLINE", 
                 "OTHER"]

# finds or return the surface type in vector format e.g[1,0,0,0,0,0,0] for plane
def one_hot(value, choices):
    result = [0.0] * len(choices)
    try:
        result[choices.index(value)] = 1.0
    except ValueError:
        result[-1] = 1.0
    return result

#Convert incoming values into floating value 
def safe_float(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default

############################################
def vec3(v): #it takes CAdQuery 3D vector object and return floating point numbers [x,y,z]
    return [safe_float(v.x), safe_float(v.y), safe_float(v.z)]

def face_features(face):
    surface = str(face.geomType()).upper()
    if "BSPLINE" in surface:
        surface = "BSPLINE"
    elif surface not in SURFACE_TYPES:
        surface = "OTHER"

    area = safe_float(face.Area())
    center = vec3(face.Center())
    try:
        normal = vec3(face.normalAt())
    except Exception:
        normal = [0.0, 0.0, 0.0]

    bb = face.BoundingBox()
    dims = [safe_float(bb.xlen), safe_float(bb.ylen), safe_float(bb.zlen)]

    # 7 surface-type indicators + area + centroid(3) + normal(3) + bbox dimensions(3)
    return one_hot(surface, SURFACE_TYPES) + [area] + center + normal + dims

def edge_features(edge):
    curve = str(edge.geomType()).upper()
    if "BSPLINE" in curve:
        curve = "BSPLINE"
    elif curve not in CURVE_TYPES:
        curve = "OTHER"

    length = safe_float(edge.Length())
    radius = 0.0
    if curve == "CIRCLE":
        try:
            radius = safe_float(edge.radius())
        except Exception:
            radius = 0.0

    center = vec3(edge.Center())
    try:
        tangent = vec3(edge.tangentAt(0.5))
    except Exception:
        tangent = [0.0, 0.0, 0.0]

    # 5 curve-type indicators + length + radius + center(3) + tangent(3)
    return one_hot(curve, CURVE_TYPES) + [length, radius] + center + tangent

def same_shape(a, b): #checks iff faces share same edge
    # IsSame compares the underlying OpenCASCADE topological shape.
    try:
        return bool(a.wrapped.IsSame(b.wrapped))
    except Exception:
        return False

def convert_step(step_path, output_dir):
    imported = cq.importers.importStep(str(step_path))
    solids = imported.solids().vals()
    if not solids:
        raise RuntimeError(f"No solid found in {step_path}")

    # Load and exact topology or loading CAD
    solid = solids[0]
    faces = solid.Faces()
    edges = solid.Edges()

    node_array = np.asarray([face_features(f) for f in faces], dtype=np.float32) #exact node features
      
    edge_array = np.asarray([edge_features(e) for e in edges], dtype=np.float32) #exact edge features

    # extract edge features and adjency,find which faces are connected
    face_pairs = []
    graph_edge_attrs = []
    for edge_idx, edge in enumerate(edges):
        incident_faces = []
        #faces that share same edges
        for face_idx, face in enumerate(faces):
            if any(same_shape(edge, face_edge) for face_edge in face.Edges()):
                incident_faces.append(face_idx)

        #create connection between faces
        if len(incident_faces) >= 2:

            for i in range(len(incident_faces)):
                for j in range(i + 1, len(incident_faces)):
                    a, b = incident_faces[i], incident_faces[j]
                    face_pairs.append((a, b))
                    graph_edge_attrs.append(edge_array[edge_idx].tolist())

    # Store directed connections in both directions for common GNN conventions.
    directed_pairs = []
    directed_attrs = []
    for (a, b), attr in zip(face_pairs, graph_edge_attrs):
        directed_pairs.extend([(a, b), (b, a)])
        directed_attrs.extend([attr, attr])

    if directed_pairs:
        edge_index = np.asarray(directed_pairs, dtype=np.int64).T
        graph_edge_features = np.asarray(directed_attrs, dtype=np.float32)
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
        graph_edge_features = np.empty((0, len(CURVE_TYPES) + 8), dtype=np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = step_path.stem
    npz_path = output_dir / f"{stem}.npz"
    json_path = output_dir / f"{stem}.json"

    np.savez_compressed(
        npz_path,
        face_features=node_array,
        edge_index=edge_index,
        edge_features=graph_edge_features,
        brep_edge_features=edge_array,
    )

    metadata = {
        "source_step": step_path.name,
        "solid_count_in_import": len(solids),
        "processed_solid_index": 0,
        "face_count": len(faces),
        "brep_edge_count": len(edges),
        "directed_graph_connection_count": int(edge_index.shape[1]),
        "face_feature_dim": int(node_array.shape[1]),
        "graph_edge_feature_dim": int(graph_edge_features.shape[1]),
        "face_feature_layout": (
            SURFACE_TYPES + ["area"] +
            ["centroid_x", "centroid_y", "centroid_z"] +
            ["normal_x", "normal_y", "normal_z"] +
            ["bbox_x", "bbox_y", "bbox_z"]
        ),
        "graph_edge_feature_layout": (
            CURVE_TYPES + ["length", "radius"] +
            ["center_x", "center_y", "center_z"] +
            ["tangent_x", "tangent_y", "tangent_z"]
        ),
        "notes": [
            "Face nodes and shared-edge face adjacency form a simple graph baseline.",
            "Graph connections are directed in both directions.",
            "This starter extractor does not calculate edge convexity or principal curvature.",
            "STEP imports containing multiple solids are currently processed one solid at a time (first solid only).",
            "Continuous features should be normalized using statistics from the training split only."
        ]
    }
    json_path.write_text(json.dumps(metadata, indent=2))
    print(
        f"{step_path.name}: faces={len(faces)}, B-Rep edges={len(edges)}, "
        f"graph connections={edge_index.shape[1]}, "
        f"face_features={node_array.shape}, edge_index={edge_index.shape}, "
        f"edge_features={graph_edge_features.shape}"
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="models", help="Folder containing STEP files")
    parser.add_argument("--output", default="graphs", help="Output folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    step_files = sorted(list(input_dir.glob("*.step")) + list(input_dir.glob("*.stp")))
    if not step_files:
        raise SystemExit(f"No .step/.stp files found in {input_dir}. Run generate_models.py first.")

    for step_path in step_files:
        try:
            convert_step(step_path, output_dir)
        except Exception as exc:
            print(f"FAILED {step_path.name}: {exc}")

if __name__ == "__main__":
    main()
