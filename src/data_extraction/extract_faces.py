import os
import glob
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.GeomAbs import (
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader

cad_paths = glob.glob(os.path.join("..", "..", "data", "CADs", "bar.STEP"))

if not cad_paths:
    raise FileNotFoundError("Could not find 'bar.STEP' in '../../data/CADs/'")

cad_locs=cad_paths[0]

step_reader = STEPControl_Reader()
status = step_reader.ReadFile(cad_locs)

if status != IFSelect_RetDone:
    raise ValueError(f"Failed to read STEP file at '{cad_locs}'")

step_reader.TransferRoot()
cad_shape = step_reader.Shape()

def get_surface_type_name(surf_type):
    """Map Open CASCADE surface type enum to human-readable string."""
    types = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
    }
    return types.get(surf_type, "BSpline/NURBS/Other")


def extract_faces(shape):
    topo_exp = TopologyExplorer(shape)
    faces_data = []

    for idx, face in enumerate(topo_exp.faces(), start=1):
        # Extract underlying surface geometry
        surf_adaptor = BRepAdaptor_Surface(face)
        surf_type_enum = surf_adaptor.GetType()
        surf_name = get_surface_type_name(surf_type_enum)

        # Count boundary edges belonging to this face
        boundary_edges = list(topo_exp.edges_from_face(face))

        face_info = {
            "face_id": idx,
            "face_shape": face,
            "surface_type": surf_name,
            "edge_count": len(boundary_edges),
        }
        faces_data.append(face_info)

    return faces_data


if __name__ == "__main__":
    # Create sample 3D Box shape

    extracted_faces = extract_faces(cad_shape)
    print(f"Total Extracted Faces: {len(extracted_faces)}\n")

    for f in extracted_faces:
        print(
            f"Face ID: {f['face_id']} | Type: {f['surface_type']} | Boundary Edges: {f['edge_count']}"
        )