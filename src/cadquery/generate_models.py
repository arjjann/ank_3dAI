from pathlib import Path
import cadquery as cq

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path.home() / "ank_3dAI" / "data" / "CADs" 
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def make_models():
    models = {}

    #Rectangular plate with two through-holes
    plate = cq.Workplane("XY").box(60, 40, 8)
    plate = plate.faces(">Z").workplane().pushPoints([(-15, 0), (15, 0)]).hole(8)
    models["plate_two_holes"] = plate

    # Block with a rectangular pocket
    pocket = cq.Workplane("XY").box(50, 40, 15)
    pocket = pocket.faces(">Z").workplane().rect(22, 16).cutBlind(-6)
    models["block_pocket"] = pocket

    #Stepped block made by fusing two rectangular prisms
    lower = cq.Workplane("XY").box(50, 30, 10)
    upper = cq.Workplane("XY").box(25, 30, 12).translate((12.5, 0, 11))
    models["stepped_block"] = lower.union(upper)

    #L-shaped bracket made from two intersecting plates
    base = cq.Workplane("XY").box(50, 12, 8).translate((0, 0, 0))
    upright = cq.Workplane("XY").box(12, 12, 42).translate((-19, 0, 25))
    bracket = base.union(upright)
    bracket = bracket.faces(">Z").workplane().pushPoints([(-15, 0), (15, 0)]).hole(5)
    models["l_bracket"] = bracket

    # Stepped cylindrical shaft
    shaft_a = cq.Workplane("XY").circle(12).extrude(12)
    shaft_b = cq.Workplane("XY").workplane(offset=12).circle(7).extrude(28)
    shaft_c = cq.Workplane("XY").workplane(offset=40).circle(10).extrude(8)
    models["stepped_shaft"] = shaft_a.union(shaft_b).union(shaft_c)

    return models

if __name__ == "__main__":
    for name, model in make_models().items():
        out = MODEL_DIR / f"{name}.step"
        cq.exporters.export(model, str(out), exportType="STEP")
        print(f"Exported {out}")
