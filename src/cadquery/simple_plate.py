import cadquery as cq
from cadquery.vis import show
import torch


height = 70.0
width = 90.0
thickness = 10.0

# make the base
import cadquery as cq
from cadquery.vis import show

height = 60.0
width = 80.0
thickness = 10.0
diameter = 22.0
padding = 12.0

# make the base
cad_result = (
    cq.Workplane("XY")
    .box(height, width, thickness)
    .faces(">Z")
    .workplane()
    .hole(diameter) #hole on center
    .faces(">Z")
    .workplane()
    .rect(height - padding, width - padding, forConstruction=True)
    .vertices()
    .cboreHole(2.4, 4.4, 2.1) #hole in corner
    .edges("|Z")
    .fillet(10.0)
)

# Render the solid
show(cad_result)
# result=cq.immporters.importStep("/data/CADs/simple_plate.STEP", unit="mM")  importing .step

cad_result.export("/data/CADs/somple_plate.STEP")

#install  conda install -c conda-forge cq-editor -y