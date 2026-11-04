import cadquery as cq
from cadquery.vis import show

height = 70.0
width = 90.0
thickness = 10.0

# make the base
result = cq.Workplane("XY").box(height, width, thickness)

# Render the solid
show(result)