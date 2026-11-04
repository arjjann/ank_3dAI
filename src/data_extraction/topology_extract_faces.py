import os
import random
import sys

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

#  Expand ~ to your home directory path safely
STEP_FILE_PATH = os.path.expanduser(os.path.join("~", "ank_3dAI", "data", "CADs", "bar.STEP"))

if __name__ == "__main__":
    #  Initialize display using PySide6 backend for Conda
    display, start_display, add_menu, add_function_to_menu = init_display(
        # backend_str="pyside6"
    )

    #  Read shape and extract faces
    shape = read_step_file(STEP_FILE_PATH)
    explorer = TopologyExplorer(shape)

    faces = list(explorer.faces())
    print(f"Total faces extracted: {len(faces)}")

    # Display each face with a random color to distinguish boundaries
    display.EraseAll()
    for face in faces:
        random_color = Quantity_Color(
            random.random(), random.random(), random.random(), Quantity_TOC_RGB
        )
        display.DisplayColoredShape(face, random_color, update=False)

    # Refresh scene and start interactive GUI loop
    display.Context.UpdateCurrentViewer()
    display.FitAll()
    start_display()