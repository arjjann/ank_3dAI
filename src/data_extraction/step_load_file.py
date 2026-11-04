import os
import os.path
import random
import sys

from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

step_path=os.path.expanduser(os.path.join("~", "ank_3dAI", "data", "CADs", "bar.STEP"))

def import_as_one_shape(event=None):
    shp = read_step_file(step_path)
    display.EraseAll()
    display.DisplayShape(shp, update=True)
    display.FitAll()


def import_as_multiple_shapes(event=None):
    compound = read_step_file(step_path)
    t = TopologyExplorer(compound)
    display.EraseAll()
    for solid in t.solids():
        color = Quantity_Color(
            random.random(), random.random(), random.random(), Quantity_TOC_RGB
        )
        display.DisplayColoredShape(solid, color)
    display.FitAll()


def exit(event=None):
    sys.exit()


if __name__ == "__main__":
    display, start_display, add_menu, add_function_to_menu = init_display()
    add_menu("STEP import")
    add_function_to_menu("STEP import", import_as_one_shape)
    add_function_to_menu("STEP import", import_as_multiple_shapes)

    display.FitAll()
    start_display()
