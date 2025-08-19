## Converts a png map file to a simple txt file where # represent walls

from pathlib import Path
from PIL import Image
import numpy as np
from dataset import generate_object_map


in_path = Path("/home/roman/learning-ml/2802ict-assignments/assignment_01/maze_src/cnn_heuristic/motion_planning_datasets/mazes/train/")
out_path = Path("/home/roman/learning-ml/2802ict-assignments/assignment_01/maze_src/cnn_compatible_maps")
number_to_generate = 10

for i in range(number_to_generate):
    read_path = in_path / f"{i}.png"
    write_path = out_path / f"{i}.txt"

    img = Image.open(read_path).convert("L")
    img = (255 - np.array(img))/255

    map = generate_object_map(img)

    with open(write_path, "w") as f:
        for row in range(len(map)):
            for col in range(len(map)):
                f.write(" " if map[row,col] == 0 else "#")
            f.write("\n")

    print(f"{i} complete")