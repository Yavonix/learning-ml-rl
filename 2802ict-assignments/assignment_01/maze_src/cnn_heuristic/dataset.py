import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from pathlib import Path
import random
from scipy.ndimage import distance_transform_edt
from PIL import Image

class SingleFolderDataset(Dataset):
    def __init__(self, folder, transform=None, exts=(".png",".jpg",".jpeg")):
        p = Path(folder)
        self.files = [f for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("L")   # single-channel greyscale
        img = np.array(img)
        if self.transform:
            img = self.transform(img)

        # Feature Processing
        
        img = (255 - np.array(img))/255

        obstacle_map_np = np.ones((224,224))
        size_h = obstacle_map_np.shape[0] - img.shape[0]
        size_w = obstacle_map_np.shape[1] - img.shape[1]

        rand_h = int(random.uniform(0, size_h))
        rand_w = int(random.uniform(0, size_w))

        ## Obstacle Map
        obstacle_map_np[rand_h:rand_h+img.shape[0], rand_w:rand_w+img.shape[1]] = img

        ## Distance Map

        # euclidian distance transform
        distance_map_np = np.array(distance_transform_edt(1 - obstacle_map_np))
        # distance_map_np = distance_map_np / distance_map_np.max()

        ## Goal Map
        goal = (int(random.uniform(rand_h, rand_h+img.shape[0])), int(random.uniform(rand_w, rand_w+img.shape[1])))

        # make sure goal is not obstacle
        while obstacle_map_np[goal[0], goal[1]] == 1:
            goal = (int(random.uniform(rand_h, rand_h+img.shape[0])), int(random.uniform(rand_w, rand_w+img.shape[1])))
        
        y_coords = np.arange(obstacle_map_np.shape[0])
        x_coords = np.arange(obstacle_map_np.shape[1])
        xx, yy = np.meshgrid(x_coords, y_coords)
        squared_dist = (xx - goal[1])**2 + (yy - goal[0])**2
        # 5. Take the square root to get the final Euclidean distance
        goal_map_np = np.sqrt(squared_dist)
        # goal_map_np = goal_map_np / goal_map_np.max()

        final = np.stack([obstacle_map_np, distance_map_np, goal_map_np], axis=-1)

        # Label processing

        heuristic_map = np.full_like(obstacle_map_np, 0)

        starting_node = (*goal, 0)
        # x, y, cost
        nodes: list[tuple[int,int,int]] = [starting_node]
        reached: set[tuple[int,int]] = {goal}

        while nodes:
            # print(nodes)
            cur_node = nodes.pop(0)
            loc = cur_node[0:2]
            # print(cur_node)
            heuristic_map[loc] = cur_node[2]

            new_nodes: list[tuple[int,int,int]] = [(cur_node[0]+1, cur_node[1], cur_node[2]+1),
                                                   (cur_node[0]-1, cur_node[1], cur_node[2]+1),
                                                   (cur_node[0], cur_node[1]+1, cur_node[2]+1),
                                                   (cur_node[0], cur_node[1]-1, cur_node[2]+1)]
            

            def valid(node: tuple[int,int,int]):
                loc = node[0:2]
                if (loc[0] < 0 or loc[0] >= heuristic_map.shape[0]): return False
                if (loc[1] < 0 or loc[1] >= heuristic_map.shape[1]): return False
                if loc in reached: return False
                if obstacle_map_np[loc] == 1: return False
                reached.add(loc)
                return True

            nodes.extend(filter(valid, new_nodes))

        # heuristic_map = heuristic_map / heuristic_map.max()
        heuristic_map = heuristic_map

        heuristic_mask = np.where(heuristic_map == 0, 0, 1)
        heuristic_mask[goal] = 1

        final_label = np.stack([heuristic_map, heuristic_mask], axis=-1)

        # heuristic_map = heuristic_map / heuristic_map.max()
        
        return final, final_label  # dummy label 0
    

def show_image(img: np.ndarray, title:str=""):
    # img = np.concatenate(img, axis=0)
    # img = img[:,:,0]
    
    if (len(img.shape) > 2):
        width = img.shape[1]
        channels = img.shape[2]
        output_img = np.zeros((img.shape[0], img.shape[1]*channels))
        for i in range(channels):
            output_img[:,i*width:(i+1)*width] = img[:,:,i]
            # normalise
            output_img[:,i*width:(i+1)*width] = output_img[:,i*width:(i+1)*width] / output_img[:,i*width:(i+1)*width].max()
    else:
        output_img = np.array(img)
        # normalise
        output_img = output_img / output_img.max()
    output_img = (output_img*255.0).astype(dtype=np.uint8) 
    # img = (255.0 - (img*255.0)).astype(dtype=np.uint8)
    image = Image.fromarray(output_img, mode="L")
    image.show(title)