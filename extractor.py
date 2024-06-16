yfcc100m = '/home/maometus/Documents/datasets/yfcc100m/YFCC100M-Downloader/data/images/'
mvtec_grid = '/home/maometus/Documents/datasets/mvtec_anomaly_detection/grid/train/good/'

import clip
import numpy as np
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt

keywords = [
    "mesh",
    "network",
    "web",
    "weave",
    "net",
    "lattice",
    "fabric",
    "interlacing",
    "entwining",
    "interweaving",
    "grid",
    "matrix",
    "layout",
    "framework",
    "arrangement",
    "structure",
    "scaffold",
    "skeleton",
    "hex",
    "hexagonal",
    "honeycomb",
    "six-sided",
    "tessellation",
    "pattern",
    "gridiron",
    "graticule",
    "reticulation",
    "reticulum",
    "reticle",
    "reticule",
    "hachure",
    "crisscross",
    "crosshatch",
    "crosshatching",
    "interlocking",
    "trellis",
    "grille",
    "checkerboard",
    "paving",
    "mosaic",
    "tiled",
    "quadrilateral",
    "quadrangle"
]
descriptions = [
    "A mesh of interconnected nodes forming a complex network.",
    "The grid-like structure provides a framework for organizing elements.",
    "Hexagonal patterns interweave to create a visually pleasing arrangement.",
    "The mesh of lines creates an intricate web of connections.",
    "A matrix of cells forms a structured layout for data representation.",
    "The honeycomb structure exhibits a hexagonal tessellation.",
    "Interlocking patterns create a crisscrossed grid.",
    "A network of interconnected nodes facilitates efficient communication.",
    "The lattice-like structure provides a foundation for organizing information.",
    "Hexagonal shapes tessellate to form an interconnected pattern.",
    "The matrix layout allows for efficient organization and retrieval of data.",
    "Interwoven lines create a complex network of connections.",
    "A grid-like framework organizes and aligns the elements.",
    "Hexagonal cells combine to form a honeycomb pattern.",
    "The interconnected nodes form a network for information exchange.",
    "The structured layout consists of a grid of interconnected elements.",
    "Hexagonal tiles create a visually appealing tessellation.",
    "A mesh of interlocking lines forms a complex grid structure.",
    "The network of nodes enables seamless communication and collaboration.",
    "The grid arrangement provides a systematic organization of elements.",
    "Hexagonal units interlock to create a cohesive pattern.",
    "The matrix structure offers a systematic arrangement of information.",
    "Interconnected lines form a web-like network.",
    "A grid-like matrix ensures efficient data representation and storage.",
    "Hexagonal shapes intertwine to form an intricate design.",
    "The lattice pattern creates a network of interconnected elements.",
    "The mesh of lines and nodes facilitates seamless connectivity.",
    "The grid-based layout allows for easy alignment and organization.",
    "Hexagonal cells interlock to create a visually striking arrangement.",
    "The interconnected nodes enable efficient data transmission.",
    "The grid-like structure offers a systematic arrangement of components.",
    "Hexagonal elements combine to form an interwoven pattern.",
    "The matrix layout provides a structured organization of data.",
    "Interlacing lines create a complex network of connections.",
    "A grid of cells ensures efficient information management.",
    "The hexagonal arrangement forms an interconnected network of elements.",
    "The lattice-like structure provides a foundation for organizing data.",
    "Interconnected lines form a web of relationships.",
    "A grid-like matrix facilitates efficient data processing.",
    "Hexagonal patterns combine to form an intricate tessellation.",
    "The network of nodes enables seamless information exchange.",
    "The grid layout ensures a systematic placement of elements.",
    "Hexagonal units interweave to create a visually appealing design.",
    "The structured matrix enables organized storage and retrieval of data."
]
descriptions = descriptions + ["This is a photo of " + keyword for keyword in keywords]
descriptions = descriptions + ["This is " + keyword for keyword in keywords]

model, preprocess = clip.load('ViT-B/32')
model.cuda().eval()

result = []

def traverse_files():
    for (root, dirs, files) in os.walk(yfcc100m, topdown=True):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image = Image.open(os.path.join(root, file)).convert("RGB")
                image_input = torch.tensor(np.stack([preprocess(image)])).cuda()
                desc_tokens = clip.tokenize(descriptions).cuda()
                with torch.no_grad():
                    image_features = model.encode_image(image_input).float()
                    text_features = model.encode_text(desc_tokens).float()
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    text_features /= text_features.norm(dim=-1, keepdim=True)
                    similarity = text_features @ image_features.T

                    if similarity.max() > 0.28:
                        result.append(os.path.join(root, file))
                    
                    if len(result) >= 100:
                        return

traverse_files()

print(result)

