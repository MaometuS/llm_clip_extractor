yfcc100m = '/beegfs/rakhimov/yfcc100m/zips/'
output_folder = '/beegfs/rakhimov/yfcc100m_extracted/'


import clip
import zipfile
import numpy as np
import torch
import os
from PIL import Image

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

def traverse_zips():
    checkpointStr = '000'

    try:
        checkpointFile = open(os.path.join(output_folder, "checkpoint.txt"), "r")
        checkpointStr = checkpointFile.readline()
        checkpointFile.close()
    except:
        checkpointFile = open(os.path.join(output_folder, "checkpoint.txt"), "w")
        checkpointStr = checkpointFile.write('000')
        checkpointFile.close()

    checkpoint = int(checkpointStr, 16)
    endpoint = int('fff', 16)+1
    for i in range(checkpoint, endpoint):
        currPoint = hex(i)[3:]
        f = open(os.path.join(output_folder, "checkpoint.txt"), "w")
        f.write(currPoint)
        f.close()
        process_zip(currPoint)

def process_zip(zipname):
    if len(zipname) == 1:
        zipname+="00"
    
    if len(zipname) == 2:
        zipname+="0"

    currZip = zipfile.ZipFile(os.path.join(yfcc100m, zipname+".zip"))
    fileList = currZip.namelist()

    for fileName in fileList:
        if fileName.endswith(".jpg") or fileName.endswith(".jpeg") or fileName.endswith(".png"):
            file = currZip.open(fileName)
            isFitting = clipTest(file)
            file.close()

            if isFitting:
                currZip.extract(fileName, os.path.join(output_folder, zipname, fileName))
    
def clipTest(file):
    image = Image.open(file).convert("RGB")
    image_input = torch.tensor(np.stack([preprocess(image)])).cuda()
    desc_tokens = clip.tokenize(descriptions).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(desc_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features @ image_features.T

        return similarity > 0.28

traverse_zips()