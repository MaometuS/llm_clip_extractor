yfcc100m = '/beegfs/rakhimov/yfcc100m/zip/'
output_folder = '/beegfs/rakhimov/yfcc100m_extracted/'


import clip
import zipfile
import numpy as np
import torch
import os
from keywords import bottle, cable, capsule, carpet, grid, hazelnut, leather, metal_nut, pill, screw, tile, toothbrush, transistor, wood, zipper
from PIL import Image


description_array = [
    ('bottle', bottle.descriptions),
    ('cable', cable.descriptions),
    ('capsule', capsule.descriptions),
    ('carpet', carpet.descriptions),
    ('grid', grid.descriptions),
    ('hazelnut', hazelnut.descriptions),
    ('leather', leather.descriptions),
    ('metal_nut', metal_nut.descriptions),
    ('pill', pill.descriptions),
    ('screw', screw.descriptions),
    ('tile', tile.descriptions),
    ('toothbrush', toothbrush.descriptions),
    ('transistor', transistor.descriptions),
    ('wood', wood.descriptions),
    ('zipper', zipper.descriptions),
]

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
        checkpointFile.write('000')
        checkpointFile.close()
        checkpointStr = '000'

    checkpoint = int(checkpointStr, 16)
    endpoint = int('fff', 16)+1
    for i in range(checkpoint, endpoint):
        currPoint = intToHexName(i)
        f = open(os.path.join(output_folder, "checkpoint.txt"), "w")
        f.write(currPoint)
        f.close()
        process_zip(currPoint)

def intToHexName(num):
    zipname = hex(num)[2:]
    if len(zipname) == 1:
        zipname="00" + zipname
    
    if len(zipname) == 2:
        zipname="0" + zipname

    return zipname

def process_zip(zipname):
    currZip = zipfile.ZipFile(os.path.join(yfcc100m, zipname+".zip"))
    fileList = currZip.namelist()

    for fileName in fileList:
        if fileName.endswith(".jpg") or fileName.endswith(".jpeg") or fileName.endswith(".png"):
            for pair in description_array:
                file = currZip.open(fileName)
                isFitting = clipTest(file, pair[1])
                file.close()

                if isFitting:
                    currZip.extract(fileName, os.path.join(output_folder, pair[0], zipname, os.path.basename(fileName)))
    
def clipTest(file, descriptions):
    image = Image.open(file).convert("RGB")
    image_input = torch.tensor(np.stack([preprocess(image)])).cuda()
    desc_tokens = clip.tokenize(descriptions).cuda()
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(desc_tokens).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features @ image_features.T

        return similarity.max() > 0.28

process_zip('000')