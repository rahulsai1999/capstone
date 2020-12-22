import os
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET

img = []
img_impression = []
img_finding = []

directory = "C:/Users/Rahul Sai/Desktop/Capstone/medical-captioning/ecgen-radiology"

for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".xml"):
        f = directory + '/' + filename
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if child.tag == 'MedlineCitation':
                for attr in child:
                    if attr.tag == 'Article':
                        for i in attr:
                            if i.tag == 'Abstract':
                                for name in i:
                                    if name.get('Label') == 'FINDINGS':
                                        finding = name.text
                                    elif name.get('Label') == 'IMPRESSION':
                                        impression = name.text
        for p_image in root.findall('parentImage'):
            img.append(p_image.get('id'))
            img_finding.append(finding)
            img_impression.append(impression)

df = pd.DataFrame({'image_uri': img, 'image_finding': img_finding,
                   'image_impression': img_impression})
df.to_csv('data.csv')
