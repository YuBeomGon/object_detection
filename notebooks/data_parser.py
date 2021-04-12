import os
import shutil
import numpy as np
import cv2
import re
import pickle
import random
import time
from tqdm import tqdm
import pandas as pd 
from glob import glob
from xml.etree.ElementTree import parse
# from pascal_voc_writer import Writer
import matplotlib.pyplot as plt


'''
    Condition Configuration
'''
# 사용 안할 annotation 목록
rej_table = [
    '삭제', 'pulled pork', 'abnormal', 'dog', 
    'ASCUS-RE',
    'ASCUS-Re',
    'ASCUS-SIL',
    'ASCwUS-SIL',
    'ASCUS-H',
    'ASC-H',
 
    'AGUS',
    'Reactive cell',
    'Reactive change',    

    'cavity',
    'ASCUS-koilocyte',
    'Lymphocytes',
    'leukocyte',
    'Lymphocyte',
    'leukocytes',
]

# 기 작성된 annotation에서 학습하고자 하는 단위로 교체
replace_table = { 
    'Normal-endocervical cells': 'Normal',
    'Normal-Autolytic parabasal cells': 'Normal',
    'Metaplastic cell-Nomal': 'Normal',
    'Normal-multi-nuclear cell': 'Normal',
    'Normal-metaplastic cell': 'Normal',
    'Normal-parabasal cell': 'Normal',
    'Normal-parabasal cells': 'Normal',
    'Normal-parabasal cells ': 'Normal',
    'Normal-Endocervical cell': 'Normal',
    'Endocervical cell-Normal': 'Normal',
    'Endocervical cell': 'Normal',
    'Endometrial cell': 'Normal',
    'Metaplastic cell': 'Normal',
    'Parabasal cell': 'Normal',
    'No malignant cell': 'Normal',
    'No nalignant cell': 'Normal',
    'No malinant cell': 'Normal',
    'No malignant cell-tissue repair': 'Normal',
    'No malignant cell-endocervical cell': 'Normal',
    'No malinant cell-endocervical cell': 'Normal',
    'No malignant cell-squamous metaplasia': 'Normal',
    'No maligant cell-parabasal cell': 'Normal',
    'No maligant cell-squamous metaplasia cell': 'Normal',
    'No maligant cell-endocervical cell': 'Normal',
    'No malignant cell-Squamous metaplastic cell': 'Normal',
    'No maligant cell-squamous metaplastic cell': 'Normal',
    'No malignant cell-metaplastic cell': 'Normal',

    'No malignant cell-Parabasal cell': 'Normal',
    'No malignant cell-parabasal cell': 'Normal',
    'No malinant cell-parabasal cell': 'Normal',
    'Autolytic parabasal cell': 'Normal',

    'normal': 'Normal',
    'ASCUS-US': 'ASCUS',
    
    'HSILw': 'HSIL',
    
    'Adenocarcinoma': 'Carcinoma',
    'Adenocarcinoma-endocervical type': 'Carcinoma',
    'Adenocarcinoma-endometrial type': 'Carcinoma',

    'Squamous metaplastic cell': 'Normal',
    
    'Squamous cell carcinomaw': 'Carcinoma',
    'Squamous cell carcinoma': 'Carcinoma',
    'Suamous cell carcinoma': 'Carcinoma',
    'Squamous cell carcinama': 'Carcinoma',
    
    #label to delete
    '삭제': '', 
    'pulled pork':'', 
    'abnormal':'', 
    'dog':'', 
    'ASCUS-RE':'',
    'ASCUS-Re':'',
    'ASCUS-SIL':'',
    'ASCwUS-SIL':'',
    'ASCUS-H':'',
    'ASC-H':'',
 
    'AGUS':'',
    'Reactive cell':'',
    'Reactive change':'',    

    'cavity':'',
    'ASCUS-koilocyte':'',
    'Lymphocytes':'',
    'leukocyte':'',
    'Lymphocyte':'',
    'leukocytes':'',  
    
    'Normal' : 'Normal', 
    'ASCUS' : 'ASCUS', 
    'HSIL' : 'HSIL', 
    'Carcinoma' : 'Carcinoma', 
    'LSIL' : 'LSIL', 
    'Benign' : 'Benign',
}

class XMLParser():
    def __init__(self, file_path):
        self.file_name = ''
        self.width = 0
        self.height = 0
        self.objects = []
#         self.labels = []
        self.rejection_size = [(3024, 4032)]
        
        tree = parse(file_path)
        root = tree.getroot()
        
        self.file_name = root.find('filename').text
        self.width = int(root.find('size').find('width').text)
        self.height = int(root.find('size').find('height').text)
        
        objs = root.findall('object')
        for obj in objs:
            orgcls = obj.find('name').text
            cls = replace_table[orgcls]
            xmin = int(obj.find('bndbox').find('xmin').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymax = int(obj.find('bndbox').find('ymax').text)
            bbox = [xmin, ymin, xmax, ymax]
            
            self.objects.append([cls,orgcls, bbox, xmin, ymin, xmax, ymax])
#             self.objectsName.append([self.file_name, cls, bbox, xmin, ymin, xmax, ymax])
#             if cls not in self.labels :
#                 self.labels.append(cls)
            
def make_3class_label(text) :
    if text == 'Normal' or text == 'Benign' :
        return 'Negative'
    elif text =='ASCUS' :
        return 'Low-Risk'
    else :
        return 'High-Risk'

    