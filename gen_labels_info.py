import numpy as np
import glob
from utils.xml_parser import XMLParser


def generate_labels_info(out_path="./data/labels_info.npy"):
    home_dir_path = "/home/Dataset/Papsmear/original/"
    file_types = ["SS/*/*.jpg", "SS2/*/*.jpg"]

    img_path_list = []
    for ftype in file_types:
        img_path_list.extend(glob.glob(home_dir_path + ftype))
    
    print("Num of Images: {}".format(len(img_path_list)))

    labels_info = {}
    for idx, img_path in enumerate(img_path_list):
        try:
            xml_path = img_path[:-3] + "xml"
            parser = XMLParser(xml_path)    
            ID = img_path.split("original/")[-1]
            labels = parser.objects
            labels_info[ID] = labels
        
        except:
            pass

        if idx % 1000 == 0:
            print(idx, img_path)

    np.save(out_path, labels_info)


if __name__ == '__main__':
    out_path = './data/labels_info.npy'
    generate_labels_info(out_path)

