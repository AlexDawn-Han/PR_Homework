import os
import numpy as np
import xml.etree.ElementTree as ET

LABEL_NAME = 'licence'  # car 

def extract_box_dimensions(xml_path):
    """
    解析给定路径的xml文件，提取object下name为LABEL_NAME的box的高宽信息。
    
    :param xml/XMLSchema Document: xml文件的路径
    :return: 包含高度和宽度的列表，每个元素为(car_width, car_height)
    """
    dimensions = []
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        if obj.find('name').text == LABEL_NAME:
            bndbox = obj.find('bndbox')
            width = int(bndbox.find('xmax').text) - int(bndbox.find('xmin').text)
            height = int(bndbox.find('ymax').text) - int(bndbox.find('ymin').text)
            dimensions.append((width, height))
    
    return dimensions


if __name__ == '__main__':
    """
    输出训练集labels文件夹下的所有xml文件的LABEL_NAME对象统计特征。
    """
    path = r'D:\Datasets\PR-homework\dataset\test\labels'
    all_dimensions = []

    # 遍历指定文件夹下的所有.xml文件
    for filename in os.listdir(path):
        if filename.endswith('.xml'):
            all_dimensions.extend(extract_box_dimensions(os.path.join(path, filename)))

    # 输出统计特征
    widths, heights = zip(*all_dimensions)
    print(f"总共有 {len(all_dimensions)} 个标注为{LABEL_NAME}的对象.")
    print(f"宽度的最大值：{max(widths)}, 最小值：{min(widths)}, 平均值：{np.mean(widths)},方差：{np.var(widths)}")
    print(f"高度的最大值：{max(heights)}, 最小值：{min(heights)}, 平均值：{np.mean(heights)},方差：{np.var(heights)}")
