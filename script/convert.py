# %%
import json
import os

# %%

def coco_to_yolo(json_path, output_dir):
    # 加载COCO格式的JSON文件
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取类别列表
    categories = coco_data['categories']
    category_mapping = {cat['id']: idx for idx, cat in enumerate(categories)}

    # 获取图像和标注信息
    images = {img['id']: img for img in coco_data['images']}
    annotations = coco_data['annotations']

    # 遍历标注数据
    for annotation in annotations:
        image_id = annotation['image_id']
        image_info = images[image_id]
        image_width = image_info['width']
        image_height = image_info['height']

        # 计算归一化坐标
        bbox = annotation['bbox']
        x_min, y_min, box_width, box_height = bbox
        x_center = (x_min + box_width / 2) / image_width
        y_center = (y_min + box_height / 2) / image_height
        width = box_width / image_width
        height = box_height / image_height

        # 获取类别ID并映射到YOLO格式的类别索引
        category_id = annotation['category_id']
        class_id = category_mapping[category_id]

        # YOLO格式标注
        yolo_annotation = f"{class_id} {x_center} {y_center} {width} {height}\n"

        # 保存为对应的txt文件
        txt_file = os.path.join(output_dir, f"{image_info['file_name'].split('.')[0]}.txt")
        with open(txt_file, 'a') as f:
            f.write(yolo_annotation)

    print(f"转换完成，YOLO格式文件已保存在 {output_dir}")


# %%
# 示例用法
json_path = r"C:\Users\jeffe\Downloads\coco2017_50\coco\annotations\instances_train2017.json"  # 替换为COCO JSON文件路径
output_dir = r"C:\Users\jeffe\Downloads\coco2017_50\coco\train2017"      # 替换为输出目录
coco_to_yolo(json_path, output_dir)


# %%
json_path = r"C:\Users\jeffe\Downloads\coco2017_50\coco\annotations\instances_val2017.json"  # 替换为COCO JSON文件路径
output_dir = r"C:\Users\jeffe\Downloads\coco2017_50\coco\val2017"      # 替换为输出目录
coco_to_yolo(json_path, output_dir)

# %%
def create_index_file(image_dir, output_file):
    """
    创建索引文件，将指定目录下的所有图像路径写入文件。

    :param image_dir: 图像文件所在目录
    :param output_file: 输出索引文件路径
    """
    # 遍历图像目录，收集所有图片路径
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    # 按路径排序（可选）
    image_paths.sort()

    # 写入索引文件
    with open(output_file, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

    print(f"索引文件已保存到 {output_file}")

# %%
# 示例用法
train_image_dir = "images/train1"  # 替换为你的训练集图像目录
val_image_dir = "images/val"      # 替换为你的验证集图像目录

create_index_file(train_image_dir, "train.txt")
create_index_file(val_image_dir, "val.txt")



