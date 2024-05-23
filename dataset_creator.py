import os
import pandas as pd

from tabulate import tabulate
from shutil import copyfile
from glob import glob
from xml.etree import ElementTree as et
from sklearn.model_selection import train_test_split


class_map = {'missing_hole': 0, 'mouse_bite': 1, 'open_circuit': 2, 'short': 3, 'spur': 4, 'spurious_copper': 5}
def get_class_id(class_name: str) -> int:
    return class_map[class_name]


def extract_img_text(xml_path: str) -> list[dict]:
    tree = et.parse(xml_path)
    root = tree.getroot()

    list_dicts = []

    objects = root.findall('object')
    for ind, obj in enumerate(objects):
        dict_info = {}

        dict_info['img_name'] = root.find('filename').text
        dict_info['img_width'] = root.find('size').find('width').text
        dict_info['img_height'] = root.find('size').find('height').text

        bndbox = obj.find('bndbox')
        dict_info['obj_name'] = obj.find('name').text
        dict_info['x_min'] = bndbox.find('xmin').text
        dict_info['x_max'] = bndbox.find('xmax').text
        dict_info['y_min'] = bndbox.find('ymin').text
        dict_info['y_max'] = bndbox.find('ymax').text

        list_dicts.append(dict_info)

    return list_dicts


def create_label(img_name: str, df: pd.DataFrame, dest_path: str) -> None:
    text_file = os.path.join(dest_path, img_name.replace("jpg","txt"))
    text_file = text_file.replace('\\', '/')

    cols2save = ['img_name', 'class_id', 'center_x', 'center_y', 'box_width', 'box_height']
    groupby_obj = df[cols2save].groupby('img_name')
    #print(tabulate(groupby_obj.get_group(img_name), headers='keys', tablefmt='psql'))
    groupby_obj.get_group(img_name).set_index('img_name').to_csv(text_file, sep=' ', index=False, header=False)


def main() -> None:
    annotations_dir = r'kaggle_pcb_dataset/Annotations'
    images_dir = r'kaggle_pcb_dataset/images'

    # Get source paths
    list_anotations_src_dirs = [os.path.join(annotations_dir, class_name) for class_name in os.listdir(annotations_dir)]
    replace_text = lambda x: x.replace('\\', '/')
    list_anotations_src_dirs = list(map(replace_text, list_anotations_src_dirs))

    list_images_src_dirs = [os.path.join(images_dir, class_name) for class_name in os.listdir(images_dir)]
    list_images_src_dirs = list(map(replace_text, list_images_src_dirs))


    # get all xml files paths
    all_ann_files = []
    for each_class in list_anotations_src_dirs:
        xml_files = glob(each_class + '/*.xml')
        all_ann_files += list(map(replace_text, xml_files))


    # get all images files paths
    all_img_files = []
    for each_class in list_images_src_dirs:
        img_files = glob(each_class + '/*.jpg')
        all_img_files += list(map(replace_text, img_files))


    # print(extract_img_text('kaggle_pcb_dataset/Annotations/Missing_hole/01_missing_hole_01.xml'))
    all_annotations = []
    for xml_file in all_ann_files:
        all_annotations += extract_img_text(xml_file)

    df_all = pd.DataFrame(all_annotations)


    # check dataframe infos
    print('---------------------# Dataframe Head #---------------------\n')
    print(tabulate(df_all.head(), headers='keys', tablefmt='psql'))
    print('\n\n---------------------# Dataframe info #---------------------\n')
    print(df_all.info())
    print(f'\n\n## Columns: {df_all.columns}')
    print(f'\n\n# Total num objects: {df_all.shape[0]}')
    print('\n\n---------------------# Amount each object #---------------------\n')
    print(df_all['obj_name'].value_counts())

    # fix dataframe types
    df_all['class_id'] = df_all['obj_name'].apply(get_class_id)

    int_cols = ['img_width', 'img_height', 'x_min', 'x_max', 'y_min', 'y_max', 'class_id']
    str_cols = ['img_name', 'obj_name']

    df_all[int_cols] = df_all[int_cols].astype(int)
    df_all[str_cols] = df_all[str_cols].astype(str)
    print('\n\n---------------------# Dataframe info #---------------------\n')
    print(df_all.info())


    # Create YOLO format box coodinates and add class ids
    df_all['center_x'] = ((df_all['x_min'] + df_all['x_max'])/2)/df_all['img_width']
    df_all['center_y'] = ((df_all['y_min'] + df_all['y_max'])/2)/df_all['img_height']
    df_all['box_width'] = (df_all['x_max'] - df_all['x_min'])/df_all['img_width']
    df_all['box_height'] = (df_all['y_max'] - df_all['y_min'])/df_all['img_height']


    print('\n\n---------------------# Dataframe Head #---------------------\n')
    print(tabulate(df_all.head(), headers='keys', tablefmt='psql'))


    # CREATING YOLO TRAIN / TEST / VALID FOLDERS
    train_images, val_images = train_test_split(all_img_files, test_size=0.3, random_state=9) # 70% train
    val_images, test_images = train_test_split(val_images, test_size=0.5, random_state=9) # 15% test / 15% validation

    print(f'\n\n# Total Images: {len(all_img_files)}')
    print(f'# test size: {len(train_images)} - {len(train_images) / len(all_img_files):.2f}')
    print(f'# test size: {len(test_images)} - {len(test_images) / len(all_img_files):.2f}')
    print(f'# test size: {len(val_images)} - {len(val_images) / len(all_img_files):.2f}')

    # start creating YOLO_DATASET
    folders = ['', 'train', 'test', 'val']
    for folder in folders:
        path = rf'yolo_pcb_dataset/{folder}'
        if not os.path.isdir(path):
            os.mkdir(path)

        if folder != '':
            img_path_exist = os.path.isdir(img_path := os.path.join(path, 'images'))
            ann_path_exist = os.path.isdir(ann_path := os.path.join(path, 'labels'))
            if not img_path_exist:
                os.mkdir(img_path)
            if not ann_path_exist:
                os.mkdir(ann_path)

    # Copie images from original dataset to the new yolo dataset
    img_src_dict = {'train': train_images, 'test': test_images, 'val': val_images}
    print_buffer = []
    for folder in folders[1:]:
        path = rf'yolo_pcb_dataset/{folder}/images'
        for src_path in img_src_dict[folder]:
            dest_path = os.path.join(path, src_path.split('/')[-1])
            dest_path = replace_text(dest_path)
            copyfile(src_path, dest_path)
            print_buffer.append(f'copied \'{src_path}\' to \'{dest_path}\'')

    with open('copies_log.txt', "w") as cpy_file:
        print("\n".join(print_buffer), file=cpy_file)

    for folder in folders[1:]:
        img_path = rf'yolo_pcb_dataset/{folder}/images'
        ann_path = rf'yolo_pcb_dataset/{folder}/labels'
        for image_src_path in img_src_dict[folder]:
            img_name = image_src_path.split('/')[-1]
            create_label(img_name=img_name, df=df_all, dest_path=ann_path)


if __name__ == '__main__':
    main()