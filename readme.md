## About

This is a small project where a treat a dataset available on kaggle to train a yolo model. 
The data set contain images of PCBs with various defects.

* Dataset source url: https://www.kaggle.com/datasets/akhatova/pcb-defects

## data.yaml file was manually made with the bellow text:

train: 'train/images'

val: 'val/images'

test: 'test/images'

nc: 6

names: ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']