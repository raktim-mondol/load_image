# How to load images for Segmentation and Classification?

### Directory Layout
## Segmentation Folder Structure

```bash
├── train_folder
│   ├── data/
│   │   ├── image
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
|   |   ├── mask
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif

OR

├── train_folder
│   ├── sample1
│   │   ├── image
|   |   ├── mask
│   ├── sample2
│   │   ├── image
|   |   ├── mask
│   ├── sample3
│   │   ├── image
|   |   ├── mask

```
## Classifcation Folder Structure 
├── train_folder
│   ├── monkey
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   ├── tiger
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
│   ├── dog
│   │    ├── 1.tif
│   │    ├── 2.tif
│   │    ├── 3.tif
```

### Data Load for Segmentation
```python

from data_class import Data
IMAGE_PATH = 'C:/Users/example/train_folder/'

data_obj= Data()
X_train, Y_train = data_obj.load_segmentation_data(IMAGE_PATH, 'tif', 256, 256)

# Visualize image with corresponding mask
data_obj.visualize(X_train,Y_train)

```
### Data Load for Classification 

```python

from data_class import Data
IMAGE_PATH = 'C:/Users/example/train_folder/'

data_obj= Data()
X_train, Y_train = data_obj.load_classification_data(IMAGE_PATH, 'tif', 256, 256, to_cat=True)
# to_catagorical=True will return one hot encoded value to Y_train

#Check Data Label
data_obj.label_check()

# Visualize image with corresponding mask
data_obj.visualize(X_train)

```



