import tensorflow_datasets as tfds

# 下載並載入 WIDER FACE 資料集
dataset, info = tfds.load('wider_face', with_info=True, as_supervised=True)
train_dataset = dataset['train']
validation_dataset = dataset['validation']
test_dataset = dataset['test']
