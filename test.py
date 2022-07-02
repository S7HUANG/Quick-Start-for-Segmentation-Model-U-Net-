from seg_unet import unet
from dataProcess import testGenerator, saveResult, color_dict
import os

model_path = "Model/0418_unet_model.hdf5"  # Model saving path

test_image_path = "Data/test/image"
test_label_path = "Data/test/label"

save_path = r"Predict"

test_num = len(os.listdir(test_image_path))

classNum = 13

input_size = (512, 512, 3)

output_size = (492, 492)

colorDict_RGB, colorDict_GRAY = color_dict(test_label_path, classNum)

model = unet(model_path)

testGene = testGenerator(test_image_path, input_size)

#  Numpy array of predicted values
results = model.predict_generator(testGene,
                                  test_num,
                                  verbose = 1)


saveResult(test_image_path, save_path, results, colorDict_GRAY, output_size)