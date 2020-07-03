# This is the code to train model.
#
# Author: Weijie Wei
# Date: 03 / Jul / 2020
#
from __future__ import division
import cv2, os, argparse
from utilities import preprocess_images, postprocess_predictions
from models import ASD_SA
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
set_session(tf.Session(config=config))




def generator_test(b_s, imgs_test_path, large_scale_dataset=False, group=0, th=500, args=None):
    images = [imgs_test_path + f for f in os.listdir(imgs_test_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()

    counter = 0
    if large_scale_dataset:
        assert group!=0
        start = (group-1)*th
        end_ = min(group*th, len(images))
        images = images[start:end_]
        while True:
            yield [preprocess_images(images[counter:counter + b_s], args.input_size[0], args.input_size[1])]
            counter = (counter + b_s) % len(images)

    else:
        while True:
            yield [preprocess_images(images[counter:counter + b_s], args.input_size[0], args.input_size[1])]
            counter = (counter + b_s) % len(images)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Method setting
    parser.add_argument('--model-path', default='weights/weights_DRE_S4ASD--0.9714--1.0364.pkl', type=str)
    # parser.add_argument('--model-path', default=None, type=str)
    parser.add_argument('--images-path', default='images/', type=str)
    parser.add_argument('--results-path', default='results/', type=str)

    # Model setting
    parser.add_argument('--init-lr', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--input-size', default=(240, 320), type=tuple,
                        help='resize the input image, (640,480) is from the training data, SALICON.')

    args = parser.parse_args()

    # some fixed parameters
    output_size = (480, 640) # this is the output size of the model and then it will be restore to the same size with input image


    images_path = args.images_path
    file_names = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    nb_imgs_test = len(file_names)
    print (nb_imgs_test)

    output_folder = args.results_path
    if os.path.isdir(output_folder) is False:
        os.makedirs(output_folder)

    print("Predict saliency maps for " + images_path + " at "+  output_folder)

    model = ASD_SA(img_cols=args.input_size[1], img_rows=args.input_size[0], learning_rate=args.init_lr)
    print("Load weights")
    weight_file = args.model_path
    if os.path.exists(weight_file):
        model.load_weights(weight_file)

    th = 500 # If the number of images is larger than the threshold, then predict the images group by group
    if nb_imgs_test <= th:
        predictions = model.predict_generator(generator_test(b_s=1, imgs_test_path=images_path, args=args), nb_imgs_test)
        predictions = predictions[-1]
        print (len(predictions))

        for pred, name in zip(predictions, file_names):
            original_image = cv2.imread(os.path.join(images_path, name), 0)
            name = name[:-4] + '.png'
            res = postprocess_predictions(pred, original_image.shape[0], original_image.shape[1])
            cv2.imwrite(output_folder + '%s' % name, res.astype(int))
        print('Done!')
    else:
        nb_groups = nb_imgs_test//th
        print('Total group: ', nb_groups)

        for i in range(nb_groups):
            group = i+1
            nb_imgs_test_in_this_group = th if group<nb_groups else (nb_imgs_test-(group-1)*th)
            predictions = model.predict_generator(
                generator_test(b_s=1, imgs_test_path=images_path, large_scale_dataset=True, group=group, args=args),
                nb_imgs_test_in_this_group)
            predictions = predictions[-1]

            start = (group - 1) * th
            end_ = min(group * th, nb_imgs_test)
            images_filename_in_this_group = file_names[start:end_]
            for pred, name in zip(predictions, images_filename_in_this_group):
                original_image = cv2.imread(os.path.join(images_path, name), 0)
                name = name[:-4] + '.png'
                res2 = postprocess_predictions(pred, original_image.shape[0], original_image.shape[1])
            print("%d / %d"%(group,nb_groups))

