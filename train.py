# This is the code to train model.
#
# Author: Weijie Wei
# Date: 02 / Jul / 2020
#
from __future__ import division
import argparse
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, preprocess_dof
import keras.backend as K
from models import ASD_SA
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction=0.9
set_session(tf.Session(config=config))


def scheduler(epoch):
    # reduce the learning rate every 3 epochs
    fac = 3

    if (epoch%fac)==0 and epoch!=0:
        lr = K.get_value(model.optimizer.lr)
        if lr > 1e-8:
          K.set_value(model.optimizer.lr, lr*.1)
          print("lr changed to {}".format(lr*.1))
    lr = K.get_value(model.optimizer.lr)
    lr = float(lr)
    print('lr: %0.9f' % lr)

    return lr


def generator(b_s, root_path, args=None, output_size=(480, 640)):

    imgs_path = root_path + '/Images/'
    maps_path = root_path + '/FixMaps/'
    fixs_path = root_path + '/FixPts/'

    images = [imgs_path + f for f in os.listdir(imgs_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    maps = [maps_path + f for f in os.listdir(maps_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    fixs = [fixs_path + f for f in os.listdir(fixs_path) if f.endswith('.mat')]

    images.sort()
    maps.sort()
    fixs.sort()

    if args.dreloss:
        maps_path_td = root_path + '/FixMaps_TD/'
        fixs_path_td = root_path + '/FixPts_TD/'

        maps_TD = [maps_path_td + f for f in os.listdir(maps_path_td) if f.endswith(('.jpg', '.jpeg', '.png'))]
        fixs_TD = [fixs_path_td + f for f in os.listdir(fixs_path_td) if f.endswith('.mat')]

        maps_TD.sort()
        fixs_TD.sort()

    counter = 0
    while True:
        Y = preprocess_maps(maps[counter:counter+b_s], output_size[0], output_size[1])
        Y_fix = preprocess_fixmaps(fixs[counter:counter + b_s], output_size[0], output_size[1])
        X = preprocess_images(images[counter:counter + b_s], args.input_size[0], args.input_size[1], 'channels_first')

        if args.dreloss:
            DOF = preprocess_dof(maps[counter:counter+b_s], maps_TD[counter:counter+b_s], output_size[0], output_size[1])
            yield [X], [Y, Y, Y_fix,
                        Y, Y, Y_fix,
                        Y, Y, Y_fix,
                        Y, Y, Y_fix,
                        Y, Y, Y_fix,
                        DOF]
        else:
            yield [X], [Y, Y, Y_fix,
                    Y, Y, Y_fix,
                    Y, Y, Y_fix,
                    Y, Y, Y_fix,
                    Y, Y, Y_fix]
        counter = (counter + b_s) % len(images)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Method setting
    parser.add_argument('--dreloss', default=False, type=bool)
    # parser.add_argument('--model_path', default='weights/weights_DRE_S4ASD--0.9714--1.0364.pkl', type=str)
    parser.add_argument('--model-path', default=None, type=str)
    parser.add_argument('--output-path', default='weights/', type=str)
    parser.add_argument('--train-set-path', default='./training_set/', type=str)
    # parser.add_argument('--train-set-path', default='path/to/training/set/', type=str)
    # parser.add_argument('--val-set-path', default='path/to/validation/set/', type=str)
    parser.add_argument('--val-set-path', default='./val_set/', type=str)

    # Model setting
    parser.add_argument('--init-lr', default=1e-4, type=float)
    parser.add_argument('--batch-size', default=5, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--input-size', default=(240, 320), type=tuple,
                        help='resize the input image, (640,480) is from the training data, SALICON.')

    args = parser.parse_args()

    # some fixed parameters
    output_size = (480, 640) # this is the output size of the model and then it will be restore to the same size with input image

    imgs_train_path = os.path.join(args.train_set_path, 'Images')
    nb_imgs_train = len([imgs_train_path + f for f in os.listdir(imgs_train_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    imgs_val_path = os.path.join(args.val_set_path, 'Images')
    nb_imgs_val = len([imgs_val_path + f for f in os.listdir(imgs_val_path) if f.endswith(('.jpg', '.jpeg', '.png'))])

    model = ASD_SA(img_cols=args.input_size[1], img_rows=args.input_size[0], DRE_Loss=args.dreloss, learning_rate=args.init_lr)
    model.summary()
    if args.model_path is not None:
        print("Load weights")
        weight_file = args.model_path
        model.load_weights(weight_file)
        print (weight_file)

    lr_sch = LearningRateScheduler(scheduler)
    checkpointdir= args.output_path
    print('save weights file at  '+checkpointdir)
    hist = model.fit_generator(generator(b_s=args.batch_size, root_path=args.train_set_path, args=args, output_size=output_size),
                               steps_per_epoch = (nb_imgs_train//args.batch_size),
                               validation_data=generator(b_s=args.batch_size, root_path=args.val_set_path, args=args, output_size=output_size),
                               validation_steps=(nb_imgs_val//args.batch_size),
                               epochs=args.epochs,
                               verbose=1,
                               # initial_epoch=2,
                               callbacks=[EarlyStopping(patience=3),
                                          ModelCheckpoint(checkpointdir+'{epoch:02d}-{val_loss:.4f}-{loss:.4f}.pkl',
                                          # save_weights_only=True,
                                          save_best_only=False),
                                          lr_sch])


    # display and save Loss Curve
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.title("model loss")
    plt.ylabel("Loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="lower right")
    # add data
    for x, y1, y2 in zip(range(len(loss)), loss, val_loss):
        plt.text(x, y1, '%0.4f'%(y1), ha='center', va='bottom', fontsize=8)
        plt.text(x, y2, '%0.4f'%(y2), ha='center', va='bottom', fontsize=8)
    plt.show()
    plt.savefig('curve.png')

