import os
import sys
from os import listdir

import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
from skimage import img_as_float
from sklearn.externals import joblib
from sklearn.svm import LinearSVC


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


NUM_CLASSES = 2


def print_params(list_params):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in xrange(1, len(sys.argv)):
        print list_params[i - 1] + '= ' + sys.argv[i]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


def read_npys(path, features, template_name, is_diff=False):
    first = True
    for i in xrange(len(features)):
        name = path + features[i] + '/'
        if is_diff is True:
            name = name + 'test_diff/'
        cur_npy = np.load(name + template_name)
        print name, cur_npy.shape
        if first is True:
            all_npys = cur_npy
            first = False
        else:
            all_npys = np.concatenate([all_npys, cur_npy], axis=3)

    print all_npys.shape
    return all_npys


def load_images(path, specific_event=None):
    images = []
    masks = []

    for d in listdir(path):
        if "tar.gz" not in d and ".py" not in d and "npy" not in d and "zip" not in d and "output" not in d and \
                        "txt" not in d and "new" not in d and "old" not in d and "test" != d and "sh" not in d and \
                        "aux" not in d and "pkl" not in d:
            if specific_event is None or (specific_event is not None and specific_event == int(d.split("_")[1])):
                print BatchColors.WARNING + "Reading event " + d.split("_")[1] + BatchColors.ENDC
                for f in listdir(path + d):
                    if "tif" not in f and ".png.aux.xml" not in f:
                        try:
                            if 'mask' in d:
                                img = scipy.misc.imread(path + d + '/' + f)
                                masks.append((int(f[9:15]), img))
                            else:
                                img = img_as_float(scipy.io.loadmat(path + d + '/' + f)['img'])
                                images.append((int(f[:-4]), img))
                        except IOError:
                            print BatchColors.FAIL + "Could not open/read file: " + \
                                  path + d + '/' + f + BatchColors.ENDC

    masks.sort(key=lambda tup: tup[0])
    images.sort(key=lambda tup: tup[0])

    return images, masks


def create_patches(data, class_distribution, crop_size=25):
    if len(class_distribution) == 2:
        class_distribution = class_distribution[0] + class_distribution[1]

    patches = []

    for i in xrange(len(class_distribution)):
        cur_map = class_distribution[i][0]
        cur_x = class_distribution[i][1][0]
        cur_y = class_distribution[i][1][1]

        patch = data[cur_map][1][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

        if len(patch) != crop_size or len(patch[0]) != crop_size:
            print "Error: Current patch size ", len(patch), len(patch[0])
            print cur_map, cur_x, cur_y
            return

        patches.append(patch)

    return np.asarray(patches)


def save_map(path, prob_im_argmax, data):
    for i in xrange(len(data)):
        name = format(data[i][0], '06')

        img = Image.fromarray(np.uint8(prob_im_argmax[i] * 255))
        img.save(path + name + '.png')

        scipy.misc.toimage(prob_im_argmax[i], cmin=0.0, cmax=255).save(path + 'seg_mask_' + name + '.png')


def svm(train_feat, train_class):
    print BatchColors.WARNING + 'Training SVM...' + BatchColors.ENDC
    clf = LinearSVC(random_state=0, max_iter=1000)
    clf.fit(train_feat, train_class)
    joblib.dump(clf, os.getcwd() + '/' + 'clf.pkl')
    print BatchColors.OKGREEN + 'SVM trained!' + BatchColors.ENDC
    return clf


def prediction(clf, test_feat):
    print BatchColors.WARNING + 'Predicting...' + BatchColors.ENDC
    pred_arr = np.empty([len(test_feat), len(test_feat[0]), len(test_feat[0][0])], dtype=np.uint8)
    for i in xrange(len(test_feat)):
        test_feat_arr = test_feat[i].reshape(len(test_feat[i]) * len(test_feat[i][0]), len(test_feat[i][0][0]))
        pred = clf.predict(test_feat_arr)
        pred_arr[i] = pred.reshape(len(test_feat[i]), len(test_feat[i][0]))
    print BatchColors.OKGREEN + '... done!' + BatchColors.ENDC
    return pred_arr


'''
python metaLearning.py /home/mediaeval17/FDSI/ /home/mediaeval17/FDSI/new/ /home/mediaeval17/FDSI/test/ 
/home/mediaeval17/FDSI/test/output_meta/
'''


def main():
    list_params = ['input_path', 'input_train_npyPath', 'input_test_npyPath', 'output_path']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    i = 1
    input_path = sys.argv[i]
    i = i + 1
    input_train_npy = sys.argv[i]
    i = i + 1
    input_test_npy = sys.argv[i]
    i = i + 1
    output_path = sys.argv[i]

    train_features = ['output_dilatedICPR', 'output_dilatedICPR_25_1', 'output_dilatedICPR_25_2',
                      'output_dilatedICPR_25_3', 'output_dilatedICPR_25_4', 'output_dilatedICPR_25_5',
                      'output_dilatedICPR_25_6', 'output_dilatedICPR_50', 'output_dilatedGRSL', 'output_segNet_25',
                      'output_segnetICPR']  # considered networks

    # read training data
    print BatchColors.WARNING + 'Reading Data...' + BatchColors.ENDC
    _, labels = load_images(input_path)
    validationclass_distribution = np.load(os.getcwd() + '/validationclass_distribution_25.npy')
    training_label_patches = create_patches(labels, validationclass_distribution, crop_size=25)
    train_feat = read_npys(input_train_npy, train_features, template_name='prob_im.npy')

    test_features = ['output_dilatedICPR_25', 'output_dilatedICPR_25_1_allValidation',
                     'output_dilatedICPR_25_2_allValidation', 'output_dilatedICPR_25_3_allValidation',
                     'output_dilatedICPR_25_4_allValidation', 'output_dilatedICPR_25_5_allValidation',
                     'output_dilatedICPR_25_6_allValidation', 'output_dilatedICPR_50', 'output_dilatedGRSL',
                     'output_segNet_25', 'output_segnetICPR']  # considered networks

    # read testing data
    print BatchColors.WARNING + 'Reading Test Data...' + BatchColors.ENDC
    test_data, _ = load_images(input_path + 'test/')
    test_diff_data, _ = load_images(input_path + 'test/new_test_diff/')
    test_feat = read_npys(input_test_npy, test_features, template_name='prob_map.npy')
    test_diff_feat = read_npys(input_test_npy, test_features, template_name='prob_map.npy', is_diff=True)

    train_feat_arr = train_feat.reshape(len(train_feat) * len(train_feat[0]) * len(train_feat[0][0]),
                                        len(train_feat[0][0][0]))
    training_label_patches_arr = training_label_patches.flatten()

    clf = svm(train_feat_arr, training_label_patches_arr)
    pred_arr = prediction(clf, test_feat)
    pred_diff_arr = prediction(clf, test_diff_feat)

    save_map(output_path, pred_arr, test_data)
    save_map(output_path + 'test_diff/', pred_diff_arr, test_diff_data)


if __name__ == "__main__":
    main()
