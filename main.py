import numpy as np
import cv2
import os
import xml.etree.ElementTree as et
import pickle
from helper_classes import WeakClassifier
from ViolaJones import convert_images_to_integral_images, Boosting, split_dataset
import time
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# I/O Directories
xml_input_dir = fr'{ROOT_DIR}/Sign_Images/annotations/'
img_input_dir = fr'{ROOT_DIR}/Sign_Images/images/'
output_dir = fr'{ROOT_DIR}/output/'

imgs_list = [f for f in os.listdir(img_input_dir) if f[0] != '.' and f.endswith('.png')]
xml_list = [f for f in os.listdir(xml_input_dir) if f[0] != '.' and f.endswith('.xml')]

def Create_Positive_Images():
    output_counter = 0
    for img, xml in zip(imgs_list, xml_list):
        tree = et.parse(os.path.join(xml_input_dir, xml))
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                xmin = child[5][0].text
                ymin = child[5][1].text
                xmax = child[5][2].text
                ymax = child[5][3].text

                image = cv2.imread(os.path.join(img_input_dir, img))
                image = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]
                image = cv2.resize(image, (200, 200))
                cv2.imwrite(fr'{output_dir}/positive/positive_image_{output_counter}.png', image)
                output_counter += 1

def Create_Negative_Images():
    output_counter = 0
    for img, xml in zip(imgs_list, xml_list):
        tree = et.parse(os.path.join(xml_input_dir, xml))
        root = tree.getroot()
        param_dict = {
            'xmin': np.array([]), 'ymin': np.array([]), 'xmax': np.array([]), 'ymax': np.array([])
        }
        for child in root:
            if child.tag == 'object':
                xmin = child[5][0].text
                param_dict['xmin'] = np.append(param_dict['xmin'], int(xmin))
                ymin = child[5][1].text
                param_dict['ymin'] = np.append(param_dict['ymin'], int(ymin))
                xmax = child[5][2].text
                param_dict['xmax'] = np.append(param_dict['xmax'], int(xmax))
                ymax = child[5][3].text
                param_dict['ymax'] = np.append(param_dict['ymax'], int(ymax))
        
        xmin = np.min(param_dict['xmin'])
        ymin = np.min(param_dict['ymin'])
        xmax = np.max(param_dict['xmax'])
        ymax = np.max(param_dict['ymax'])

        image = cv2.imread(os.path.join(img_input_dir, img))

        try:
            image_one = image[int(0) : int(ymin), int(0) : int(xmin)]
            image_one = cv2.resize(image_one, (200, 200))
            cv2.imwrite(fr'{output_dir}/negative/negative_image_{output_counter}.png', image_one)
            output_counter += 1
        except:
            pass
        
        try:
            image_two = image[int(0) : int(ymin), int(xmax) : ]
            image_two = cv2.resize(image_two, (200, 200))
            cv2.imwrite(fr'{output_dir}/negative/negative_image_{output_counter}.png', image_two)
            output_counter += 1
        except:
            pass
        
        try:
            image_three = image[int(ymax) : , 0 : int(xmin)]
            image_three = cv2.resize(image_three, (200, 200))
            cv2.imwrite(fr'{output_dir}/negative/negative_image_{output_counter}.png', image_three)
            output_counter += 1
        except:
            pass
        
        try:
            image_four = image[int(ymax) : , int(xmax) : ]
            image_four = cv2.resize(image_four, (200, 200))
            cv2.imwrite(fr'{output_dir}/negative/negative_image_{output_counter}.png', image_four)
            output_counter += 1
        except:
            pass

def Create_Positive_Images_By_Class():
    output_counter = 0
    for img, xml in zip(imgs_list, xml_list):
        tree = et.parse(os.path.join(xml_input_dir, xml))
        root = tree.getroot()
        for child in root:
            if child.tag == 'object':
                label = child[0].text
                xmin = child[5][0].text
                ymin = child[5][1].text
                xmax = child[5][2].text
                ymax = child[5][3].text

                image = cv2.imread(os.path.join(img_input_dir, img))
                image = image[int(ymin) : int(ymax), int(xmin) : int(xmax)]
                image = cv2.resize(image, (200, 200))
                cv2.imwrite(fr'{output_dir}/positive_multiclass/{label}_{output_counter}.png', image)
                output_counter += 1


def Boost_Classifier_Multiclass_Stage_1(save_pickle):
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive/', f)) for f in os.listdir(fr'{output_dir}/positive') if f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])

    neg = [cv2.imread(os.path.join(fr'{output_dir}/negative/', f)) for f in os.listdir(fr'{output_dir}/negative') if f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])
    neg = np.concatenate((neg, neg), axis = 0)

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    p = 0.9
    Xtrain, ytrain, Xtest, ytest = split_dataset(images, labels, p)

    num_iter = 100

    boost = Boosting(Xtrain, ytrain, num_iter)
    boost.train()

    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))

    y_pred = boost.predict(Xtest)

    acc_list = y_pred - ytest
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(ytest) * 100
    print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))

    positive_idx = np.where(y_pred == 1)
    negative_idx = np.where(y_pred == -1)

    acc_list = y_pred[positive_idx] - ytest[positive_idx]
    true_positive_rate = len(np.where(acc_list == 0)[0]) / len(ytest[positive_idx]) * 100
    print('(Boosting) True positive rate {0:.2f}%'.format(true_positive_rate))

    false_negative_rate = 100 - true_positive_rate
    print('(Boosting) False Negative rate {0:.2f}%'.format(false_negative_rate))

    acc_list = y_pred[negative_idx] - ytest[negative_idx]
    true_negative_rate = len(np.where(acc_list == 0)[0]) / len(ytest[negative_idx]) * 100
    print('(Boosting) True Negative rate {0:.2f}%'.format(true_negative_rate))

    false_positive_rate = 100 - true_negative_rate
    print('(Boosting) False positive rate {0:.2f}%'.format(false_positive_rate))

    if save_pickle[1] == True:
        model_no = save_pickle[0]
        with open(fr'{model_no}_stage_1.pickle', 'wb') as handle:
            pickle.dump(boost, handle, protocol = pickle.HIGHEST_PROTOCOL)

def Boost_Classifier_Multiclass_Stage_2(save_pickle):
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'crosswalk' in f and f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])

    neg = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if ('speedlimit' in f or 'stop' in f or 'trafficlight' in f) and f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    ## Increase representation of pos so that overfitting does not occur
    pos = np.concatenate((pos, pos, pos, pos, pos), axis = 0)

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    p = 0.9
    Xtrain, ytrain, Xtest, ytest = split_dataset(images, labels, p)

    num_iter = 100

    boost = Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))

    y_pred = boost.predict(Xtest)

    acc_list = y_pred - ytest
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(ytest) * 100
    print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))

    positive_idx = np.where(y_pred == 1)
    negative_idx = np.where(y_pred == -1)

    acc_list = y_pred[positive_idx] - ytest[positive_idx]
    true_positive_rate = len(np.where(acc_list == 0)[0]) / len(ytest[positive_idx]) * 100
    print('(Boosting) True positive rate {0:.2f}%'.format(true_positive_rate))

    false_negative_rate = 100 - true_positive_rate
    print('(Boosting) False Negative rate {0:.2f}%'.format(false_negative_rate))

    acc_list = y_pred[negative_idx] - ytest[negative_idx]
    true_negative_rate = len(np.where(acc_list == 0)[0]) / len(ytest[negative_idx]) * 100
    print('(Boosting) True Negative rate {0:.2f}%'.format(true_negative_rate))

    false_positive_rate = 100 - true_negative_rate
    print('(Boosting) False positive rate {0:.2f}%'.format(false_positive_rate))

    if save_pickle[1] == True:
        model_no = save_pickle[0]
        with open(fr'{model_no}_stage_2.pickle', 'wb') as handle:
            pickle.dump(boost, handle, protocol = pickle.HIGHEST_PROTOCOL)

def Boost_Classifier_Multiclass_Stage_3(save_pickle):
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'speedlimit' in f and f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])
    
    neg = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if ('stop' in f or 'trafficlight' in f) and f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    ## Increase representation of pos so that overfitting does not occur
    neg = np.concatenate((neg, neg, neg), axis = 0)

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    p = 0.9
    Xtrain, ytrain, Xtest, ytest = split_dataset(images, labels, p)

    num_iter = 100

    boost = Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))

    y_pred = boost.predict(Xtest)

    acc_list = y_pred - ytest
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(ytest) * 100
    print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))

    positive_idx = np.where(y_pred == 1)
    negative_idx = np.where(y_pred == -1)

    acc_list = y_pred[positive_idx] - ytest[positive_idx]
    true_positive_rate = len(np.where(acc_list == 0)[0]) / len(ytest[positive_idx]) * 100
    print('(Boosting) True positive rate {0:.2f}%'.format(true_positive_rate))
    false_negative_rate = 100 - true_positive_rate
    print('(Boosting) False Negative rate {0:.2f}%'.format(false_negative_rate))

    acc_list = y_pred[negative_idx] - ytest[negative_idx]
    true_negative_rate = len(np.where(acc_list == 0)[0]) / len(ytest[negative_idx]) * 100
    print('(Boosting) True Negative rate {0:.2f}%'.format(true_negative_rate))
    false_positive_rate = 100 - true_negative_rate
    print('(Boosting) False positive rate {0:.2f}%'.format(false_positive_rate))

    if save_pickle[1] == True:
        model_no = save_pickle[0]
        with open(fr'{model_no}_stage_3.pickle', 'wb') as handle:
            pickle.dump(boost, handle, protocol = pickle.HIGHEST_PROTOCOL)

def Boost_Classifier_Multiclass_Stage_4(save_pickle):
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'stop' in f and f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])
    
    neg = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'trafficlight' in f and f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    ## Increase representation of pos so that overfitting does not occur
    pos = np.concatenate((pos, pos), axis = 0)

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    p = 0.9
    Xtrain, ytrain, Xtest, ytest = split_dataset(images, labels, p)

    num_iter = 100

    boost = Boosting(Xtrain, ytrain, num_iter)
    boost.train()
    good, bad = boost.evaluate()
    boost_accuracy = 100 * float(good) / (good + bad)
    print('(Boosting) Training accuracy {0:.2f}%'.format(boost_accuracy))

    y_pred = boost.predict(Xtest)

    acc_list = y_pred - ytest
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(ytest) * 100
    print('(Boosting) Testing accuracy {0:.2f}%'.format(boost_accuracy))

    positive_idx = np.where(y_pred == 1)
    negative_idx = np.where(y_pred == -1)

    acc_list = y_pred[positive_idx] - ytest[positive_idx]
    true_positive_rate = len(np.where(acc_list == 0)[0]) / len(ytest[positive_idx]) * 100
    print('(Boosting) True positive rate {0:.2f}%'.format(true_positive_rate))
    false_negative_rate = 100 - true_positive_rate
    print('(Boosting) False Negative rate {0:.2f}%'.format(false_negative_rate))

    acc_list = y_pred[negative_idx] - ytest[negative_idx]
    true_negative_rate = len(np.where(acc_list == 0)[0]) / len(ytest[negative_idx]) * 100
    print('(Boosting) True Negative rate {0:.2f}%'.format(true_negative_rate))
    false_positive_rate = 100 - true_negative_rate
    print('(Boosting) False positive rate {0:.2f}%'.format(false_positive_rate))

    if save_pickle[1] == True:
        model_no = save_pickle[0]
        with open(fr'{model_no}_stage_4.pickle', 'wb') as handle:
            pickle.dump(boost, handle, protocol = pickle.HIGHEST_PROTOCOL)

def Stage_1_Ensemble():
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive/', f)) for f in os.listdir(fr'{output_dir}/positive') if f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])

    neg = [cv2.imread(os.path.join(fr'{output_dir}/negative/', f)) for f in os.listdir(fr'{output_dir}/negative') if f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    with open('1_stage_1.pickle', 'rb') as handle:
        model_1 = pickle.load(handle)

    with open('2_stage_1.pickle', 'rb') as handle:
        model_2 = pickle.load(handle)
    
    with open('3_stage_1.pickle', 'rb') as handle:
        model_3 = pickle.load(handle)

    start_time = time.time()
    y_pred_model_1 = model_1.predict(images)
    end_time = time.time()
    total_time = (end_time - start_time) / len(images)
    print(fr'A single image is processed in {total_time} seconds.')

    y_pred_model_2 = model_2.predict(images)
    y_pred_model_3 = model_3.predict(images)

    y_pred = np.array([np.median(np.array([pred_1, pred_2, pred_3])) for pred_1, pred_2, pred_3 in zip(y_pred_model_1, y_pred_model_2, y_pred_model_3)])

    acc_list = y_pred_model_1 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 1 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_2 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 2 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_3 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 3 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for ensemble {0:.2f}%'.format(boost_accuracy))

def Stage_2_Ensemble():
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'crosswalk' in f and f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])

    neg = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if ('speedlimit' in f or 'stop' in f or 'trafficlight' in f) and f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    with open('1_stage_2.pickle', 'rb') as handle:
        model_1 = pickle.load(handle)

    with open('2_stage_2.pickle', 'rb') as handle:
        model_2 = pickle.load(handle)
    
    with open('3_stage_2.pickle', 'rb') as handle:
        model_3 = pickle.load(handle)

    start_time = time.time()
    y_pred_model_1 = model_1.predict(images)
    end_time = time.time()
    total_time = (end_time - start_time) / len(images)
    print(fr'A single image is processed in {total_time} seconds.')
    y_pred_model_2 = model_2.predict(images)
    y_pred_model_3 = model_3.predict(images)

    y_pred = np.array([np.median(np.array([pred_1, pred_2, pred_3])) for pred_1, pred_2, pred_3 in zip(y_pred_model_1, y_pred_model_2, y_pred_model_3)])

    acc_list = y_pred_model_1 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 1 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_2 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 2 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_3 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 3 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for ensemble {0:.2f}%'.format(boost_accuracy))

def Stage_3_Ensemble():
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'speedlimit' in f and f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])
    
    neg = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if ('stop' in f or 'trafficlight' in f) and f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    with open('1_stage_3.pickle', 'rb') as handle:
        model_1 = pickle.load(handle)

    with open('2_stage_3.pickle', 'rb') as handle:
        model_2 = pickle.load(handle)
    
    with open('3_stage_3.pickle', 'rb') as handle:
        model_3 = pickle.load(handle)

    start_time = time.time()
    y_pred_model_1 = model_1.predict(images)
    end_time = time.time()
    total_time = (end_time - start_time) / len(images)
    print(fr'A single image is processed in {total_time} seconds.')

    y_pred_model_2 = model_2.predict(images)
    y_pred_model_3 = model_3.predict(images)

    y_pred = np.array([np.median(np.array([pred_1, pred_2, pred_3])) for pred_1, pred_2, pred_3 in zip(y_pred_model_1, y_pred_model_2, y_pred_model_3)])

    acc_list = y_pred_model_1 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 1 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_2 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 2 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_3 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 3 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for ensemble {0:.2f}%'.format(boost_accuracy))

def Stage_4_Ensemble():
    pos = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'stop' in f and f[0] != '.' and f.endswith('.png')]
    pos = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in pos]
    pos = np.array([cv2.resize(x, (24, 24)).flatten() for x in pos])
    
    neg = [cv2.imread(os.path.join(fr'{output_dir}/positive_multiclass/', f)) for f in os.listdir(fr'{output_dir}/positive_multiclass') if 'trafficlight' in f and f[0] != '.' and f.endswith('.png')]
    neg = [cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in neg]
    neg = np.array([cv2.resize(x, (24, 24)).flatten() for x in neg])

    images = np.concatenate((pos, neg), axis = 0)
    labels = np.concatenate((np.array(len(pos) * [1]), np.array(len(neg) * [-1])), axis = 0)

    with open('1_stage_4.pickle', 'rb') as handle:
        model_1 = pickle.load(handle)

    with open('2_stage_4.pickle', 'rb') as handle:
        model_2 = pickle.load(handle)
    
    with open('3_stage_4.pickle', 'rb') as handle:
        model_3 = pickle.load(handle)

    start_time = time.time()
    y_pred_model_1 = model_1.predict(images)
    end_time = time.time()
    total_time = (end_time - start_time) / len(images)
    print(fr'A single image is processed in {total_time} seconds.')
    
    y_pred_model_2 = model_2.predict(images)
    y_pred_model_3 = model_3.predict(images)

    y_pred = np.array([np.median(np.array([pred_1, pred_2, pred_3])) for pred_1, pred_2, pred_3 in zip(y_pred_model_1, y_pred_model_2, y_pred_model_3)])

    acc_list = y_pred_model_1 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 1 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_2 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 2 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred_model_3 - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for model 3 {0:.2f}%'.format(boost_accuracy))
    
    acc_list = y_pred - labels
    boost_accuracy = len(np.where(acc_list == 0)[0]) / len(labels) * 100
    print('(Boosting) Overall accuracy for ensemble {0:.2f}%'.format(boost_accuracy))

def Predict_from_Ensemble(image, model_1, model_2, model_3):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.array([cv2.resize(image, (24, 24)).flatten()])

    y_pred_model_1 = model_1.predict(image)
    y_pred_model_2 = model_2.predict(image)
    y_pred_model_3 = model_3.predict(image)

    y_pred = np.array([np.median(np.array([pred_1, pred_2, pred_3])) for pred_1, pred_2, pred_3 in zip(y_pred_model_1, y_pred_model_2, y_pred_model_3)])
    return y_pred

def get_clusters(cluster_image, n_clusters):
    white_indices = np.where(cluster_image == 255)
    white_indices = np.array([[row, col] for row, col in zip(white_indices[0], white_indices[1])])

    k_means = KMeans(n_clusters)
    k_means.fit(white_indices)
    identified_clusters = k_means.fit_predict(white_indices)

    new_cluster_image = np.zeros(
        (cluster_image.shape[0], cluster_image.shape[1], 3),
        dtype = np.uint8
    )

    colors = [(255, 0, 0), (0, 0, 255), (0, 128, 0)]
    for index, cluster in zip(white_indices, identified_clusters):
        new_cluster_image = cv2.circle(new_cluster_image, (index[1], index[0]), 2, colors[cluster])

    return new_cluster_image, white_indices, identified_clusters

def color_based_classification(image):
    sign_types = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']

    ## Remember: Image is BGR
    colored_image_flattened = image.reshape(image.shape[0] * image.shape[1], 3)

    yellow = 0
    blue = 0
    red = 0
    black = 0
    white = 0
    green = 0
    for pixel in colored_image_flattened:
        if pixel[1] > pixel[0] * 1.2 and pixel[2] > pixel[0] * 1.2: ## Yellow
            yellow += 1
        if pixel[0] > pixel[1] * 1.2 and pixel[0] > pixel[2] * 1.2: ## Blue
            blue += 1
        if pixel[2] > pixel[1] * 1.2 and pixel[2] > pixel[0] * 1.2: ## Red
            red += 1
        if pixel[1] > pixel[2] * 1.3 and pixel[1] > pixel[0] * 1.3: ## Green
            green += 1
        if np.mean(pixel) < 50: ## Black
            black += 1
        if np.mean(pixel) > 180: ## White
            white += 1

    total_pixels = image.shape[0] * image.shape[1]

    yellow /= total_pixels
    blue /= total_pixels
    red /= total_pixels
    black /= total_pixels
    white /= total_pixels
    green /= total_pixels

    ## High Red Presence: Stop
    if red > 0.4:
        return sign_types[2]

    ## High Yellow or Black Presence (with tinge of red): Traffic Light
    elif (black > 0.5 and 0 < red < 0.25) or yellow > 0.15 or green > 0.2:
        return sign_types[3]

    ## High Blue Presence: Crosswalk
    elif blue > 0.15:
        return sign_types[0]

    ## High White Presence: Speed Limit
    elif white > 0.4:
        return sign_types[1]

    else:
        return 'stop'

def classify(image, ensembles):
    sign_types = ['crosswalk', 'speedlimit', 'stop', 'trafficlight']
    image_copy = image.copy()

    y_pred = Predict_from_Ensemble(image_copy, ensembles[0], ensembles[1], ensembles[2])
    if y_pred == 1:
        classification = 0
    else:
        y_pred = Predict_from_Ensemble(image_copy, ensembles[3], ensembles[4], ensembles[5])
        if y_pred == 1:
            classification = 1
        else:
            y_pred = Predict_from_Ensemble(image_copy, ensembles[6], ensembles[7], ensembles[8])
            if y_pred == 1:
                classification = 2
            else:
                classification = 3
    
    classification = np.array(classification)
    sign_type = sign_types[stats.mode(classification)[0][0]]
    return sign_type

def draw_bounding_boxes(locations, clusters, image, ensemble_array):
    dictionary = {}
    temp_image = image.copy()
    bounded_image = image.copy()

    all_unique_clusters = set(clusters)
    clusters = np.array(clusters)
    locations = np.array(locations)

    for unique_cluster in all_unique_clusters:
        indices = np.where(clusters == unique_cluster)
        cluster_points = locations[indices]

        image_rows = np.array(cluster_points[:, 0])
        image_cols = np.array(cluster_points[:, 1])

        row_mean, row_std = np.mean(image_rows), np.std(image_rows)
        cut_off = row_std * 2
        row_lower, row_upper = max(0, row_mean - cut_off), min(bounded_image.shape[0], row_mean + cut_off)

        col_mean, col_std = np.mean(image_cols), np.std(image_cols)
        cut_off = col_std * 2
        col_lower, col_upper = max(0, col_mean - cut_off), min(bounded_image.shape[1], col_mean + cut_off)

        bounded_image = cv2.rectangle(bounded_image, (int(col_lower), int(row_lower)), (int(col_upper), int(row_upper)), (0, 0, 255), 5)
        
        image_for_classification = temp_image[int(col_lower) : int(col_upper), int(row_lower) : int(row_upper)]
        # sign_type = classify(image_for_classification, ensemble_array) //deprecated
        sign_type = color_based_classification(image_for_classification)

        font = cv2.FONT_HERSHEY_SIMPLEX
        pos = (
            int(col_lower - 10),
            int(row_lower + 30)
        )
        dictionary[fr'item_{unique_cluster}_{sign_type}'] = (col_lower, row_lower)
        x, y = pos
        font_scale = 1
        font_thickness = 1
        text_color = (255, 255, 255)
        text_color_bg = (0, 0, 0)
        text_size, _ = cv2.getTextSize(sign_type, font, font_scale, font_thickness)
        text_width, text_height = text_size

        bounded_image = cv2.rectangle(bounded_image, pos, (x + text_width, y + text_height), text_color_bg, -1)
        bounded_image = cv2.putText(
            bounded_image,
            sign_type,
            (x, y + text_height + font_scale - 1),
            font, font_scale,
            text_color, font_thickness
        )
    return bounded_image, dictionary

def Sliding_Window(image_name = 'road1', bounding_box_dimension_ratio = (5, 5), increment_ratio = (50, 50), multipliers = (1, 1), n_clusters = 1):
    ## Ensemble 1
    with open('1_stage_1.pickle', 'rb') as handle:
        model_1_ensemble_1 = pickle.load(handle)
    with open('2_stage_1.pickle', 'rb') as handle:
        model_2_ensemble_1 = pickle.load(handle)
    with open('3_stage_1.pickle', 'rb') as handle:
        model_3_ensemble_1 = pickle.load(handle)

    ## Ensemble 2
    with open('1_stage_2.pickle', 'rb') as handle:
        model_1_ensemble_2 = pickle.load(handle)

    with open('2_stage_2.pickle', 'rb') as handle:
        model_2_ensemble_2 = pickle.load(handle)
    
    with open('3_stage_2.pickle', 'rb') as handle:
        model_3_ensemble_2 = pickle.load(handle)

    ## Ensemble 3
    with open('1_stage_3.pickle', 'rb') as handle:
        model_1_ensemble_3 = pickle.load(handle)

    with open('2_stage_3.pickle', 'rb') as handle:
        model_2_ensemble_3 = pickle.load(handle)
    
    with open('3_stage_3.pickle', 'rb') as handle:
        model_3_ensemble_3 = pickle.load(handle)

    ## Ensemble 4
    with open('1_stage_4.pickle', 'rb') as handle:
        model_1_ensemble_4 = pickle.load(handle)

    with open('2_stage_4.pickle', 'rb') as handle:
        model_2_ensemble_4 = pickle.load(handle)
    
    with open('3_stage_4.pickle', 'rb') as handle:
        model_3_ensemble_4 = pickle.load(handle)

    image = np.array(cv2.imread(os.path.join(fr'{img_input_dir}/{image_name}.png')))
    image_copy = image.copy()
    cluster_copy = image.copy()

    bounding_box_dimensions = (int(image.shape[0] // bounding_box_dimension_ratio[0]), int(image.shape[1] // bounding_box_dimension_ratio[1]))
    x_multiplier = multipliers[0] ## Linear effect on execution time
    y_multiplier = multipliers[1] ## Linear effect on execution time
    x_increment = int(image.shape[1] // increment_ratio[0]) * x_multiplier
    y_increment = int(image.shape[0] // increment_ratio[1]) * y_multiplier

    cluster_image = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)

    time_start = time.time()
    for y in range(0, image.shape[0] - bounding_box_dimensions[0], y_increment):
        for x in range(0, image.shape[1] - bounding_box_dimensions[1], x_increment):
            image_window = image[y : y + bounding_box_dimensions[0], x : x + bounding_box_dimensions[1]].copy()
            y_pred = Predict_from_Ensemble(image_window, model_1_ensemble_1, model_2_ensemble_1, model_3_ensemble_1)
            if y_pred == 1:
                image_copy = cv2.circle(image_copy, (x + bounding_box_dimensions[1] // 2, y + bounding_box_dimensions[0] // 2), 2, (255, 255, 255))

                cluster_copy = cv2.circle(cluster_copy, (x + bounding_box_dimensions[1] // 2, y + bounding_box_dimensions[0] // 2), 2, (0, 0, 0))
                cluster_image[y + bounding_box_dimensions[0] // 2, x + bounding_box_dimensions[1] // 2] = 255

                
    time_end = time.time()
    total_time = time_end - time_start
    print(total_time)

    cv2.imwrite(fr'{output_dir}/paper_samples/{image_name}.png', image_copy)

    new_cluster_image, locations, clusters = get_clusters(cluster_image, n_clusters)
    final_clustered_image = np.add(new_cluster_image, cluster_copy)
    cv2.imwrite(fr'{output_dir}/paper_samples/{image_name}_cluster.png', final_clustered_image)

    ensemble_array = [
        model_1_ensemble_2, model_2_ensemble_2, model_3_ensemble_2,
        model_1_ensemble_3, model_2_ensemble_3, model_3_ensemble_3,
        model_1_ensemble_4, model_2_ensemble_4, model_3_ensemble_4
    ]
    try:
        bounded_image, dictionary = draw_bounding_boxes(locations, clusters, image, ensemble_array)
        cv2.imwrite(fr'{output_dir}/paper_samples/{image_name}_bounded.png', bounded_image)
        return dictionary
    except Exception as e:
        print(e)
        pass
        return None

if __name__ == '__main__':
    # Create_Images_Advanced()
    
    # Create_Positive_Images()
    # Create_Positive_Images_By_Class()
    # Create_Negative_Images()
    # Increase_Negative_Images()

    # save_pickle = (1, True) # (1, True) / (2, True) / (3, True)
    # Boost_Classifier_Multiclass_Stage_1(save_pickle)

    # save_pickle = (2, True) # (1, True) / (2, True) / (3, True)
    # Boost_Classifier_Multiclass_Stage_1(save_pickle)

    # save_pickle = (3, True) # (1, True) / (2, True) / (3, True)
    # Boost_Classifier_Multiclass_Stage_1(save_pickle)
    
    # Stage_1_Ensemble()

    # save_pickle = (3, True)
    # Boost_Classifier_Multiclass_Stage_2(save_pickle)
    # Stage_2_Ensemble()

    # save_pickle = (3, True)
    # Boost_Classifier_Multiclass_Stage_3(save_pickle)
    # Stage_3_Ensemble()
    
    # save_pickle = (3, True)
    # Boost_Classifier_Multiclass_Stage_4(save_pickle)
    # Stage_4_Ensemble()

    ## Traffic Lights
    dictionary = Sliding_Window('road0', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road1', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road2', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road3', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road4', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road5', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road6', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road7', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road8', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)
    dictionary = Sliding_Window('road9', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)
    dictionary = Sliding_Window('road10', (5, 5), (50, 50), (1, 1), 1)

    # ## Stop Signs
    dictionary = Sliding_Window('road52', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road59', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road63', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road68', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road85', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road98', (5, 5), (50, 50), (1, 1), 1)

    ## Cross Walks
    dictionary = Sliding_Window('road122', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)
    dictionary = Sliding_Window('road123', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road124', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road125', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road126', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road134', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road135', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road137', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road156', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)

    ## Speed Limit
    dictionary = Sliding_Window('road660', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road666', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road693', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road694', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road726', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)
    dictionary = Sliding_Window('road732', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road733', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road734', (5, 5), (50, 50), (1, 1), 3)
    print(dictionary)
    dictionary = Sliding_Window('road747', (5, 5), (50, 50), (1, 1), 1)
    print(dictionary)
    dictionary = Sliding_Window('road778', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)
    dictionary = Sliding_Window('road779', (5, 5), (50, 50), (1, 1), 2)
    print(dictionary)
