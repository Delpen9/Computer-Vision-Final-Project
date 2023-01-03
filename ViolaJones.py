import numpy as np
import cv2
import os
from helper_classes import WeakClassifier, VJ_Classifier

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))

# I/O directories
output_dir = fr'{ROOT_DIR}/output'

def split_dataset(X, y, p):
    """Split dataset into training and test sets.

    Let M be the number of images in X, select N random images that will
    compose the training data (see np.random.permutation). The images that
    were not selected (M - N) will be part of the test data. Record the labels
    accordingly.

    Args:
        X (numpy.array): 2D dataset.
        y (numpy.array): 1D array of labels (int).
        p (float): Decimal value that determines the percentage of the data
                   that will be the training data.

    Returns:
        tuple: Four-element tuple containing:
            Xtrain (numpy.array): Training data 2D array.
            ytrain (numpy.array): Training data labels.
            Xtest (numpy.array): Test data test 2D array.
            ytest (numpy.array): Test data labels.
    """
    indices = np.random.permutation(len(X))
    mid_index = int(len(X) * p)
    indices_train = indices[ : mid_index]
    indices_test = indices[mid_index : ]
    
    Xtrain = X[indices_train]
    ytrain = y[indices_train]

    Xtest = X[indices_test]
    ytest = y[indices_test]

    return Xtrain, ytrain, Xtest, ytest
    
class Boosting:
    """Boosting classifier.

    Args:
        X (numpy.array): Data array of flattened images
                         (row:observations, col:features) (float).
        y (numpy.array): Labels array of shape (observations, ).
        num_iterations (int): number of iterations
                              (ie number of weak classifiers).

    Attributes:
        Xtrain (numpy.array): Array of flattened images (float32).
        ytrain (numpy.array): Labels array (float32).
        num_iterations (int): Number of iterations for the boosting loop.
        weakClassifiers (list): List of weak classifiers appended in each
                               iteration.
        alphas (list): List of alpha values, one for each classifier.
        num_obs (int): Number of observations.
        weights (numpy.array): Array of normalized weights, one for each
                               observation.
        eps (float): Error threshold value to indicate whether to update
                     the current weights or stop training.
    """

    def __init__(self, X, y, num_iterations):
        self.Xtrain = np.float32(X)
        self.ytrain = np.float32(y)
        self.num_iterations = num_iterations
        self.weakClassifiers = []
        self.alphas = []
        self.num_obs = X.shape[0]
        self.weights = np.array([1.0 / self.num_obs] * self.num_obs)  # uniform weights
        self.eps = 0.00000001

    def train(self):
        """Implement the for loop shown in the problem set instructions."""
        for i in range(self.num_iterations):
            self.weights = self.weights / np.sum(self.weights)
            self.weakClassifiers.append(WeakClassifier(self.Xtrain, self.ytrain, self.weights.copy()))
            self.weakClassifiers[i].train()
            predictions = np.array([self.weakClassifiers[i].predict(x) for x in self.Xtrain])
            match_output = predictions - self.ytrain
            match_output_indices = np.where(match_output != 0)[0]
            epsilon = np.sum(self.weights[match_output_indices])
            self.alphas.append(0.5 * np.log((1 - epsilon) / epsilon))
            if epsilon > self.eps:
                self.weights = self.weights * np.exp(-1 * self.ytrain * self.alphas[i] * predictions)
            else:
                break

    def evaluate(self):
        """Return the number of correct and incorrect predictions.

        Use the training data (self.Xtrain) to obtain predictions. Compare
        them with the training labels (self.ytrain) and return how many
        where correct and incorrect.

        Returns:
            tuple: two-element tuple containing:
                correct (int): Number of correct predictions.
                incorrect (int): Number of incorrect predictions.
        """
        predictions = self.predict(self.Xtrain)
        accuracy_array = predictions - self.ytrain
        correct = len(np.where(accuracy_array == 0)[0])
        incorrect = len(np.where(accuracy_array != 0)[0])
        return correct, incorrect

    def predict(self, X):
        """Return predictions for a given array of observations.

        Use the alpha values stored in self.aphas and the weak classifiers
        stored in self.weakClassifiers.

        Args:
            X (numpy.array): Array of flattened images (observations).

        Returns:
            numpy.array: Predictions, one for each row in X.
        """
        observations = X.copy()
        predictions = []
        for i in range(len(observations)):
            current_observation_prediction = 0
            for j in range(len(self.alphas)):
                current_observation_prediction += self.alphas[j] * self.weakClassifiers[j].predict(observations[i])
            predictions.append(np.sign(current_observation_prediction))

        predictions = np.array(predictions)
        return predictions


class HaarFeature:
    """Haar-like features.

    Args:
        feat_type (tuple): Feature type {(2, 1), (1, 2), (3, 1), (2, 2)}.
        position (tuple): (row, col) position of the feature's top left corner.
        size (tuple): Feature's (height, width)

    Attributes:
        feat_type (tuple): Feature type.
        position (tuple): Feature's top left corner.
        size (tuple): Feature's width and height.
    """

    def __init__(self, feat_type, position, size):
        self.feat_type = feat_type
        self.position = position
        self.size = size

    def _create_two_horizontal_feature(self, shape):
        """Create a feature of type (2, 1).

        Use int division to obtain half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        ## position = (25, 30);
        ## shape = (50, 100)
        haar_image = np.zeros((shape), dtype = np.uint8)
        haar_image[self.position[0] : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 255
        haar_image[int(self.position[0] + self.size[0] / 2) : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 126
        return haar_image

    def _create_two_vertical_feature(self, shape):
        """Create a feature of type (1, 2).

        Use int division to obtain half the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        ## position = (10, 25);
        ## shape = (50, 150)
        haar_image = np.zeros((shape), dtype = np.uint8)
        haar_image[self.position[0] : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 255
        haar_image[self.position[0] : self.position[0] + self.size[0], int(self.position[1] + self.size[1] / 2) : self.position[1] + self.size[1]] = 126
        return haar_image

    def _create_three_horizontal_feature(self, shape):
        """Create a feature of type (3, 1).

        Use int division to obtain a third of the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        ## position = (50, 50);
        ## shape = (100, 50)
        haar_image = np.zeros((shape), dtype = np.uint8)
        haar_image[self.position[0] : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 255
        haar_image[int(self.position[0] + self.size[0] / 3) : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 126
        haar_image[int(self.position[0] + 2 * self.size[0] / 3) : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 255
        return haar_image

    def _create_three_vertical_feature(self, shape):
        """Create a feature of type (1, 3).

        Use int division to obtain a third of the width.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        ## position = (50, 125);
        ## shape = (100, 50)
        haar_image = np.zeros((shape), dtype = np.uint8)
        haar_image[self.position[0] : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 255
        haar_image[self.position[0] : self.position[0] + self.size[0], int(self.position[1] + self.size[1] / 3) : self.position[1] + self.size[1]] = 126
        haar_image[self.position[0] : self.position[0] + self.size[0], int(self.position[1] + 2 * self.size[1] / 3) : self.position[1] + self.size[1]] = 255
        return haar_image

    def _create_four_square_feature(self, shape):
        """Create a feature of type (2, 2).

        Use int division to obtain half the width and half the height.

        Args:
            shape (tuple):  Array numpy-style shape (rows, cols).

        Returns:
            numpy.array: Image containing a Haar feature. (uint8).
        """
        ## position = (50, 125);
        ## shape = (100, 50)
        haar_image = np.zeros((shape), dtype = np.uint8)
        haar_image[self.position[0] : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 126
        haar_image[self.position[0] : self.position[0] + self.size[0], int(self.position[1] + self.size[1] / 2) : self.position[1] + self.size[1]] = 255
        haar_image[int(self.position[0] + self.size[0] / 2) : self.position[0] + self.size[0], self.position[1] : self.position[1] + self.size[1]] = 255
        haar_image[int(self.position[0] + self.size[0] / 2) : self.position[0] + self.size[0], int(self.position[1] + self.size[1] / 2) : self.position[1] + self.size[1]] = 126
        
        return haar_image

    def preview(self, shape=(24, 24), filename=None):
        """Return an image with a Haar-like feature of a given type.

        Function that calls feature drawing methods. Each method should
        create an 2D zeros array. Each feature is made of a white area (255)
        and a gray area (126).

        The drawing methods use the class attributes position and size.
        Keep in mind these are in (row, col) and (height, width) format.

        Args:
            shape (tuple): Array numpy-style shape (rows, cols).
                           Defaults to (24, 24).

        Returns:
            numpy.array: Array containing a Haar feature (float or uint8).
        """

        if self.feat_type == (2, 1):  # two_horizontal
            X = self._create_two_horizontal_feature(shape)

        if self.feat_type == (1, 2):  # two_vertical
            X = self._create_two_vertical_feature(shape)

        if self.feat_type == (3, 1):  # three_horizontal
            X = self._create_three_horizontal_feature(shape)

        if self.feat_type == (1, 3):  # three_vertical
            X = self._create_three_vertical_feature(shape)

        if self.feat_type == (2, 2):  # four_square
            X = self._create_four_square_feature(shape)

        if filename is None:
            cv2.imwrite(fr'{output_dir}/{self.feat_type}_feature.png', X)

        else:
            cv2.imwrite(fr'{output_dir}/{filename}.png', X)

        return X

    def evaluate(self, ii):
        """Evaluates a feature's score on a given integral image.

        Calculate the score of a feature defined by the self.feat_type.
        Using the integral image and the sum / subtraction of rectangles to
        obtain a feature's value. Add the feature's white area value and
        subtract the gray area.

        For example, on a feature of type (2, 1):
        score = sum of pixels in the white area - sum of pixels in the gray area

        Keep in mind you will need to use the rectangle sum / subtraction
        method and not numpy.sum(). This will make this process faster and
        will be useful in the ViolaJones algorithm.

        Args:
            ii (numpy.array): Integral Image.

        Returns:
            float: Score value.
        """
        ## Two horizontal
        if self.feat_type == (2, 1):
            # A = np.sum(test_image[pos[0]:pos[0] + size[0] // 2, pos[1]:pos[1] + size[1]])
            # B = np.sum(test_image[pos[0] + size[0] // 2:pos[0] + size[0], pos[1]:pos[1] + size[1]])

            bottom_right = ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] - 1, self.position[1] - 1]
            A = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] - 1]
            B = bottom_right + bottom_left + top_right + top_left            
            
            ref = A - B
            
        ## Two vertical
        elif self.feat_type == (1, 2):
            # A = np.sum(test_image[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1] // 2])
            # B = np.sum(test_image[pos[0]:pos[0] + size[0], pos[1] + size[1] // 2:pos[1] + size[1]])

            bottom_right = ii[self.position[0] + self.size[0] - 1, int(self.position[1] + self.size[1] // 2) - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            top_left = ii[self.position[0] - 1, self.position[1] - 1]
            A = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            B = bottom_right + bottom_left + top_right + top_left            
            
            ref = A - B

        ## Three horizontal
        elif self.feat_type == (3, 1):
            # A = np.sum(test_image[pos[0]:pos[0] + size[0] // 3, pos[1]:pos[1] + size[1]])
            # B = np.sum(test_image[pos[0] + size[0] // 3:pos[0] + 2 * size[0] // 3, pos[1]:pos[1] + size[1]])
            # C = np.sum(test_image[pos[0] + 2 * size[0] // 3:pos[0] + size[0], pos[1]:pos[1] + size[1]])

            bottom_right = ii[self.position[0] + self.size[0] // 3 - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] // 3 - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] - 1, self.position[1] - 1]
            A = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + 2 * self.size[0] // 3 - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + 2 * self.size[0] // 3 - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] + self.size[0] // 3 - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] + self.size[0] // 3 - 1, self.position[1] - 1]
            B = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] + 2 * self.size[0] // 3 - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] + 2 * self.size[0] // 3 - 1, self.position[1] - 1]
            C = bottom_right + bottom_left + top_right + top_left

            ref = A - B + C

        ## Three vertical
        elif self.feat_type == (1, 3):
            # A = np.sum(test_image[pos[0]:pos[0] + size[0], pos[1]:pos[1] + size[1] // 3])
            # B = np.sum(test_image[pos[0]:pos[0] + size[0], pos[1] + size[1] // 3:pos[1] + 2 * size[1] // 3])
            # C = np.sum(test_image[pos[0]:pos[0] + size[0], pos[1] + 2 * size[1] // 3:pos[1] + size[1]])

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] // 3 - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] // 3 - 1]
            top_left = ii[self.position[0] - 1, self.position[1] - 1]
            A = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + 2 * self.size[1] // 3 - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] // 3 - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + 2 * self.size[1] // 3 - 1]
            top_left = ii[self.position[0] - 1, self.position[1] + self.size[1] // 3 - 1]
            B = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] + 2 * self.size[1] // 3 - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] - 1, self.position[1] + 2 * self.size[1] // 3 - 1]
            C = bottom_right + bottom_left + top_right + top_left

            ref = A - B + C
        
        ## Four Square
        elif self.feat_type == (2, 2):
            # A = np.sum(test_image[pos[0]:pos[0] + size[0] // 2, pos[1]:pos[1] + size[1] // 2])
            # B = np.sum(test_image[pos[0]:pos[0] + size[0] // 2, pos[1] + size[1] // 2:pos[1] + size[1]])
            # C = np.sum(test_image[pos[0] + size[0] // 2:pos[0] + size[0], pos[1]:pos[1] + size[1] // 2])
            # D = np.sum(test_image[pos[0] + size[0] // 2:pos[0] + size[0], pos[1] + size[1] // 2:pos[1] + size[1]])

            bottom_right = ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] // 2 - 1]
            bottom_left = -ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            top_left = ii[self.position[0] - 1, self.position[1] - 1]
            A = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] // 2 - 1]
            top_right = -ii[self.position[0] - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            B = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] - 1]
            top_right = -ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] // 2 - 1]
            top_left = ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] - 1]
            C = bottom_right + bottom_left + top_right + top_left

            bottom_right = ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] - 1]
            bottom_left = -ii[self.position[0] + self.size[0] - 1, self.position[1] + self.size[1] // 2 - 1]
            top_right = -ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] - 1]
            top_left = ii[self.position[0] + self.size[0] // 2 - 1, self.position[1] + self.size[1] // 2 - 1]
            D = bottom_right + bottom_left + top_right + top_left

            ref = -A + B + C - D

        return ref

def convert_images_to_integral_images(images):
    """Convert a list of grayscale images to integral images.

    Args:
        images (list): List of grayscale images (uint8 or float).

    Returns:
        (list): List of integral images.
    """
    integral_images = []
    for image in images:
        integral_image = np.cumsum(np.cumsum(image, axis = 1), axis = 0)
        integral_images.append(integral_image)

    return integral_images
