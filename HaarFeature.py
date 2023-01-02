import numpy as np

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