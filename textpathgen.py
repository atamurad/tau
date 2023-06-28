import numpy as np
import scipy
import skimage.io
from shapely import LinearRing, Polygon
from skimage import measure, transform
from skimage.draw import line
from skimage.morphology import medial_axis
from skimage.color import rgb2gray


class DefaultParams:
    def __init__(self):
        # input file name
        self.input_file = None
        self.input = None

        # Text path/shape generation

        # skeleton to outside contour distance threshold
        self.skel_dist_threshold = 10.0
        # width of the skeleton
        self.skel_width = 2.0
        # spacing between points when dividing contours
        self.dtw_spacing = 10.0

        # Text to fill shapes with
        self.text = "quickbrownfoxjumpedoverthelazydog".upper() * 5  # repeat5x
        self.font = "./fonts/Silkscreen-Bold.ttf"
        self.font_size = 16

        # output params
        self.output = None


class TextPathGen:

    def __init__(self, params):
        self.params = params
        # read input image
        print(f"Opening input image {self.params.input_file}")
        self.input = rgb2gray(skimage.io.imread(self.params.input_file))
        params.input = self.input

    def find_skeleton_contours(self, blobs):
        """
        Input: binary image
        Output: contours of skeletons
        """
        # Compute the medial axis (skeleton) and the distance transform
        skel, distance = medial_axis(blobs, return_distance=True)
        # Distance to the background for pixels of the skeleton
        dist_on_skel = distance * skel
        skeleton = dist_on_skel > self.params.skel_dist_threshold
        # Apply diamond filter to thicken pixels
        diamond = skimage.morphology.diamond(self.params.skel_width)
        skeleton = scipy.ndimage.convolve(skeleton, diamond)
        # find contours of skeleton
        return skeleton, measure.find_contours(skeleton)

    def calc_dtw(self, lr1, lr2):
        """
        Run DTW to align inner and outer contours
        Input: lr1 and lr2 - LinearRing (list of points)
        Output: List of matched point pairs
        """

        spaces1 = np.linspace(0.0, 1.0, int(lr1.length/self.params.dtw_spacing))[:-1]
        spaces2 = np.linspace(0.0, 1.0, int(lr2.length/self.params.dtw_spacing))[:-1]

        steps1 = len(spaces1)
        steps2 = len(spaces2)

        dtw = np.ones([steps1 + 1, steps2 + 1]) * 1000000
        dtwi = np.zeros([steps1 + 1, steps2 + 1])

        points1 = []
        points2 = []

        for t in spaces1:
            points1.append(lr1.interpolate(t, normalized=True))
        for t in spaces2:
            points2.append(lr2.interpolate(t, normalized=True))

        # find alignment
        p1 = points1[0]
        offset = np.argmin([p1.distance(p2) for p2 in points2])
        spaces2 = np.roll(spaces2, -offset)
        points2 = np.roll(points2, -offset)

        dtw[0, 0] = 0
        for i in range(1, steps1 + 1):
            for j in range(1, steps2 + 1):
                cost = points1[i - 1].distance(points2[j - 1])
                opts = np.array([dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1]])
                dtw[i, j] = cost + np.min(opts)
                dtwi[i, j] = np.argmin(opts)

        # traceback
        i = steps1
        j = steps2

        steps = []
        steps.append((points1[i - 1], points2[j - 1]))
        while i >= 1:
            if dtwi[i, j] == 0:
                i -= 1
            elif dtwi[i, j] == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
            steps.append((points1[i - 1], points2[j - 1]))
        return reversed(steps)

    def fill_contours(self, outer, inner, clockwise=True):
        lr1 = LinearRing(outer)
        lr2 = LinearRing(inner)

        renderer = self.params.renderer

        print("Blob:")
        print(f"Outer length: {lr1.length}")
        print(f"Inner length: {lr2.length}")

        path_upper = []
        path_lower = []

        dtwsteps = self.calc_dtw(lr1, lr2)
        for (p1, p2) in dtwsteps:
            path_upper.append(p1)
            path_lower.append(p2)

        if not clockwise:
            path_upper = list(reversed(path_upper))
            path_lower = list(reversed(path_lower))

        renderer.rewind()
        for i in range(1, len(path_upper)):

            step_h = renderer.height
            step_w = max(
                path_lower[i].distance(path_lower[i - 1]),
                path_upper[i].distance(path_upper[i - 1]),
            )
            # correct for ratio
            step_w *= step_h/max(
                    path_lower[i].distance(path_upper[i]),
                    path_lower[i-1].distance(path_upper[i-1]))

            src = [(0, 0), (step_w, 0), (step_w, step_h), (0, step_h)]

            dst = [
                (path_upper[i - 1].y, path_upper[i - 1].x),
                (path_upper[i].y, path_upper[i].x),
                (path_lower[i].y, path_lower[i].x),
                (path_lower[i - 1].y, path_lower[i - 1].x),
            ]

            # check if current advance is doable?
            clockwise_check = True
            for j in range(4):
                v1 = (dst[j - 1][0] - dst[j - 2][0], dst[j - 1][1] - dst[j - 2][1])
                v2 = (dst[j][0] - dst[j - 1][0], dst[j][1] - dst[j - 1][1])
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                if cross < 0:
                    clockwise_check = False
                    break
            if not clockwise_check:
                print("not clockwise!!!")
                print(dst)
                continue

            # calculate projection matrix
            tform3 = transform.AffineTransform()
            if tform3.estimate(src, dst):
                renderer.advance(step_w, tform3.params)
                rr, cc = line(int(path_lower[i].x), int(path_lower[i].y), int(path_upper[i].x), int(path_upper[i].y))
                self.dtwlines[rr, cc] = 1.0
            else:
                print("no solution - all points in one line?")
                print(dst)


    def match_contours(self, contours, sk_contours, ax=None):
        used = set()
        used_sk = set()

        for idx_o in range(len(contours)):
            contour = contours[idx_o]

            if ax:
                ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

            p_o = Polygon(contour)

            # find inner skeleton polygon
            max_length = -1
            idx_i = None
            for i in range(len(sk_contours)):
                if i in used_sk:
                    continue
                p_i = Polygon(sk_contours[i])
                if p_o.contains(p_i):
                    if idx_i is None:
                        max_length = p_i.length
                        idx_i = i
                    elif p_i.length > max_length:
                        max_length = p_i.length
                        idx_i = i

            if idx_i is None:
                continue

            self.fill_contours(contour, sk_contours[idx_i])
            used.add(idx_o)
            used_sk.add(idx_i)

        for idx_o in range(len(sk_contours)):
            sk = sk_contours[idx_o]

            if idx_o in used_sk:
                continue

            if ax:
                ax.plot(sk[:, 1], sk[:, 0], color='green', linewidth=1)
            p_o = Polygon(sk)

            # find inner contour with max area..
            max_length = -1
            idx_i = None
            for i in range(len(contours)):
                if i in used:
                    continue
                p_i = Polygon(contours[i])
                if p_o.contains(p_i):
                    if idx_i is None:
                        max_length = p_i.length
                        idx_i = i
                    elif p_i.length > max_length:
                        max_length = p_i.length
                        idx_i = i

            if idx_i is None:
                continue

            self.fill_contours(sk, contours[idx_i], False)
            used.add(idx_i)
            used_sk.add(idx_o)

    def process(self, ax=None):

        self.dtwlines = np.zeros(self.input.shape)

        # find contours of input image
        self.contours = measure.find_contours(self.input)
        self.skeleton, self.sk_contours = self.find_skeleton_contours(self.input)

        self.match_contours(self.contours, self.sk_contours, ax=ax)
