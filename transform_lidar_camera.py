import numpy as np
import cv2
import matplotlib.pyplot as plt
import statistics
import random

class LiDAR2Camera(object):
    def __init__(self, calib_file):
        calibs = self.read_calib_file(calib_file)
        P = calibs["P2"]
        self.P = np.reshape(P, [3, 4])
        # Rigid transform from Velodyne coord to reference camera coord
        V2C = calibs["Tr_velo_to_cam"]
        self.V2C = np.reshape(V2C, [3, 4])
        # Rotation from reference camera coord to rect camera coord
        R0 = calibs["R0_rect"]
        self.R0 = np.reshape(R0, [3, 3])

    def read_calib_file(self, filepath):
        """ Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        """
        data = {}
        with open(filepath, "r") as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def cart2hom(self, pts_3d):
        """
        Cartesian to Homogeneous Coordinates
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_velo_to_image(self, pts_3d_velo):
        '''
        Project from Velodyne frame to Camera Frame
        '''

        R0_homo = np.vstack([self.R0, [0, 0, 0]])
        R0_homo_2 = np.column_stack([R0_homo, [0, 0, 0, 1]])
        p_r0 = np.dot(self.P, R0_homo_2)  # PxR0
        p_r0_rt = np.dot(p_r0, np.vstack((self.V2C, [0, 0, 0, 1])))  # PxROxRT
        pts_3d_homo = np.column_stack([pts_3d_velo, np.ones((pts_3d_velo.shape[0], 1))])
        p_r0_rt_x = np.dot(p_r0_rt, np.transpose(pts_3d_homo))  # PxROxRTxX
        pts_2d = np.transpose(p_r0_rt_x)

        pts_2d[:, 0] /= pts_2d[:, 2]
        pts_2d[:, 1] /= pts_2d[:, 2]
        return pts_2d[:, 0:2]

    def get_lidar_in_image_fov(self, pc_velo, xmin, ymin, xmax, ymax, return_more=False, clip_distance=2.0):
        """ Filter lidar points, keep those in image FOV """
        pts_2d = self.project_velo_to_image(pc_velo)
        fov_inds = (
                (pts_2d[:, 0] < xmax)
                & (pts_2d[:, 0] >= xmin)
                & (pts_2d[:, 1] < ymax)
                & (pts_2d[:, 1] >= ymin)
        )
        fov_inds = fov_inds & (
                    pc_velo[:, 0] > clip_distance)
        imgfov_pc_velo = pc_velo[fov_inds, :]
        if return_more:
            return imgfov_pc_velo, pts_2d, fov_inds
        else:
            return imgfov_pc_velo

    def show_lidar_on_image(self, pc_velo, img):
        """ Project LiDAR points to image """
        imgfov_pc_velo, pts_2d, fov_inds = self.get_lidar_in_image_fov(
            pc_velo, 0, 0, img.shape[1], img.shape[0], True)

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        self.imgfov_pc_velo = imgfov_pc_velo

        self.imgfov_pts_2d = pts_2d[fov_inds, :]

        for i in range(self.imgfov_pts_2d.shape[0]):
            depth = imgfov_pc_velo[i, 0]
            color = cmap[int(510.0 / depth), :]
            cv2.circle(
                img, (int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))), 2,
                color=tuple(color),
                thickness=-1,
            )

        return img

    def rectContains(self,rect, pt, w, h, shrink_factor=0):
        x1 = int(rect[0] * w - rect[2] * w * 0.5 * (1 - shrink_factor))  # center_x - width /2 * shrink_factor
        y1 = int(rect[1] * h - rect[3] * h * 0.5 * (1 - shrink_factor))  # center_y - height /2 * shrink_factor
        x2 = int(rect[0] * w + rect[2] * w * 0.5 * (1 - shrink_factor))  # center_x + width/2 * shrink_factor
        y2 = int(rect[1] * h + rect[3] * h * 0.5 * (1 - shrink_factor))  # center_y + height/2 * shrink_factor

        return x1 < pt[0] < x2 and y1 < pt[1] < y2

    def filter_outliers(self,distances):
        inliers = []
        mu = statistics.mean(distances)
        std = statistics.stdev(distances)
        for x in distances:
            if abs(x - mu) < std:
                # This is an INLIER
                inliers.append(x)
        return inliers

    def get_best_distance(self, distances, technique="closest"):
        if technique == "closest":
            return min(distances)
        elif technique == "average":
            return statistics.mean(distances)
        elif technique == "random":
            return random.choice(distances)
        else:
            return statistics.median(sorted(distances))

    def lidar_camera_fusion(self, pred_bboxes, image):
        img_bis = image.copy()

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
        distances = []
        for box in pred_bboxes:
            distances = []
            for i in range(self.imgfov_pts_2d.shape[0]):
                depth = self.imgfov_pc_velo[i, 0]
                if (self.rectContains(box, self.imgfov_pts_2d[i], image.shape[1], image.shape[0],
                                 shrink_factor=0.20) == True):
                    distances.append(depth)

                    color = cmap[int(510.0 / depth), :]
                    cv2.circle(img_bis,
                               (int(np.round(self.imgfov_pts_2d[i, 0])), int(np.round(self.imgfov_pts_2d[i, 1]))), 2,
                               color=tuple(color), thickness=-1, )
            h, w, _ = img_bis.shape
            if (len(distances) > 2):
                distances = self.filter_outliers(distances)
                best_distance = self.get_best_distance(distances, technique="average")
                cv2.putText(img_bis, '{0:.2f} m'.format(best_distance), (int(box[0] * w), int(box[1] * h)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2, cv2.LINE_AA)

        return img_bis, distances
