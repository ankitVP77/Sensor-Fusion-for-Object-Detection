from transform_lidar_camera import LiDAR2Camera
from detect_obstacles import YoloOD
import glob
import open3d as o3d
import numpy as np
import cv2

def final_pipeline(lidar2cam_obj, yolo_obj, image, point_cloud):
    img = image.copy()

    lidar_img = lidar2cam_obj.show_lidar_on_image(point_cloud[:, :3], image)

    result, pred_bboxes = yolo_obj.run_obstacle_detection(img)

    img_final, _ = lidar2cam_obj.lidar_camera_fusion(pred_bboxes, result)

    return img_final


if __name__ == '__main__':

    image_files = sorted(glob.glob("data/img/*.png"))
    point_files = sorted(glob.glob("data/velodyne/*.pcd"))
    label_files = sorted(glob.glob("data/label/*.txt"))
    calib_files = sorted(glob.glob("data/calib/*.txt"))

    index = 3
    pcd_file = point_files[index]

    lidar2cam = LiDAR2Camera(calib_files[index])
    cloud = o3d.io.read_point_cloud(pcd_file)
    points= np.asarray(cloud.points)
    image = cv2.cvtColor(cv2.imread(image_files[index]), cv2.COLOR_BGR2RGB)
    yolo_obj= YoloOD(tiny_model=False)

    final_result = final_pipeline(lidar2cam_obj=lidar2cam, yolo_obj=yolo_obj,
                                  image=image.copy(), point_cloud=points)

    final_result= cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
    cv2.imshow("Final", final_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()