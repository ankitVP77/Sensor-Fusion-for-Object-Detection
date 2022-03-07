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

    video_images = sorted(glob.glob("data/videos/video_4/images/*.png"))
    video_points = sorted(glob.glob("data/videos/video_4/points/*.pcd"))
    calib_files = sorted(glob.glob("data/calib/*.txt"))

    lidar2cam_video = LiDAR2Camera(calib_files[0])
    yolo_obj= YoloOD(tiny_model=False)

    result_video = []

    for idx, img in enumerate(video_images):
        print(idx)
        image = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        point_cloud = np.asarray(o3d.io.read_point_cloud(video_points[idx]).points)
        result_video.append(final_pipeline(lidar2cam_obj=lidar2cam_video,
                                           yolo_obj= yolo_obj,
                                           image=image,
                                           point_cloud=point_cloud))

    out = cv2.VideoWriter('out_4.mp4', cv2.VideoWriter_fourcc(*'MP4V'), 15, (image.shape[1], image.shape[0]))

    print("Done")

    for i in range(len(result_video)):
        print(i)
        out.write(cv2.cvtColor(result_video[i], cv2.COLOR_BGR2RGB))

    out.release()