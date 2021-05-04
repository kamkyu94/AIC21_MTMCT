import os
import cv2

modes = ['train', 'validation']
basic_path = '../../dataset/AIC21_Track3/'

for mode in modes:
    # Data Path
    data_path = basic_path + mode + '/'

    # Extract frame
    scenes = os.listdir(data_path)
    for scene in scenes:
        cams = os.listdir(data_path + scene)
        for cam in cams:
            # Set paths
            cam_path = data_path + scene + '/' + cam + '/'
            video_path, frame_path = cam_path + 'vdo.avi', cam_path + 'frame/'

            # Create frame directory
            if not os.path.exists(frame_path):
                os.mkdir(frame_path)

            # Read video
            video = cv2.VideoCapture(video_path)

            # Save each frame
            f_num = 1
            while 1:
                success, image = video.read()
                if not success:
                    break
                cv2.imwrite(frame_path + '%s_f%04d.jpg' % (cam, f_num), image)
                f_num += 1

            # Print current status
            print('%s_%s Finished' % (scene, cam))
