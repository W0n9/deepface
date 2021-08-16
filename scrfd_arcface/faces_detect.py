import argparse
import cv2
import torch
import numpy as np
from mmdet.apis import inference_detector, init_detector


def parse_args():
    parser = argparse.ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('--config', default='D:/face/insightface/detection/scrfd/configs/scrfd/scrfd_500m.py', help='test config file path')
    parser.add_argument("--checkpoint", default='D:/face/insightface/detection/scrfd_500m_model.pth', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cpu', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', default="C:/Users/DELL/Desktop/face_det/320.mp4", help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.7, help='bbox score threshold')
    args = parser.parse_args()
    return args

def faces_detect(img, config, checkpoint, score_thr):
    device = torch.device('cpu')
    model = init_detector(config, checkpoint, device=device)
    result = inference_detector(model, img)
    bboxes = np.vstack(result)

    if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = scores > score_thr
            bboxes = bboxes[inds, :]
    
    detected_faces = []        
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        if bbox_int[2] - bbox_int[0] > 20 and bbox_int[3] - bbox_int[1] > 20:
            detected_face = img[bbox_int[1]:bbox_int[3], bbox_int[0]:bbox_int[2]]
            detected_face = cv2.resize(detected_face, (112, 112))
            detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
            detected_face = np.transpose(detected_face, (2, 0, 1))
            detected_face = torch.from_numpy(detected_face).float()
            detected_face.div_(255).sub_(0.5).div_(0.5)
            detected_faces.append(detected_face)
    
    return detected_faces, bboxes    
            
def main():
    args = parse_args()
    camera = cv2.VideoCapture(args.camera_id)
    # print('Press "Esc", "q" or "Q" to exit.')
    while True:
        ret_val, img = camera.read()

        if img is None:
            break
        
        detected_faces, bboxes = faces_detect(img, config=args.config, checkpoint=args.checkpoint, score_thr=args.score_thr)        
        for i, bbox in enumerate(bboxes):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])   
            cv2.rectangle(img, left_top, right_bottom, (0, 255, 0), thickness=1)
        cv2.imshow('', img)
        
        ch = cv2.waitKey(1)
        if ch == ord('q') or ch == ord('Q'):
            break

    cv2.destroyAllWindows()
    camera.release()


if __name__ == '__main__':
    main()

