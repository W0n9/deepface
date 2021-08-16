import numpy as np
import torch
import torch.nn as nn
import cv2
import os
import pickle
from faces_detect import faces_detect
from inference import inference
from torch.utils.data.dataloader import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder


def main(db_path, config, checkpoint, model_path, model_name, source = 0, score_thr=0.7, metric_thr=30):

    transform_dict = transforms.Compose(
        [transforms.Resize([112, 112]),
        transforms.ToTensor(),
        transforms.Normalize(std=[0.5,0.5,0.5], mean=[0.5,0.5,0.5]),]
    )
    dataset = ImageFolder(root=db_path, transform=transform_dict)
    classes = dataset.classes
    targets = dataset.targets
    dataloader = DataLoader(dataset, shuffle=False, num_workers=4, batch_size=4)
    db_embeddings = []
    for idx, (data, target) in enumerate(dataloader):
        outs = inference(model_path, model_name, data)
        db_embeddings.append(outs)
    db_embeddings = torch.vstack(db_embeddings)
    print(db_embeddings.size())
    print('finish extracting!')
    frame_count = 0
    cap = cv2.VideoCapture(source)
    while True:
        frame_count += 1
        ret, frame = cap.read()

        if frame is None:
            break

        if frame_count % 10 == 0:
            detected_faces, bboxes = faces_detect(frame, config=config, checkpoint=checkpoint, score_thr=score_thr)
            if len(detected_faces) > 0:
                detected_faces = torch.stack(detected_faces)
                embeddings = inference(model_path, model_name, detected_faces)
                # calculate distance metric (cosine similarity)
                normed_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
                normed_db_embeddings = nn.functional.normalize(db_embeddings, p=2, dim=1)
                sim = normed_embeddings@normed_db_embeddings.t()
                pred = torch.max(sim, dim=1)[1]
                dist_metric = torch.max(sim, dim=1)[0]
                pred_targets = [targets[t] for t in pred]
                pred_labels = [classes[c] for c in pred_targets]

                # draw bboxes and labels
                for i, bbox in enumerate(bboxes):
                    bbox_int = bbox.astype(np.int32)
                    left_top = (bbox_int[0], bbox_int[1])
                    right_bottom = (bbox_int[2], bbox_int[3])   
                    cv2.rectangle(frame, left_top, right_bottom, (0, 255, 0), thickness=1)
                    cv2.putText(frame, pred_labels[i]+str(format(dist_metric[i],".2f")), left_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    print(pred_labels[i], format(dist_metric[i],".2f"))
                    
        cv2.imshow('img', cv2.resize(frame, (1600, 900)))
        # cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): #press q to quit
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(db_path="D:/face/database_aligned/", 
        config='D:/face/insightface/detection/scrfd/configs/scrfd/scrfd_500m.py',  
        checkpoint='D:/face/insightface/detection/scrfd_500m_model.pth', 
        model_path="C:/Users/DELL/Downloads/ms1mv3_arcface_r50_fp16_backbone.pth" ,
        model_name='r50', source = "C:/Users/DELL/Desktop/face_det/320.mp4", score_thr=0.7)