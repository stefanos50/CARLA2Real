import torchvision
from torchvision import models
import torch
import numpy as np
from torchvision import transforms
import cv2
from PIL import Image
from epe.autonomous_driving.helper_methods import Helper
from ultralytics import YOLO
import yaml

ad_task_name = 'object_detection' #switch between semantic segmentation and object detection
#with open("../config/carla_config.yaml", 'r') as file:
    #carla_config = yaml.safe_load(file)

#Define and initialize the semantic segmentation model. In this example the
#predefined DeepLabV3 from pytorch is used.
def initialize_model(num_classes, use_pretrained=True):

    if ad_task_name == 'semantic_segmentation':
        model_deeplabv3 = models.segmentation.deeplabv3_resnet50(pretrained=use_pretrained, progress=True)
        model_deeplabv3.aux_classifier = None
        model_deeplabv3.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, num_classes)
        return model_deeplabv3
    elif ad_task_name == "object_detection":
        model = YOLO('..\\checkpoints\\YOLO\\best.pt')
        return model

#Colors for each class of Object Detection bounding boxes
colors = [
    [255, 0, 0],    # Red
    [0, 255, 0],    # Green
    [0, 0, 255],    # Blue
    [255, 255, 0],  # Yellow
    [128, 0, 128],  # Purple
    [255, 165, 0],   # Orange
    [0, 128, 128]   # Teal
]

#Define the palette that will be used
#to colorize the predicted label maps
#before rendering to pygame.
CITYSCAPES_PALETTE_MAP = [
[0, 0, 0],
[128, 64, 128],
[244, 35, 232],
[70, 70, 70],
[102, 102, 156],
[190, 153, 153],
[153, 153, 153],
[250, 170, 30],
[220, 220, 0],
[107, 142, 35],
[152, 251, 152],
[70, 130, 180],
[220, 20, 60],
[255, 0, 0],
[0, 0, 142],
[0, 0, 70],
[0, 60, 100],
[0, 80, 100],
[0, 0, 230],
[119, 11, 32],
[110, 190, 160],
[170, 120, 50],
[55, 90, 80],
[45, 60, 150],
[157, 234, 50],
[81, 0, 81],
[150, 100, 100],
[230, 150, 140],
[180, 165, 180]
]

#Calculating IOU for all classes or a specific class given an image and
#its corresponding ground truth labels. Both pred and target should be
#pytorch tensors. If specific_class is set to -1 then the iou will be
#calculated for all classes.
def ciou(pred, target, n_classes = 3 , specific_class=-1):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  if specific_class == -1:
      for cls in range(0, n_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
        union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
        if union > 0:
            ious.append(float(intersection) / float(max(union, 1)))
  else:
      cls = specific_class
      pred_inds = pred == cls
      target_inds = target == cls
      intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()
      union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
      if union > 0:
          ious.append(float(intersection) / float(max(union, 1)))

  return np.array(ious)

#If any real dataset was used for comparisons, then the Carla ground truth labels
#may need further preprocessing to be compatible.
def make_compatible(label_map):
    old_class_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                         24, 25, 26, 27, 28, 29]
    new_integer_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                       1, 25, 26, 27, 28, 29]

    label_map = np.select([label_map == old_idx for old_idx in old_class_indexes],
                          [new_id for new_id in new_integer_ids],
                          default=label_map)
    return label_map

class ADTask:
    #this function is called when camera_output is set as ad_task
    #here you should load a pretrained model for tasks like
    #semantic segmentation , object detectiion or lane detection
    #that you want to apply and test in real time with an enhanced or carla
    #rgb camera output.
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if ad_task_name == "semantic_segmentation":
            self.num_classes = 30
            self.model = initialize_model(self.num_classes, use_pretrained=True)
            state_dict = torch.load("..\\ad_checkpoints\\Deeplabv3\\best_enhanced_town10.pth", map_location=self.device)
            self.model = self.model.to(self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.sum_iou = 0
            self.counter = 0
        elif ad_task_name == "object_detection":
            self.model = initialize_model(0,use_pretrained=True)
            self.model = self.model.to(self.device)
            self.model.eval()


    #this function gets a carla enhanced or rgb frame and predicts the output frame for the specific task
    #the input is a frame in the form of a numpy array (uint8) and the function should return the predicted frame
    #in the same way (uint8 numpy array). Ground truth labels from the semantic segmentation camera are also
    #given in a form of numpy array for real-time evaluation. World, Camera and Vehicle parameters may be usefull for
    #other kind of tasks like object detection. Data_dict contains all the data from all the active sensors of the
    #ego vehicle.
    def predict_output(self,frame,ground_truth,world,vehicle,camera,data_dict):

        if ad_task_name == "semantic_segmentation":
            transforms_image =  transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            frame_np = frame
            frame = transforms_image(frame)
            frame = frame.unsqueeze(0)

            frame = frame.to(self.device)

            outputs = self.model(frame)["out"]

            _, preds = torch.max(outputs, 1)

            preds = preds.to("cpu")

            preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)

            semantic = np.zeros((preds_np.shape[0], preds_np.shape[1], 3))
            for i in range(len(preds_np)):
                for j in range(len(preds_np[i])):
                    label = int(preds_np[i][j])
                    semantic[i][j][0] = CITYSCAPES_PALETTE_MAP[label][0]
                    semantic[i][j][1] = CITYSCAPES_PALETTE_MAP[label][1]
                    semantic[i][j][2] = CITYSCAPES_PALETTE_MAP[label][2]

            self.sum_iou += ciou(torch.from_numpy(preds_np),torch.from_numpy(make_compatible(ground_truth[:, :, 0])),self.num_classes).mean()
            self.counter += 1
            print("Current IOU: "+str(self.sum_iou/self.counter))

            #Render as a transparent mask on the original frame
            semantic = cv2.addWeighted(frame_np.astype(np.uint8), 0.5, semantic.astype(np.uint8), 0.5, 0)

            return semantic.astype(np.uint8)
        elif ad_task_name == "object_detection":
            frame = np.ascontiguousarray(frame, dtype=np.uint8)

            pil_frame = Image.fromarray(frame.astype(np.uint8))
            results = self.model.predict(pil_frame, save=False, conf=0.5)[0]  # ✅ take first result

            cls_names = ['person', 'rider', 'vehicle', 'truck', 'bus', 'motorcycle', 'bicycle']

            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf.cpu().numpy())
                class_idx = int(box.cls.cpu().numpy())
                class_name = cls_names[class_idx]

                color = colors[int(class_idx) % len(colors)]
                thickness = 2

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return np.array(frame)

            return np.array(frame)



