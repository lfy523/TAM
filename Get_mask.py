from PIL import Image

import numpy as np
from lang_sam.utils import draw_image
import torch
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import draw_segmentation_masks
import matplotlib as plt

def draw_image(image, masks, boxes, labels, alpha=0.4):
    image = torch.from_numpy(image).permute(2, 0, 1)
    masks_np = np.array(masks)
    if len(boxes) > 0:
        image = draw_bounding_boxes(image, boxes, colors=['red'] * len(boxes), labels=labels, width=2)
    id_dict = torch.full(masks.shape, 0)
    for i in range(len(boxes)):
        matrix = torch.full(masks.shape, False)
        # masks = torch.full(masks.shape,False)
        matrix[:,  int(boxes[i][1]): int(boxes[i][3]), int(boxes[i][0]):int(boxes[i][2])] = True
        matrix = matrix & masks
        id_dict = torch.where(matrix == True, 255, id_dict)
    
    # Image.fromarray(np.uint8(np.array(id_dict[0,:,:]))).save('map.png')
    masks = torch.where(id_dict > 0, True, False)
    

    
    # Image.fromarray(np.array(matrix)).save('matrix.png')

    if len(masks) > 0:
        image = draw_segmentation_masks(image, masks=masks, colors=['cyan'] * len(masks), alpha=alpha)
    return image.numpy().transpose(1, 2, 0)

# color_list = plt.cm.tab10(np.linspace(0, 1, 12))
# print(color_list)

def Get_mask(video_state, model, text_prompt = "tweezers"):
    # print(input_image)
    input_image = video_state["origin_images"][video_state["select_frame_number"]]
    
    image_pil = Image.fromarray(input_image).convert("RGB")
    image_pil.save('image_test.png')
    masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
    boxes = bb_intersection_over_union(boxes)
    # masks = np.array(torch.where(masks == True, 255, 0))
    
    
    labels = [f"{phrase} {logit:.2f}" for phrase, logit in zip(phrases, logits)]
    image_array = np.asarray(image_pil)
    image = draw_image(image_array, masks, boxes, labels)
    image = Image.fromarray(np.uint8(image)).convert("RGB")
    mask_final = masks[0]
    for i in range(masks.shape[0]):
        mask_final = mask_final | masks[i]
    # print(mask_final.shape, mask_final)
    # Image.fromarray(np.array(torch.where(mask_final == True, 255, 0)).astype(np.uint8)).save('mask.png')
    video_state["masks"][video_state["select_frame_number"]] = np.uint8(mask_final)
    video_state["logits"][video_state["select_frame_number"]] = logits
    video_state["painted_images"][video_state["select_frame_number"]] = image
    return video_state

def bb_intersection_over_union(boxes):
    boxes.sort()
    for i in range(1, len(boxes)):
        boxA = [int(x) for x in boxes[i - 1]]
        boxB = [int(x) for x in boxes[i]]

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
        iou = interArea / float(boxAArea + boxBArea - interArea)
        if iou > 0.9:
            if boxAArea >= boxBArea: boxes.pop(i)
            else: boxes.pop(i - 1)
            

    return boxes
# Get_mask('心血管手术.png')
# 
# image = draw_image(image_array, masks, boxes, labels)
# 
# 