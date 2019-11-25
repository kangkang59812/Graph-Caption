import torchvision
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from detection.faster_rcnn import fasterrcnn_resnet50_fpn

import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision import transforms as T
from scipy.misc import imread, imresize
confidence_threshold = 0.7
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])


def select_top_predictions(predictions):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score
    """
    scores = predictions[0]['scores']
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions[0]['scores'] = predictions[0]['scores'][keep]
    predictions[0]['boxes'] = predictions[0]['boxes'][keep]
    predictions[0]['labels'] = predictions[0]['labels'][keep]

    scores = predictions[0]["scores"]
    _, idx = scores.sort(0, descending=True)
    predictions[0]['scores'] = predictions[0]['scores'][idx]
    predictions[0]['boxes'] = predictions[0]['boxes'][idx]
    predictions[0]['labels'] = predictions[0]['labels'][idx]

    return predictions


def imshow(img):
    cv2.imwrite('c.jpg', img)
    # plt.imshow(img[:, :, [2, 1, 0]])
    # plt.axis("off")
    # plt.savefig('test_image.png')


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors


def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image

    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """

    labels = predictions[0]["labels"]
    boxes = predictions[0]['boxes']

    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), (0, 255, 0), 2
        )

    return image

# transform = T.Compose(
#             [
#                 T.ToPILImage(),
#                 Resize(min_size, max_size),
#                 T.ToTensor(),
#                 to_bgr_transform,
#                 normalize_transform,
#             ]
#         )


# img = imread('/home/lkk/code/my_faster/image2.jpg')
# H0, W0 = img.shape[0], img.shape[1]
# img = imresize(img, float(1024) / max(H0, W0))
# H, W = img.shape[0], img.shape[1]
# # swap rgb to bgr. This can't be the best way right? #fail
# r = img[:, :, 0].copy()
# img[:, :, 0] = img[:, :, 2]
# img[:, :, 2] = r


img1 = Image.open('/home/lkk/code/my_faster/image1.jpg')
#result = np.array(img1)[:, :, [2, 1, 0]].copy()
result = cv2.imread('/home/lkk/code/my_faster/image1.jpg')
img1 = np.array(img1)[:, :, [2, 1, 0]].copy()
#img1 = np.array(img1)[:, :, [2, 1, 0]].copy()
# img2=Image.open('/home/lkk/code/my_faster/model/image2.jpg')
# img2=transforms.ToTensor()(img2)

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
# 运行前后，x会在模型内部变形状,hook要注意用新的输入
x = [transforms.ToTensor()(img1)]
predictions = model(x)

prediction = select_top_predictions(predictions)

result = overlay_boxes(result, prediction)
imshow(result)


features = torch.Tensor()
scores = torch.Tensor()
boxes = torch.Tensor()


def hook_features(model, input, output):
    features.resize_as_(output)
    features.copy_(output.data)


def hook_scores_box(model, input, output):
    scores.resize_as_(output[0])
    boxes.resize_as_(output[1])
    scores.copy_(output[0].data)
    boxes.copy_(output[1].data)


handle = model.roi_heads.box_head.register_forward_hook(hook_features)
handle2 = model.roi_heads.box_predictor.register_forward_hook(hook_scores_box)
x = [transforms.ToTensor()(img1)]
_ = model(x)
handle.remove()
handle2.remove()
print('')
