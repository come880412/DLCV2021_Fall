import torch
import numpy as np
from Dataset_test import test_seg
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import DataLoader
from utils import pred_to_mask
from PIL import Image
import sys

def test(model, out_path, test_loader):
    cuda = True if torch.cuda.is_available() else False
    if cuda:
        model = model.cuda()
    model.eval()
    with torch.no_grad():
        for image, image_name in test_loader:
            image = image.cuda()
            pred = model(image)
            pred = pred['out']
            pred = pred.cpu().detach().numpy()
            pred_label = np.argmax(pred, axis=1)
            segment_image = pred_to_mask(pred_label)
            for batch in range(len(segment_image)):
                output_image = np.array(segment_image[batch]).astype(np.uint8)
                image_save_name = '%s/%s' % (out_path, image_name[batch])
                save_image = Image.fromarray(output_image)
                save_image.save(image_save_name)
                    
if __name__ == "__main__":
    test_data_path = sys.argv[1]
    out_path = sys.argv[2]

    test_data = test_seg(test_data_path)
    test_loader = DataLoader(test_data, batch_size=4, shuffle=False, num_workers=8)

    model = deeplabv3_resnet50(pretrained=False, num_classes=7, aux_loss=True)
    model.load_state_dict(torch.load('model_miou71.54.pth?dl=1'))
    
    test(model, out_path, test_loader)