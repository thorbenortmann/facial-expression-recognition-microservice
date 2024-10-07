import os
import platform
from collections import OrderedDict
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from fer.posterv2.PosterV2_7cls import pyramid_trans_expr2

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


class PosterV2Recognizer:
    def __init__(self,
                 model_name: str = 'affectnet-7-model_best_state_dict_only.pth',
                 emotion_labels: Optional[List[str]] = None):
        self.model_path = Path(__file__).parent / model_name
        default_emotion_labels = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger']
        self.emotion_labels = emotion_labels if emotion_labels is not None else default_emotion_labels

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.print_system_report()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self.load_model()

    def load_model(self):
        model = pyramid_trans_expr2(img_size=224, num_classes=7)
        model = model.to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device)

        if self.device.type == 'cpu' or torch.cuda.device_count() <= 1:
            new_state_dict = OrderedDict({k[7:] if k.startswith('module.') else k: v
                                          for k, v in state_dict.items()})
            model.load_state_dict(new_state_dict)
        else:
            model = torch.nn.DataParallel(model)
            model.load_state_dict(state_dict)

        model.eval()
        return model

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        :param image: numpy array of shape (height, width, channels)
        :return: numpy array of shape (1, channels, height, width)
        """
        image = Image.fromarray(image.astype('uint8'), 'RGB')
        image = self.transform(image)
        image = Variable(image.unsqueeze(0)).to(self.device)
        return image

    def predict_emotions(self, face_img: np.ndarray) -> list[float]:
        """
        :param face_img: numpy array of shape (height, width, channels)
        :return: probability scores in the order given in self.emotion_labels.
        """
        with torch.no_grad():
            output = self.model(self.preprocess(face_img))
        probabilities = F.softmax(output, dim=1)
        return probabilities.cpu().detach().numpy()[0].tolist()

    def print_system_report(self):
        print(f'\n---System Info---')
        print(f'Operating System: {platform.system()} {platform.version()}')
        print(f'Python Version: {platform.python_version()}')
        print(f'PyTorch Version: {torch.__version__}')
        if self.device.type == 'cuda':
            print('\n---CUDA info---')
            print(f'CUDA Version: {torch.version.cuda}')
            print(f'cuDNN Version: {torch.backends.cudnn.version()}')
            print(f'CUDA Available: {torch.cuda.is_available()}')
            print(f'GPU(s) available: {torch.cuda.device_count()}')
            print(f'Current GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        else:
            print('\n---CPU info---')
            print('Processor: ', platform.processor())
            print('CPU count: ', os.cpu_count())

        print('\n')
