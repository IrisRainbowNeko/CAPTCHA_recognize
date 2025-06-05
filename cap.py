# -*- coding: utf-8 -*-
import torch
import argparse
from models import ResnetEncoderDecoder
from utils import remove_rptch
from safetensors import safe_open
from torchvision import transforms as T
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

char_dict = '_0123456789abcdefghijklmnopqrstuvwxyz'
id_chr_map = {i: c for i, c in enumerate(char_dict)}


class Predictor:
    def __init__(self, model_path, char_dict=char_dict):
        self.model = ResnetEncoderDecoder(char_dict).to(device)
        self.model.eval()
        if str(device)=='cpu':
            check_point = self.load_safetensor(model_path, map_location=torch.device('cpu'))
        else:
            check_point = self.load_safetensor(model_path)
        self.model.load_state_dict(check_point)
        self.char_dict = char_dict
        
        self.trans = T.Compose([
                        T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

    # >>>>> from RainbowNeko Engine >>>>>
    @staticmethod
    def fold_dict(safe_f, split_key=':'):
        dict_fold = {}

        for k in safe_f.keys():
            k_list = k.split(split_key)
            dict_last = dict_fold
            for item in k_list[:-1]:
                if item not in dict_last:
                    dict_last[item] = {}
                dict_last = dict_last[item]
            dict_last[k_list[-1]]=safe_f.get_tensor(k)

        return dict_fold

    def load_safetensor(self, ckpt_f, map_location='cpu'):
        with safe_open(ckpt_f, framework="pt", device=map_location) as f:
            sd_fold = self.fold_dict(f)
        return sd_fold
    # <<<<< from RainbowNeko Engine <<<<<

    def pred(self, input):
        pred = self.model(input.to(device))

        B, H, W, C = pred.size()
        T_ = H * W
        pred = pred.view(B, T_, -1)
        pred = pred + 1e-10

        pred_cls = torch.max(pred, 2)[1].data.cpu().numpy()[0]

        pred_cls = pred_cls.reshape((H, W)).T.reshape((H * W,))
        final_str = remove_rptch(''.join(self.char_dict[x] for x in pred_cls if x))
        
        return pred_cls, final_str, (H, W)

    def pred_img(self, img, show=True):
        image = Image.open(img).convert('RGB')
        image = self.trans(image)
        pred_cls, final_str, (H, W) = self.pred(image.unsqueeze(0))

        if show:
            pred_string = ''.join(['%2s' % self.char_dict[pn] for pn in pred_cls])
            pred_string_set = [pred_string[i:i + W * 2] for i in range(0, len(pred_string), W * 2)]
            print('Prediction: ')
            for pre_str in pred_string_set:
                print(pre_str)
            
            print('Result:', final_str)

        return final_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAPTCHA Recognizer')
    parser.add_argument('--model_path', type=str, default='exps/captcha/ckpts/model-2000.safetensors', help='Path to the model file')
    parser.add_argument('--image_path', type=str, default=[
            '/data1/dzy/CAPTCHA_recognize/data3/test/2.jpg',
            '/data1/dzy/Verification_Code_CV_v1.1/imgs/00097.png',
            '/data1/dzy/Verification_Code_CV_v1.1/imgs/00098.png',
            '/data1/dzy/Verification_Code_CV_v1.1/imgs/00099.png',
        ], nargs='+', help='Path to the image file')
    args = parser.parse_args()

    predictor = Predictor(args.model_path)
    for path in args.image_path:
        result = predictor.pred_img(path)
        print(f'Recognized CAPTCHA: {result}')