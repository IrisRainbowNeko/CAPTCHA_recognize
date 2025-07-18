import json
import random
import math
from pathlib import Path

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from tqdm import tqdm

char_dict = '0123456789abcdefghijklmnopqrstuvwxyz'
char_dict_pp = '0123456789abcdefghijklmnopqrstuvwxyz()+-*/='

class FontCache:
    def __init__(self, font_root, font_size_range=(24, 38)):
        self.font_dict = {}
        font_root = Path(font_root)
        for font_path in font_root.iterdir():
            self.font_dict[font_path.name] = {}
            for font_size in range(font_size_range[0], font_size_range[1]+1):
                self.font_dict[font_path.name][font_size] = ImageFont.truetype(font_path, size=font_size)
        
        self.font_names = list(self.font_dict.keys())
        self.font_sizes = list(range(font_size_range[0], font_size_range[1]+1))

    def get(self, name, size):
        return self.font_dict[name][size]

    def random_font(self):
        font_name = random.choice(self.font_names)
        font_size = random.choice(self.font_sizes)
        return self.font_dict[font_name][font_size]


def random_color(low=0, high=255):
    c1 = random.randint(low, high)
    c2 = random.randint(low, high)
    c3 = random.randint(low, high)
    return c1, c2, c3


def generate_picture(width=120, height=40):
    image = Image.new('RGB', (width, height), random_color(low=200))
    return image


def random_str(char_dict=char_dict_pp):
    '''
    获取一个随机字符
    :return:
    '''
    random_char = random.choice(char_dict)
    return random_char


def draw_str(count, image, font_cache:FontCache, angle_range=(-15, 15)):
    """
    在图片上写随机字符（带随机缩放和旋转）
    :param count: 字符数量
    :param image: 图片对象
    :param font_size: 字体大小
    :return: 验证码字符串和处理后的图片
    """
    temp = []
    avg_char_width = image.width // count  # 计算每个字符的平均宽度

    for i in range(count):
        random_char = random_str()
        temp.append(random_char)

        # 随机字体与大小
        font = font_cache.random_font()

        # 计算字符的边界框
        bbox = font.getbbox(random_char)
        char_width = math.ceil(bbox[2] - bbox[0])  # 获取字符的宽度
        char_height = math.ceil(bbox[3] - bbox[1])  # 获取字符的高度
        char_x = 0
        char_y = -bbox[1]  # 垂直居中

        # 随机旋转角度 (-30到30度)
        rotation_angle = random.randint(angle_range[0], angle_range[1])

        # 创建临时图片来绘制单个字符
        # 需要足够大的画布来容纳旋转后的字符
        temp_img = Image.new('RGBA', (char_width, char_height), (255, 255, 255, 0))
        temp_draw = ImageDraw.Draw(temp_img)

        # 在临时图片中心绘制字符
        char_color = random_color()
        temp_draw.text((char_x, char_y), random_char, char_color, font=font)

        # 旋转字符
        if rotation_angle != 0:
            temp_img = temp_img.rotate(rotation_angle, expand=True)

        # 计算粘贴位置
        dy = (image.height - temp_img.height) // 2  # 垂直居中
        ofy = (char_height*0.2)
        ofx = (avg_char_width*0.15)
        char_x = i * avg_char_width + int(random.gauss(0, ofx))
        char_y = dy + int(random.gauss(0, ofy))

        # 确保不超出边界
        char_x = max(0, min(char_x, image.width - temp_img.width))
        # char_y = max(0, min(char_y, image.height - temp_img.height))

        # 将旋转后的字符粘贴到主图片上
        if temp_img.mode == 'RGBA':
            image.paste(temp_img, (char_x, char_y), temp_img)
        else:
            image.paste(temp_img, (char_x, char_y))

    valid_str = "".join(temp)  # 验证码
    return valid_str, image


def draw_lines(image, line_count_range=(4, 7), line_width_range=(0, 0)) -> Image.Image:
    '''

    :param image: 图片对象
    :param width: 图片宽度
    :param height: 图片高度
    :param line_count: 线条数量
    :param point_count: 点的数量
    :return:
    '''
    width, height = image.size

    draw = ImageDraw.Draw(image)
    line_count = random.randint(line_count_range[0], line_count_range[1])
    for i in range(line_count):
        x1 = random.randint(0, width)
        x2 = random.randint(0, width)
        y1 = random.randint(0, height)
        y2 = random.randint(0, height)

        line_width = random.randint(line_width_range[0], line_width_range[1])
        draw.line((x1, y1, x2, y2), fill=random_color(), width=line_width)

    return image


def cap_gen(data_root, name, font_cache, str_count_rage=(4,4), width=120, height=40, scale_factor=4):
    """
    生成图片验证码,并对图片进行base64编码
    :return:
    """

    image = generate_picture(width=int(width * scale_factor), height=int(height * scale_factor))
    text, image = draw_str(random.randint(*str_count_rage), image, font_cache)
    image = draw_lines(image, line_width_range=(int(2 * scale_factor), int(4.5 * scale_factor)))
    image = image.resize((width, height), resample=Image.Resampling.BICUBIC)

    image.save(data_root / f'{name}.jpg', quality=100)  # 保存到BytesIO对象中, 格式为png

    return text


if __name__ == '__main__':
    data_path = Path('data_pp_v2/test')
    data_path.mkdir(parents=True, exist_ok=True)

    width_list = [80, 100, 120, 140]
    height_list = [35, 40, 45, 50]
    scale_factor = 4

    font_size_range = (int(24 * scale_factor), int(38 * scale_factor))
    font_cache = FontCache('fonts/', font_size_range=font_size_range)

    cap_dict = {}
    header = ['ID', 'label']
    # for i in tqdm(range(0, 200000)):
    #     cnt = cap_gen(data_path, i, font_cache, str_count_rage=(3,6), width=random.choice(width_list),
    #                    height=random.choice(height_list), scale_factor=scale_factor)
    #     cap_dict[f'{data_path.name}/{i}.jpg'] = cnt
    for i in tqdm(range(0, 1000)):
        cnt = cap_gen(data_path, i, font_cache, width=100, height=40, scale_factor=scale_factor)
        cap_dict[f'{data_path.name}/{i}.jpg'] = cnt

    with open(f'{data_path}.json', 'w') as f:
        json.dump(cap_dict, f, ensure_ascii=False, indent=2)