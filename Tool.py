import os
from tkinter import Image
import requests
import os
import re
import time
import torch
from torch.utils import data
from PIL import Image
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn
############################################################################################################################

# 文件重命名函数
def rename_file(path, newname):
    # path = "img/angry"
    # new_name = "angry_"

    fileList = os.listdir(path)

    for i, _ in enumerate(fileList):
        
        # 设置旧文件名（就是路径+文件名）
        # os.sep添加系统分隔符
        old_name = path + os.sep + fileList[i]   
        
        # 设置新文件名
        new_name = path + os.sep + newname + str(i+1) + '.JPG'
        
        # 用os模块中的rename方法对文件改名
        os.rename(old_name, new_name)

    print("rename over")   


# 百度图片下载函数
def get_images_from_baidu(url, keyword, page_num, save_dir):
    # UA 伪装：当前爬取信息伪装成浏览器, 将 User-Agent 封装到一个字典中
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36 Edg/105.0.1343.27'}
    
    # 请求的 url, 这里是百度图片网址
    url = url
    n = 0
    for pn in range(0, 30 * page_num, 30):
        # 请求参数
        param = {'tn': 'resultjson_com',
                # 'logid': '7293928089621024592'
                'ipn': 'rj',
                'ct': 201326592,
                'is': '',
                'fp': 'result',
                'fr': '',
                'word': keyword,
                'queryWord': keyword,
                'cl': 2,
                'lm': -1,
                'ie': 'utf-8',
                'oe': 'utf-8',
                'adpicid': '',
                'st': '-1',
                'z': '',
                'ic': 0,
                'hd': '',
                'latest': '',
                'copyright': '',
                's': '',
                'se': '',
                'tab': '',
                'width': '',
                'height': '',
                'face': 0,
                'istype': 2,
                'qc': '',
                'nc': 1,
                'expermode': '',
                'nojc': '',
                'isAsync': '',
                'pn': pn,
                'rn': 30,
                'gsm': 78,
                '1662537094424': '',
                }

        request = requests.get(url=url, headers=header, params=param)

        if request.status_code == 200:
            print('Request success.')

        request.encoding = 'utf-8'

        # 正则方式提取图片链接
        # 打开网页源码，搜索thumbURL，这就是图片连接，(.*?)这个正则式表示匹配到？为止，re.S表示仅在改行匹配
        html = request.text
        image_url_list = re.findall('"thumbURL":"(.*?)",', html, re.S)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for image_url in image_url_list:
            image_data = requests.get(url=image_url, headers=header).content
            with open(os.path.join(save_dir, f'{n:d}.jpg'), 'wb') as fp:
                fp.write(image_data)
            n = n + 1
            time.sleep(1)
        print("download over")


# 自定义dataset
class Weather_dataset(data.Dataset):
    def __init__(self, root_path, labels, transform):
        self.imgs_path = root_path
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.imgs_path[index]
        label = self.labels[index]
        # 这里由于我爬取的图片即有灰度图又有 RGB 图，所以需要处理一下统一转换为 RGB
        img = self.transform(Image.open(img_path).convert('RGB'))
        return img, label

    def __len__(self):
        return len(self.imgs_path)

    
# 图片显示
types = ["sun", "rain", "snow", "Foggy"]
type_to_index = dict((type, index) for index, type in enumerate(types))
index_to_type = dict((index, type) for type, index in type_to_index.items())

def show_img(imgs, labels):
    plt.figure(figsize=(12, 8))
    for i, (img, label) in enumerate(zip(imgs[:8], labels[:8])):
        img = img.permute(1, 2, 0).numpy()
        plt.subplot(2, 4, i+1)      # 这里设置显示8张图
        plt.title(index_to_type.get(int(label)))
        plt.imshow(img)
    plt.show()

    
# 训练函数
def train_ch6(net, train_iter, test_iter, num_epochs, lr, wd, device):
    """用GPU训练模型"""
    
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
            
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=lr,  weight_decay=wd)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
                
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


# 预测函数
def predict(net, dataloader, device):
    net.eval()
    for imgs, labels in dataloader:
        break 
    pred = net(imgs.to(device)).argmax(axis=1)
    show_img(imgs, pred)
    
###############################################################################################################################
