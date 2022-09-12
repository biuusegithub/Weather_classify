from random import random
import Tool as tl
from torchvision import transforms
import glob
from torch.utils import data
import model as ml
from d2l import torch as d2l
import torch

'''
# 爬取数据
keywords = ["晴天", "雨天", "雪天", "雾天"]
savewords = ["sun", "rain", "snow", "Foggy"]


#save_dir = "img/" + keyword
for keyword, saveword in zip(keywords, savewords):
    url = 'https://image.baidu.com/search/acjson?'
    page_num = 8
    save_dir = "img/" + saveword
    tl.get_images_from_baidu(url, keyword, page_num, save_dir)
'''


'''
# 批量改文件名
path = "img/sun"
new_name = "sun"
tl.rename_file(path, new_name)
'''



imgs_path = "img"
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

all_imgs_path = glob.glob(r"C:\Users\80516\Desktop\emotion_detect\img\*\*.JPG")

types = ["sun", "rain", "snow", "Foggy"]

all_labels = []
for img in all_imgs_path:
    for index, type in enumerate(types):
        if type in img:
            all_labels.append(index)

weather_dataset = tl.Weather_dataset(all_imgs_path, all_labels, transform)

weather_dataloader = data.DataLoader(weather_dataset, batch_size=batch_size, shuffle=True)

train_len, test_len = int(len(weather_dataset)*0.8), int(len(weather_dataset)*0.2)

# 随机划分数据集
train_dataset, test_dataset = data.random_split(
    dataset=weather_dataset, 
    lengths=[train_len, test_len]
    )


train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# imgs, labels = next(iter(train_dataloader))
# tl.show_img(imgs, labels)


net = ml.get_net_2()

lr, num_epochs, wd = 2e-4, 30, 0.1
tl.train_ch6(net, train_dataloader, test_dataloader, num_epochs, lr, wd, d2l.try_gpu())

tl.predict(net, test_dataloader, d2l.try_gpu())

d2l.plt.show()














