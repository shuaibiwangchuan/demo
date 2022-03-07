# demo
简化图片
Paddlehub助力转化图片风格



一、模型背景介绍
小伙子，想拥有一份独一无二的图片风格嘛，想把你的图片转换成第一无二的简笔画嘛，只需三步即可，fork一下试一试吧

二、数据介绍
image_adress:需要转换的图片的绝对路径或者相对路径，如"/home/aistudio/image.jpg" style:选择的风格，现提供两种参数，分别为"style1"(简笔黑白)， "style2"(简笔彩色)， "style3"(水墨风格) alpha:风格迁移的程度(强度)，可选择的范围是为0-1

三、模型介绍
艺术风格迁移模型可以将给定的图像转换为任意的艺术风格。本模型StyleProNet整体采用全卷积神经网络架构(FCNs)，通过encoder-decoder重建艺术风格图片。StyleProNet的核心是无参数化的内容-风格融合算法Style Projection，模型规模小，响应速度快。模型训练的损失函数包含style loss、content perceptual loss以及content KL loss，确保模型高保真还原内容图片的语义细节信息与风格图片的风格信息。预训练数据集采用MS-COCO数据集作为内容端图像，WikiArt数据集作为风格端图像。

四、模型训练


来试试paddlehub吧

tips: 本项目利用艺术风格迁移模型stylepro_artistic，进行简约风格迁移，生成独一无二的图片



①安装paddlehub

In [1]
!pip install paddlehub==1.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
②下载模型

In [2]
!hub install stylepro_artistic==1.0.0
③一键生成头像






使用说明2

①图片的位置



②参数的调整



In [3]
import paddlehub as hub
import matplotlib.pyplot as plt
import cv2

# 定义头像生成函数
def GetImage(image_adress, style, alpha):
    style_adress = '/home/aistudio/work/' + style + '.jpg'

    stylepro_artistic = hub.Module(name="stylepro_artistic")

    results = stylepro_artistic.style_transfer(
        images=[{
            'content': cv2.imread(image_adress),
            'styles': [cv2.imread(style_adress)]
        }],
        alpha = alpha,
        visualization = True,
    )

    cv2.imwrite('/home/aistudio/transfer.jpg', results[0]['data'])
展示效果

In [4]
# 原图
img = cv2.imread('1.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2349: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  if isinstance(obj, collections.Iterator):
/opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/matplotlib/cbook/__init__.py:2366: DeprecationWarning: Using or importing the ABCs from 'collections' instead of from 'collections.abc' is deprecated, and in 3.8 it will stop working
  return list(data) if isinstance(data, collections.MappingView) else data

<Figure size 432x288 with 1 Axes>
Style1(简笔黑白)

In [5]
GetImage(image_adress='1.jpg', style='style1', alpha=0.98)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()
[2022-03-02 14:16:03,991] [    INFO] - Installing stylepro_artistic module
[2022-03-02 14:16:04,212] [    INFO] - Module stylepro_artistic already installed in /home/aistudio/.paddlehub/modules/stylepro_artistic
W0302 14:16:04.214905   101 analysis_predictor.cc:1350] Deprecated. Please use CreatePredictor instead.

<Figure size 432x288 with 1 Axes>
Style2(简笔彩色)

In [7]
GetImage(image_adress='1.jpg', style='style2', alpha=1)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()
[2022-03-02 14:16:38,660] [    INFO] - Installing stylepro_artistic module
[2022-03-02 14:16:38,663] [    INFO] - Module stylepro_artistic already installed in /home/aistudio/.paddlehub/modules/stylepro_artistic

<Figure size 432x288 with 1 Axes>
Style3(水墨风格)

In [8]
GetImage(image_adress='1.jpg', style='style3', alpha=0.9)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
plt.show()
[2022-03-02 14:17:06,500] [    INFO] - Installing stylepro_artistic module
[2022-03-02 14:17:06,504] [    INFO] - Module stylepro_artistic already installed in /home/aistudio/.paddlehub/modules/stylepro_artistic

<Figure size 432x288 with 1 Axes>
小伙子，发挥你的想象力，尝试一下组合风格吧


In [9]
GetImage(image_adress='1.jpg', style='style1', alpha=0.9)
GetImage(image_adress='transfer.jpg', style='style3', alpha=1)
img = cv2.imread('transfer.jpg')
b,g,r = cv2.split(img)
img = cv2.merge([r,g,b])
%matplotlib inline
plt.imshow(img)
[2022-03-02 14:18:38,590] [    INFO] - Installing stylepro_artistic module
[2022-03-02 14:18:38,654] [    INFO] - Module stylepro_artistic already installed in /home/aistudio/.paddlehub/modules/stylepro_artistic
[2022-03-02 14:18:50,649] [    INFO] - Installing stylepro_artistic module
[2022-03-02 14:18:50,653] [    INFO] - Module stylepro_artistic already installed in /home/aistudio/.paddlehub/modules/stylepro_artistic
<matplotlib.image.AxesImage at 0x7f6546759390>

<Figure size 432x288 with 1 Axes>
五、模型评估
六、总结与升华
使用说明1

"""
image_adress:需要转换的图片的绝对路径或者相对路径，如"/home/aistudio/image.jpg"
style:选择的风格，现提供两种参数，分别为"style1"(简笔黑白)， "style2"(简笔彩色)， "style3"(水墨风格)
alpha:风格迁移的程度(强度)，可选择的范围是为0-1
"""

七、个人总结
关于作者

学校	青岛科技大学  研一在读
感兴趣的方向	图像视频、强化学习
个人兴趣	任何有趣的事情
