# Motorcycle-License-Plate-Recognition-Helmet-Detection-YOLO-CRNN-Graduation-Project
本项目通过YOLOv11/10/9/8/7/6/5的摩托车/电动车相关目标检测+CRNN摩托车/电动车车牌字符识别+逻辑后处理算法，实现了一个检测二轮车相关多个目标（二轮车、二轮车车牌、头盔、未戴头盔），并识别二轮车车牌，输出未戴头盔的二轮车车牌的算法，可用于实际企业工业项目，也可作为同学们的毕设参考或自己的学习资料。可提供整套代码(含详细注释)、训练好的权重、数据集、测试视频和详细说明文档。可以部署到树莓派、香橙派、Jetson Nano、瑞芯微RK3588等开发板上，也可调用摄像头输入视频流进行实时推理。


# 一、前言
**本项目通过YOLOv11/10/9/8/7/6/5的摩托车/电动车相关目标检测+CRNN摩托车/电动车车牌字符识别+逻辑后处理算法，实现了一个检测二轮车相关多个目标（二轮车、二轮车车牌、头盔、未戴头盔），并识别二轮车车牌，输出未戴头盔的二轮车车牌的算法，可用于实际企业工业项目，也可作为同学们的毕设参考或自己的学习资料。可提供整套代码(含详细注释)、训练好的权重、数据集、测试视频和详细说明文档。可以部署到树莓派、香橙派、Jetson Nano、瑞芯微RK3588等开发板上，也可调用摄像头输入视频流进行实时推理。**


**效果视频：**
  [https://www.bilibili.com/video/BV1Rrz4YTEFL/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a](https://www.bilibili.com/video/BV1Rrz4YTEFL/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a)

**操作过程视频：**
 [https://www.bilibili.com/video/BV18LzhYrEd6/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a](https://www.bilibili.com/video/BV18LzhYrEd6/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a)

## 1、测试效果展示
可以看到，我们实验室的项目能有效实现头盔佩戴情况检测的各种功能(多目标检测、头盔佩戴情况检测、在图片或视频中显示车牌号并将未戴头盔的车牌号输出保存为txt文档)。.

[video(video-lSW9B68w-1733045450043)(type-csdn)(url-https://live.csdn.net/v/embed/436663)(image-https://i-blog.csdnimg.cn/img_convert/d6734dccbad25ad5091075289b8d055a.jpeg)(title-未戴头盔检测-摩托车/电动车车牌识别-原创毕设)]




![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/57cf3624d40d4523b99fa07517a8a128.jpeg#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4c11fec667b64afcba18773233ece697.png#pic_center) 

## 2、项目介绍
二轮车未戴头盔的车牌识别算法的实际应用意义在于提高交通管理的效率和安全性。通过自动化检测未戴头盔的骑行者并识别其车牌号码，该技术有助于交通管理部门快速识别和处理交通违规行为，从而减少交通事故和伤亡。此外，该算法还能辅助智慧城市建设，通过智能监控和数据分析，提升城市交通的智能化管理水平

我们的项目可为兄弟们的毕设、课设、大作业等提供参考，也可为工业实际项目提供技术支撑，可训练自己的数据集，可以换成yolov11/10/9/8/7/6/5各种版本的权重。包含特别详细的read.md文件和常见问题解答，关于本项目的任何问题都能在其中找到答案，对刚接触深度学习、目标检测的小白非常友好，兄弟们放心哈。


# 二、项目环境配置
我们的项目环境要求如下：
1、Python 3.7+
2、OpenCV、 Scikit-learn、onnxruntime、pycuda、pytorch
3、我们项目中requirements.txt里面包含的所有包

**强调一下**：建议直接拿我们的项目过去，一边配环境一边调试运行，看代码运行窗口输出报什么错，就对应搜csdn，针对性装什么包。不要想着一次性装好所有东西再执行项目，一方面有可能自己感觉装好了，还是缺包，一方面很可能存在包的版本不兼容等问题，这样效率非常低。

不熟悉pycharm的anaconda的大兄弟请先看这篇csdn博客，了解pycharm和anaconda的基本操作。
[https://blog.csdn.net/ECHOSON/article/details/117220445](https://blog.csdn.net/ECHOSON/article/details/117220445)
anaconda安装完成之后请切换到国内的源来提高下载速度 ，命令如下：

```python
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```
首先创建python3.8的虚拟环境，请在命令行中执行下列操作：

```python
conda create -n yolov8 python==3.8.5
conda activate yolov8
```
## 1、pytorch安装
实际测试情况是我们的项目在CPU和GPU的情况下均可使用，不过CPU推理稍慢而已，有独显的小伙伴还是尽量安装GPU版本的Pytorch。GPU版本安装的具体步骤可以参考这篇文章：[https://blog.csdn.net/ECHOSON/article/details/118420968](https://blog.csdn.net/ECHOSON/article/details/118420968)。
需要注意以下几点：
1、安装之前一定要先更新你的显卡驱动，去官网下载对应型号的驱动安装
2、30系显卡只能使用cuda11的版本
3、一定要创建虚拟环境，这样的话各个深度学习框架之间不发生冲突
我这里创建的是python3.8的环境，安装的Pytorch的版本是1.8.0，命令如下：

```python
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
```
安装完毕之后，我们来测试一下GPU是否可以有效调用：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a79b18bf74273e070fa9ff255a6fe311.png#pic_center)
## 2、pycocotools的安装
```python
pip install pycocotools-windows
```
## 3、pycuda安装
参考博客链接：[http://t.csdnimg.cn/AqIYW](http://t.csdnimg.cn/AqIYW)
## 4、其他包的安装
另外的话大家还需要安装程序其他所需的包，包括opencv，matplotlib这些包，不过这些包的安装比较简单，直接通过pip指令执行即可，我们cd到项目代码的目录下，在终端中直接执行下列指令即可完成。

```python
pip install -r requirements.txt
```

# 三、摩托车/电动车相关目标检测(yolo)和二轮车车牌字符识别(CRNN)
## 1、yolov9算法（以yolov9为例进行展示，可提供多种yolo版本）
YOLOv9是YOLO (You Only Look Once)系列实时目标检测系统的最新版本。它建立在以前的版本之上，融合了深度学习技术和架构设计的进步，以在对象检测任务中实现卓越的性能。YOLOv9将可编程梯度信息(PGI)概念与通用ELAN(GELAN)架构相结合而开发，代表了准确性、速度和效率方面的重大飞跃。

YOLOv9主要特点：
1.实时对象检测: YOLOv9通过提供实时对象检测功能保持了YOLO系列的标志性功能。这意味着它可以快速处理输入图像或视频流，并准确检测其中的对象，而不会影响速度。
2.PGI集成: YOLOv9融合了可编程梯度信息(PGI)概念，有助于通过辅助可逆分支生成可靠的梯度。这确保深度特征保留执行目标任务所需的关键特征，解决深度神经网络前馈过程中信息丢失的问题。
3.GELAN架构: YOLOv9采用通用ELAN (GELAN)架构，旨在优化参数、计算复杂度、准确性和推理速度。通过允许用户为不同的推理设备选择合适的计算模块，GELAN增强了YOLOv9的灵活性和效率。
4.性能提升:实验结果表明，YOLOv9在MS COCO等基准数据集上的目标检测任务中实现了最佳性能。它在准确性、速度和整体性能方面超越了现有的实时物体检测器，使其成为需要物体检测功能的各种应用的最先进的解决方案。
5.灵活性和适应性: YOLOv9旨在适应不同的场景和用例。其架构可以轻松集成到各种系统和环境中，使其适用于广泛的应用，包括监控、自动驾驶车辆、机器人等。
6.主分支集成: PGI的主分支代表网络在推理过程中的主要路径，可以无缝集成到YOLOv9架构中。这种集成确保推理过程保持高效，而不会产生额外的计算成本。
7.辅助可逆分支: YOLOv9和许多深度神经网络一样，随着网络的加深，可能会遇到信息瓶颈的问题。可以合并PGI的辅助可逆分支来解决这个问题，为梯度流提供额外的路径，从而确保损失函数的梯度更可靠。
8.多级辅助信息: YOLOv9通常采用特征金字塔来检测不同大小的物体。通过集成来自PGI的多级辅助信息，YOLOv9可以有效处理与深度监督相关的错误累积问题，特别是在具有多个预测分支的架构中。这种集成确保模型可以从多个级别的辅助信息中学习，从而提高不同尺度的对象检测性能。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/56fbb55ee6475af24a7785fb481466c6.png#pic_center)
## 2、CRNN算法介绍
CRNN是“卷积递归神经网络”（Convolutional Recurrent Neural Network）的缩写。它是一种深度学习架构，结合了卷积神经网络（CNN）和循环神经网络（RNN）的优势，主要用于处理具有序列性和空间信息的数据，比如图像中的文字识别。

CRNN的结构包含了卷积层、循环层和连接层。首先，卷积层用于提取图像特征，将输入图像转换为高层次的抽象特征表示。这些特征捕获了文字在不同尺度和方向上的信息，使得模型对文字的变化和形态有较强的理解能力。

接着，循环层（通常采用长短时记忆网络，LSTM，或者门控循环单元，GRU）用于处理序列数据，它能够保留文字之间的上下文信息。这使得CRNN能够更好地理解文字之间的关系，并且有助于纠正识别错误。

最后，连接层用于将卷积层和循环层的输出结合起来，并通过全连接层进行最终的分类或识别。这个结构允许模型同时利用局部特征和全局上下文信息，提高了对文字的准确识别能力。

CRNN在文字识别领域取得了很大成功，特别是在场景文本识别（如自然场景中的文字识别）方面。它能够处理不同字体、大小、角度和背景的文字，并且对于不同语言的文字具有一定的通用性。

总的来说，CRNN作为结合了CNN和RNN的深度学习架构，具有处理序列数据和空间信息的能力，特别适用于文字识别等领域，为处理具有结构性数据的任务提供了一种有效的解决方案。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aa9d005e8d6f4383732e3726c9e967aa.png#pic_center)
# 四、算法流程设计
## 1、整体算法逻辑 
（1）输入数据准备
接收输入参数，包括图片路径、视频路径或目录路径，并验证输入文件是否存在；同时指定模型权重的加载路径和处理结果的输出路径，确保环境准备就绪。如果是目录，则逐一加载其中的图片文件。
（2）模型加载与初始化
加载两个模型：一个是目标检测模型（用于检测二轮车、车牌和头盔），另一个是车牌识别模型（用于识别车牌号码）；根据设备可用性（GPU 或 CPU）选择硬件支持，并完成初始化。
（3）图像预处理
对输入的图片或从视频中提取的帧进行标准化处理，包括调整大小、归一化及通道转换，确保输入数据符合模型的要求，提升检测的准确性和效率。
（4）目标检测
使用目标检测模型对图像中的所有相关对象（二轮车、车牌、头盔）进行识别，返回每个对象的边界框位置、类别标签和置信度，以及关键点信息（车牌的四个关键角点坐标）。目标检测的标签信息：
```python
# class names
names: ['two_wheeler', 'two_wheeler_license_plate', "helmetless", "helmet"]
```
（5）车牌识别
针对检测到的车牌区域，使用四点透视变换将车牌裁剪为矩形图像，并将其传入车牌识别模型，输出车牌号码，为后续逻辑判断提供基础数据。
（6）逻辑关联与判定
遍历检测结果，将二轮车、车牌和头盔的检测结果关联起来。检查每个二轮车区域是否包含车牌，且是否存在未佩戴头盔的驾驶员，若同时满足，则记录该车辆的车牌号码。
（7）结果绘制
在检测后的图片或视频帧中绘制边界框及识别信息，对未佩戴头盔的二轮车使用红框高亮显示，同时将车牌号码标注在对应的二轮车区域上（不管是否佩戴头盔，均识别且标注车牌），使结果直观可视化。
（8）结果保存
将绘制完成的图像或视频保存到指定的输出路径，同时生成一个文本文件，记录所有未佩戴头盔车辆的车牌号码列表，供后续分析或执法使用。

## 2、摩托车/电动车车牌识别逻辑 
首先，通过卷积神经网络（CNN）提取输入图像的特征。然后，使用Anchor Boxes来生成候选区域，这些区域包含可能的目标边界框。通过对这些候选区域进行分类和定位回归，确定最终的目标边界框和其类别。YOLOv9采用多尺度特征融合，以捕捉不同尺度的信息，提高检测性能。此外，它使用自适应的Anchor Box来适应不同目标形状。整个过程通过端到端的训练来优化网络参数，实现高效、准确的车牌检测。YOLOv9检测到的车牌如图：         
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/45d1728859d93ad39d8b3a1b3e9ad6c6.jpeg#pic_center =30%x)

如上图所示，检测有可能定位不准，导致车牌周边图像也被包含在感兴趣区域内。另外，检测出来的车牌会存在一定倾角，不利于后续的车牌字符识别。因此，对车牌进行关键点回归定位。如图所示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/db69da7102b8254476831ef3c8c603ad.jpeg#pic_center =30%x)
定位到车牌四个角点之后，使用数学图像处理中的透视变化技术对其进行矫正。透视变换原理详见[http://t.csdnimg.cn/RcdKB](http://t.csdnimg.cn/RcdKB)，此处不再赘述。具体代码实现如下：

```python
def four_point_transform(image, pts):                       #透视变换得到车牌小图
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
```
得到的矫正后车牌图像：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d8cbb2a50b0aeead58949dbe81a17a6d.jpeg#pic_center)
再将矫正后的车牌输入CRNN中进行字符识别，得到最终的字符识别效果，并在图像上以文本的形式输出。

```python
    class_label= int(class_num)  #车牌的的类型0代表单牌，1代表双层车牌
    roi_img = four_point_transform(img,landmarks_np)   #透视变换得到车牌小图
    if class_label:        #判断是否是双层车牌，是双牌的话进行分割后然后拼接
        roi_img=get_split_merge(roi_img)
    plate_number ,plate_color= get_plate_result(roi_img,device,plate_rec_model)                 #对车牌小图进行识别,得到颜色和车牌号
    # cv2.imwrite("roi.jpg",roi_img)
    result_dict['class_type']=class_type[class_label]
    result_dict['rect']=rect                      #车牌roi区域
    result_dict['landmarks']=landmarks_np.tolist() #车牌角点坐标
    result_dict['plate_no']=plate_number   #车牌号
    result_dict['roi_height']=roi_img.shape[0]  #车牌高度
    result_dict['plate_color']=plate_color   #车牌颜色
    result_dict['object_no']=class_label   #单双层 0单层 1双层
    result_dict['score']=conf           #车牌区域检测得分
    return result_dict
```
## 3、判断未戴头盔，并与二轮车车牌进行映射的逻辑
在 draw_result 函数中，判断未戴头盔并将其与二轮车的车牌号进行映射的逻辑可分为以下几个步骤：

```python
# 定义 draw_result 函数，用于在原始图像上绘制检测和识别结果
def draw_result(orgimg, dict_list):
    """
    在图像上绘制检测结果，显示车牌号和关键点，并记录未戴头盔车辆的车牌号。
    """
    no_helmet_plates = []  # 每张图片的未戴头盔车牌号列表
    for result in dict_list:
        rect_area = result['rect']
        object_no = result['object_no']
        if object_no == 0:  # 如果是 two_wheeler
            # 检查是否包含 license_plate 和 helmetless
            license_plate_obj = None
            contains_helmetless = False
            for obj in dict_list:
                # 判断目标是否在摩托车区域内
                if rect_area[0] <= obj['rect'][0] and rect_area[2] >= obj['rect'][2] and \
                        rect_area[1] <= obj['rect'][1] and rect_area[3] >= obj['rect'][3]:
                    if obj['class_type'] == 'two_wheeler_license_plate':
                        license_plate_obj = obj  # 记录车牌相关目标
                    if obj['class_type'] == 'helmetless':
                        contains_helmetless = True
                        ....
```
（1）遍历检测结果
遍历 dict_list 中的所有目标，寻找当前是二轮车的目标。二轮车目标由类别编号 object_no 等于 0 标识；每个二轮车目标的边界框信息保存在 rect_area 中，用于确定其区域范围。
（2）在二轮车区域内查找相关目标
检查 dict_list 中是否存在属于该二轮车范围内的其他目标。具体检查两个类别：
车牌目标（类别为 "two_wheeler_license_plate"）。如果目标的边界框完全在当前二轮车的 rect_area 范围内，则该目标为该二轮车的车牌。找到后，记录该车牌目标信息（车牌号 plate_no）。
未戴头盔目标（类别为 "helmetless"）。同样检查目标边界框是否在当前二轮车的 rect_area 内，如果找到，标记该二轮车包含未戴头盔的驾驶员。
（3）判断与记录
如果在二轮车范围内找到车牌目标和未戴头盔目标，说明该未佩戴头盔的行为属于该二轮车。则获取车牌目标的车牌号 plate_no，并将其添加到 no_helmet_plates 列表中，以记录该未佩戴头盔的二轮车信息。
（4）绘制信息到图像
在图像的车牌目标对应位置，绘制车牌号；在未戴头盔目标区域绘制红色透明矩形，用于视觉上标注该违规行为。

**这部分逻辑的作用是**：
通过空间位置匹配，将未佩戴头盔的驾驶员信息准确映射到具体的二轮车和其车牌号上。提供完整的违法记录信息，为后续分析或处罚提供依据。
# 五、代码使用
**使用pycharm打开项目，直接单击右键执行Car_recognition.py即可。测试视频、模型权重、类别等参数已经在代码中设置好，可直接运行！**
--detect_model：指定检测二轮车及其目标的模型权重路径，默认为 weights/best_detect.pt
--rec_model：指定车牌识别模型的权重路径，默认为 weights/plate_rec_color.pth
--image_path：输入图像所在的目录路径，默认为 test-new
--img_size：推理阶段输入图像的尺寸，默认为 480 像素
--output：指定保存检测结果的输出目录，默认为 result
--video：可选参数，用于指定视频输入路径（如果提供视频而非图像作为输入）


```python
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default='weights/best_detect.pt',
                        help='model.pt path(s)')  # 二轮车检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth',
                        help='model.pt path(s)')  # 车牌识别模型,对车牌小图进行识别得到车牌号
    parser.add_argument('--image_path', type=str, default='test-new', help='source')
    parser.add_argument('--img_size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='result', help='source')
    parser.add_argument('--video', type=str, default='', help='source')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```
# 六、自己训练的步骤
对于兄弟们的毕设、课设项目来说，没有必要再重新训练一遍。一方面耗时费力，自己的电脑也不一定跑的动；另一方面我这边会提供所有的训练过程曲线、数据、和训练好的权重，直接调用就行。
## 1、下载数据集
 数据是从CCPD和CRPD数据集中选取并转换的，为yolo格式：

```python
label x y w h  pt1x pt1y pt2x pt2y pt3x pt3y pt4x pt4y
```

关键点依次是（左上、右上、右下、左下）。坐标都是经过归一化，x、y是中心点除以图片宽高，w、h是框的宽高除以图片宽高，ptx、pty是关键点坐标除以宽高。车辆标注不需要关键点，关键点全部置为-1即可。
## 2、修改路径
换成自己的数据集路径。
```python
   train: /your/train/path #修改成你的路径
   val: /your/val/path     #修改成你的路径
   # number of classes
   nc: 3                #这里用的是3分类，0 单层车牌 1 双层车牌 2 车辆

   # class names
   names: [ 'single_plate','double_plate','Car'] 
```
## 3、开始训练

```python
python3 train.py --data data/plateAndCar.yaml --cfg models/yolov5n-0.5.yaml --weights weights/detect.pt --epoch 250
```
# 七、二轮车相关目标检测和车牌识别数据集
## 1、二轮车相关目标检测数据集
我们实验室手动收集、整理、标注了TWHD (two wheeler helmet dataset)数据集，包含5448张二轮车图片和对应的xml格式标签。我们对图片中的二轮车整体（two_wheeler）、未戴头盔的人头（without_helmet）、戴头盔的人头（helmet）使用软件进行了手动标注，并按4:1的比例划分训练集与测试集。此外，为了丰富数据集背景，还融合了来自bike helmet dataset以及网络爬虫的738张图片。本数据集可直接用于训练yolo系列等神经网络，可提供给兄弟们的毕设、课设项目及企业课题进行使用。数据集展示如下：
![请添加图片描述](https://i-blog.csdnimg.cn/blog_migrate/842eb39f65a10b47a4fee86ae42f895b.png#pic_center)
其中“Annotations”包含了5448个手动标注的xml格式标签，“ImageSets”文件夹里面为训练集、验证集划分对应的txt，“JPEGImages”文件夹里面为5448张JPG格式的包含电动车、摩托车、头盔的图片。

![请添加图片描述](https://i-blog.csdnimg.cn/blog_migrate/b509598ecfa4734acc9d95b94a967d52.png)
## 2、二轮车车牌识别数据集
我们实验室手动收集、整理了一个高质量的车牌识别、检测数据集，包含41892张车牌图片和对应的txt格式标签。已将其划分为训练集、测试集。本数据集可直接用于训练yolo系列等神经网络，可提供给兄弟们的毕设、课设项目及企业课题进行使用。数据集展示如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/bdbe8886ee4b170ba671c8f6970908f5.jpeg#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/166f8e7841f3d2c80260c6b72e1f288f.jpeg#pic_center)
# 八、训练曲线等介绍
我们的项目代码还能自动生成训练过程的loss损失曲线、map平均准确度曲线，不用手动画（太麻烦了，能用代码做的事尽量不手动），兄弟可以直接将这些图插入论文或课设报告中。当然，也可以自己训练，重新生成对应的图。训练结束后，这些图和训练数据会(以envents文件形式)存放在根目录下的runs文件夹中。我项目中已导出为PNG图片和CSV表格，可以直接拿去用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d20421588cc73a43c4d54fbf1826dffe.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/f711b335c6caa11f2f6b9a2163d64b65.png#pic_center)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1fc2fd19398240b79485f646105db12f.png#pic_center)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/95dde34a24d3be29e9f6478a18ae607b.jpeg#pic_center)
包含完整word版本说明文档（共16页，6102字），可用于写论文、课设报告的参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/083d835e23194473bdae21c9500c2eb3.png#pic_center)
# 九、资源获取(yolov11/yolov10/yolov9/yolov8/yolov7/yolov5版本均可提供)
**可提供整套代码加训练好的权重，还有测试视频和详细说明文档。代码有详细注释，包全程指导，任何问题都可以随时问我。不过有的时候我太忙，可能不会及时回复消息，看到了肯定回你哈**

资源获取：
```python
获取整套代码、测试视频、训练好的权重和说明文档(有偿)
上交硕士，技术够硬，也可以指导深度学习毕设、大作业等。
--------------->qq------------
           3582584734
------------------------------
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c7e9309a04bd6f22ae3f1138149f65ea.png)


