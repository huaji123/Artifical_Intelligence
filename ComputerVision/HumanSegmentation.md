# 人像分割

人像分割任务旨在识别图像中的人体轮廓，与背景进行分离，返回分割后的二值图、灰度图、前景人像图，适应多个人体、复杂背景、各类人体姿态。
可应用于人像扣图、人体特效和影视后期处理等场景。

## step1 PaddleSeg安装

      # 解压从PaddleSeg Github仓库下载好的压缩包
      
      !unzip -o PaddleSeg.zip
      
      # 运行脚本需在PaddleSeg目录下
      
      %cd PaddleSeg
      
      # 安装所需依赖项
      
      !pip install -r requirements.txt
      
      若想在自己的电脑环境中安装paddleseg，运行如下命令：
      
      # 从PaddleSeg的github仓库下载代码
      
      git clone https://github.com/PaddlePaddle/PaddleSeg.git

      # 运行PaddleSeg的程序需在PaddleSeg目录下
      
      cd PaddleSeg/

      # 安装所需依赖项
      
      pip install -r requirements.txt
      
## step2 模型简介

### DeepLabv3+ 介绍

      DeepLabv3+是DeepLab语义分割系列网络的最新作，其前作有 DeepLabv1，DeepLabv2, DeepLabv3, 在最新作中，DeepLab的作者通过encoder-decoder进行多尺度信息的融合，同时保留了原来的空洞卷积和ASSP层， 其骨干网络使用了Xception模型，提高了语义分割的健壮性和运行速率，在 PASCAL VOC 2012 dataset取得新的state-of-art performance，89.0mIOU。

### Xception

      Xception是DeepLabv3+原始实现的backbone网络，兼顾了精度和性能，适用于服务端部署。

## step3 预训练模型下载

      # 下载预训练模型并放入./pretrained_model目录下
      
      %cd /home/aistudio/PaddleSeg/pretrained_model/ 
      
      !wget https://paddleseg.bj.bcebos.com/models/deeplabv3p_xception65_humanseg.tgz 
      
      !tar -xf deeplabv3p_xception65_humanseg.tgz  
      
      %cd ..

## step4 数据准备

      # 将测试数据集放入./dataset目录下
      
      !cp ~/data/data10908/humanseg.zip dataset/
      
      !unzip -o dataset/humanseg.zip -d dataset/

## step5 模型预测和可视化

      pdseg/vis.py是模型预测和可视化的脚本。

      模型配置说明：

      PaddleSeg中关于模型的配置记录在yaml文件里。

      configs目录存放各个模型的yaml文件。

      主要参数：
      
          --cfg 指定yaml配置文件的路径
          
          --vis_dir 指定预测结果图片存放位置
          
          --use-gpu 是否启用gpu
          
          DATASET.DATA_DIR 数据集存放位置
          
          DATASET.VIS_FILE_LIST 测试集列表
          
          TEST.TEST_MODEL 测试模型路径
          
        #!/bin/bash

        # 将配置文件humanseg.yaml复制到configs目录下
        
        !cp ~/work/humanseg.yaml configs/
        
        # 模型预测
        
        # Note: 若你没有gpu计算资源，只需要在以下脚本中删除参数`--use_gpu`重新运行即可。
        
        !python ./pdseg/vis.py  --cfg ./configs/humanseg.yaml \
        
                                --vis_dir ./visual \
                                
                                DATASET.DATA_DIR "dataset/humanseg" \
                                
                                DATASET.VIS_FILE_LIST "dataset/humanseg/test_list.txt" \
                                
                                TEST.TEST_MODEL "pretrained_model/deeplabv3p_xception65_humanseg"

## step6 显示分割结果

      import matplotlib.pyplot as plt

      # 定义显示函数

      def display(img_dir):
      
          plt.figure(figsize=(15, 15))

          title = ['Input Image', 'Predicted Mask']

          for i in range(len(title)):
          
              plt.subplot(1, len(img_dir), i+1)
              
              plt.title(title[i])
              
              img = plt.imread(img_dir[i])
              
              plt.imshow(img)
              
              plt.axis('off')
              
          plt.show()

      # 显示分割效果
      
      # 注：仅显示其中一张图片的效果。    
      
      image_dir = "dataset/humanseg/aa6b34b24414bafa7fab8393239c793587513ce6.jpg"
      
      mask_dir = "visual/aa6b34b24414bafa7fab8393239c793587513ce6.png"
      
      imgs = [image_dir, mask_dir]
      
      display(imgs)
