## train
- python train.py --multi-scale --img-size 320 --multi-scale --batch-size 32

## reference
- https://github.com/ultralytics/yolov3.git

## 22修改
- python train22.py 训练非量化模型时不用加其他参数
- train22.py 80行改使用的模型，初始训练使用UltraNet_Float或RepVGG
- QAT时把模型改为量化模型例如UltraNet_ismart，并且用--weights载入全精度权重
- 测试精度使用python test.py --weights <权重路径> --model <模型类名称>

### 注意
- 余弦学习率衰减最好跑满全部epoch
- RepVGG全精度训练后量化训练精度更好
- quant_dorefa.py 第18行改权重量化为[-8,7]还是[-7,7]
