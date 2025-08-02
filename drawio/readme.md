# 复刻DiffDis图表为drawio XML文件

## Notes
- 用户要求将所提供的流程图高度还原为drawio可导入的xml文件。
- 文件夹需命名为drawio，xml文件命名为diffdis.xml。
- 图表细节（结构、连线、图例、说明、配色、标注等）需尽量贴合原图。
- 用户使用中文沟通。

## Task List
- [ ] 新建drawio文件夹
- [ ] 创建diffdis.xml文件
- [ ] 复刻原图的主要结构（VAE、UNet、Batch-Discriminative Embedding等模块）
- [ ] 添加流程连线与箭头
- [ ] 补充图例与说明（如Batch-wise Concat、Channel-wise Concat、Injector等）
- [ ] 检查整体还原度，调整细节

## Current Goal
新建drawio文件夹并准备复刻图表

新建一个文件夹，名为drawio，把这张图以xml格式复刻出来名为diffdis.xml，我要导入到drawio中，尽量复合原图，这对我很重要。