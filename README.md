# Tile-object-detection
将原数据集解压

将coco_annotation文件夹复制到tile_round1_train_20201231目录下：
```
tile_round1_train_20201231/
  coco_annotation/
    instances_train2017.json
    instances_val2017.json
  train_imgs/
  ...
```

修改`SplitOnlyImage.py` [第240行](https://github.com/HaoIrving/Tile-object-detection/blob/1a55970c7ac895c0ed5a1e57192f3fd4591eb2d3/SplitOnlyImage.py#L240)

运行命令`python SplitOnlyImage.py`

运行之后的文件目录如下：
```
tile_round1_train_20201231/
  coco_annotation/
  train_imgs/
  coco/
    annotations/
    train2017/
    val2017/
```

大尺寸的框会被过滤掉，对应于原图的标注会被记录下来备用，保存在coco_annotation文件夹
