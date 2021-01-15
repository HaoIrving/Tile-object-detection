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

大尺寸的框会被过滤掉，将对应于原图的标注记录下来备用

