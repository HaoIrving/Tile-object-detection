import os
import numpy as np
import cv2
import copy
# import dota_utils as util
from multiprocessing import Pool
from functools import partial
from pycocotools.coco import COCO
import json

def iou(BBGT, imgRect):
    """
    并不是真正的iou。计算每个BBGT和图像块所在矩形区域的交与BBGT本身的的面积之比，比值范围：0~1
    输入：BBGT：n个标注框，大小为n*4,每个标注框表示为[xmin,ymin,xmax,ymax]，类型为np.array
          imgRect：裁剪的图像块在原图上的位置，表示为[xmin,ymin,xmax,ymax]，类型为np.array
    返回：每个标注框与图像块的iou（并不是真正的iou），返回大小n,类型为np.array
    """
    left_top = np.maximum(BBGT[:, :2], imgRect[:2])
    right_bottom = np.minimum(BBGT[:, 2:], imgRect[2:])
    wh = np.maximum(right_bottom-left_top, 0)
    inter_area = wh[:, 0]*wh[:, 1]
    iou = inter_area/((BBGT[:, 2]-BBGT[:, 0])*(BBGT[:, 3]-BBGT[:, 1]))
    return iou


class splitbase():
    def __init__(self,
                 srcpath,
                 dstpath,
                 srcpath_ann,
                 dstpath_ann,
                 split_mode='train',
                 image_id=1,
                 box_id=1,
                 gap=100,
                 subsize=1024,
                 ext='.jpg'):
        self.srcpath = srcpath
        self.dstpath = dstpath
        self.srcpath_ann = srcpath_ann
        self.dstpath_ann = dstpath_ann
        
        self.split_mode = split_mode
        self.ann_file = f'instances_{split_mode}2017.json'
        self.src_ann_file = os.path.join(srcpath_ann, self.ann_file)
        _COCO = COCO(self.src_ann_file)
        self._COCO = _COCO
        self.categories = _COCO.dataset['categories']
        indexes = _COCO.getImgIds()
        self.image_indexes = indexes

        self.gap = gap 
        self.subsize = subsize
        self.slide = self.subsize - self.gap
        self.ext = ext
        
        self.image_id = image_id
        self.box_id = box_id

    def reset_json(self):
        self.json_dict = {"images":[], "type": "instances", "annotations": [], "categories": self.categories}
        # 大尺寸的框会被过滤掉，记录下来备用
        self.losed_json_dict = {"images":[], "type": "instances", "annotations": [], "categories": self.categories}
    
    def save_json(self):
        new_ann_file = os.path.join(self.dstpath_ann , self.ann_file)
        print('Writing {} json to {}'.format(self.split_mode, new_ann_file))
        with open(new_ann_file, 'w') as fid:
            json.dump(self.json_dict, fid)
    
    def save_losed_json(self): 
        losed_ann_file = os.path.join(self.srcpath_ann , f'instances_losed_{self.split_mode}2017.json')  # saved at coco_annotation folder
        print('Writing losed {} json to {}'.format(self.split_mode, losed_ann_file))
        with open(losed_ann_file, 'w') as fid_losed:
            json.dump(self.losed_json_dict, fid_losed)

    def saveimagepatches(self, img, subimgname, left, up, ext='.jpg'):
        subimg = copy.deepcopy(img[up: (up + self.subsize), left: (left + self.subsize)])
        outdir = os.path.join(self.dstpath, subimgname + ext)
        cv2.imwrite(outdir, subimg)

    def _annotation_from_index(self, index, _COCO):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = _COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = _COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = _COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        for ix, obj in enumerate(objs):
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = obj['category_id']

        return res, objs

    def SplitSingle(self, index, rate, extent, iou_thresh):
        print('\nprocessing No. {}'.format(index))
        img_file = self._COCO.loadImgs(index)[0]['file_name']
        image_path = os.path.join(self.srcpath, img_file)
        name = img_file[:-4]
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)

        img = cv2.imread(image_path)
        assert np.shape(img) != ()

        if (rate != 1):
            resizeimg = cv2.resize(img, None, fx=rate, fy=rate, interpolation=cv2.INTER_CUBIC)
        else:
            resizeimg = img
        outbasename = name + '__' + str(rate) + '__'

        width = np.shape(resizeimg)[1]
        height = np.shape(resizeimg)[0]

        BBGT, obj_anns = self._annotation_from_index(index, self._COCO)

        dummy_mask = np.zeros(len(BBGT)).astype(bool)
        left, up = 0, 0
        while (left < width):
            if (left + self.subsize >= width):
                left = max(width - self.subsize, 0)
            up = 0
            while (up < height):
                if (up + self.subsize >= height):
                    up = max(height - self.subsize, 0)
                
                imgrect = np.array([left, up, left + self.subsize, up + self.subsize]).astype('float32')
                ious = iou(BBGT[:, : 4].astype('float32'), imgrect)
                mask = ious >= iou_thresh
                BBpatch = BBGT[mask]
                restrict_filter = BBGT[ious == 1]  # sub image 中必须包含一个完整box, 同时边缘上被切割到的box，若和img patch的iou小于0.8，则舍弃这个box的标注
                if len(BBpatch) > 0 and len(restrict_filter) > 0:
                    dummy_mask += mask
                    # print("before crop : %d" % len(BBGT))
                    # print("ious:")
                    # print(ious)
                    # print("after crop: %d" % len(BBpatch))

                    # save image patch 
                    subimgname = outbasename + str(left) + '__' + str(up)
                    self.saveimagepatches(resizeimg, subimgname, left, up)
                    
                    # save new ann to self.json 
                    filename = subimgname + extent
                    image_id = self.image_id
                    print('image_id: {}'.format(self.image_id))
                    self.image_id += 1
                    image = {'file_name': filename, 'height': self.subsize, 'width': self.subsize, 'id': image_id}
                    self.json_dict['images'].append(image)

                    # calculate new box and save to self.json
                    for bb in BBpatch:
                        xmin, ymin, xmax, ymax, category_id = int(bb[0]) - left, int(bb[1]) - up, int(bb[2]) - left, int(bb[3]) - up, int(bb[4])
                        # 将0.8 < iou < 1 的框的坐标限定在img patch内
                        if xmin < 0 or ymin < 0 or xmax > self.subsize or ymax > self.subsize:
                            print('found a non restrict box: {}, refining it within the image patch.'.format([xmin, ymin, xmax, ymax]))
                            xmin = max(xmin, 0)
                            ymin = max(ymin, 0)
                            xmax = min(xmax, self.subsize)
                            ymax = min(ymax, self.subsize)
                        
                        o_width = xmax - xmin + 1
                        o_height = ymax - ymin + 1
                        box_id = self.box_id
                        # print('box_id: {}'.format(self.box_id))
                        self.box_id += 1
                        ann = {
                            'area': o_width*o_height, 'iscrowd': 0, 'image_id': image_id, 
                            'bbox':[xmin, ymin, o_width, o_height],
                            'category_id': category_id, 'id': box_id, 'ignore': 0,
                            'segmentation': []  # 将segmentation标注丢弃
                            }
                        self.json_dict['annotations'].append(ann)

                if (up + self.subsize >= height):
                    break
                else:
                    up = up + self.slide
            if (left + self.subsize >= width):
                break
            else:
                left = left + self.slide
        if len(BBGT) - dummy_mask.sum() > 0:
            print('losed boxes num: {}'.format(len(BBGT) - dummy_mask.sum()))
            not_keep = ~ dummy_mask

            losed_img = self._COCO.imgs[index]
            self.losed_json_dict['images'].append(losed_img)
            
            losed_ann_indexes = np.argwhere(not_keep == 1)[0]
            for idx in losed_ann_indexes:
                losed_ann = obj_anns[idx]
                losed_ann.pop('clean_bbox')
                self.losed_json_dict['annotations'].append(losed_ann)
                print('losed boxes shape:')
                print('box: {}, category_id: {}'.format(losed_ann['bbox'], losed_ann['category_id']))



    def splitdata(self, rate, iou_thresh):
        self.reset_json()
        
        # self.image_indexes = [41]
        for index in self.image_indexes:
            self.SplitSingle(index, rate, self.ext, iou_thresh)

        self.save_json()
        self.save_losed_json()
        return 

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

if __name__ == '__main__':
    root = '/media/sun/DATA/tile_round1_train_20201231'  # change this line
    srcpath_img       = os.path.join(root, 'train_imgs')
    srcpath_ann       = os.path.join(root, 'coco_annotation')
    coco_root         = os.path.join(root, 'coco')
    dstpath_img_train = os.path.join(coco_root, 'train2017')
    dstpath_img_val   = os.path.join(coco_root, 'val2017')
    dstpath_ann       = os.path.join(coco_root, 'annotations')
    for path in [coco_root, dstpath_img_train, dstpath_img_val, dstpath_ann]:
        if not os.path.exists(path):
            os.mkdir(path)

    iou_thresh = 0.8  # 保留和裁剪后图片iou 大于0.8的box 标注
    
    # 裁剪训练集
    split_train = splitbase(srcpath_img,
                            dstpath_img_train,
                            srcpath_ann,
                            dstpath_ann,
                            split_mode='train',
                            image_id=1, 
                            box_id=1,
                            gap=100,  # 重叠区域大小，重叠太多的话，会产生很多重复的框
                            subsize=500)  # 裁剪出的小图尺寸
    split_train.splitdata(1, iou_thresh)

    image_id = split_train.image_id
    box_id = split_train.box_id
    print('In train set, after spliting, total num of images: {}, boxes: {}'.format(image_id, box_id))    
    
    # 裁剪验证集
    split_val = splitbase(srcpath_img,
                            dstpath_img_val,
                            srcpath_ann,
                            dstpath_ann,
                            split_mode='val',
                            image_id=image_id + 1, 
                            box_id=box_id + 1,
                            gap=100,  
                            subsize=500)
    split_val.splitdata(1, iou_thresh)

    image_id = split_val.image_id
    box_id = split_val.box_id
    print('In train and val set, after spliting, total num of images: {}, boxes: {}'.format(image_id, box_id))

