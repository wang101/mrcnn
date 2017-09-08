import cv2
import os
import copy
import pdb
import numpy as np


def watershed(mname,class_name,img_shape):
    mask_img = cv2.imread(mname)
    gray_ = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    gray = np.uint8(np.round(cv2.resize(np.float32(gray_),(img_shape[1],img_shape[0]))))

    dist_transform = cv2.distanceTransform(gray, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.05 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    # plt.imshow(markers)
    # plt.show()

    boxes = []
    masks = []

    instance_label = 1

    while True:
        one_marker = np.where(markers == instance_label, 1, 0)
        one_marker = np.uint8(one_marker)
        one_marker_contour = copy.deepcopy(one_marker)
        if np.max(one_marker) == 0:
            break
        # pdb.set_trace()
        cimg, contours, hierarchy = cv2.findContours(one_marker_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert len(contours) == 1
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        x1 = max(1, x - 2)
        y1 = max(1, y - 2)
        x2 = min(x + w + 2, mask_img.shape[1])
        y2 = min(y + h + 2, mask_img.shape[0])

        #x1 = np.int32(rows*1.0*x1/mask_rows)
        #x2 = np.int32(rows*1.0*x2/mask_rows)
        #y1 = np.int32(cols*1.0*y1/mask_cols)
        #y2 = np.int32(cols*1.0*y2/mask_cols)
        boxes.append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})
        print(mname,boxes[-1])
        masks.append(copy.deepcopy(one_marker[y1:y2, x1:x2]))
        # plt.imshow(one_marker)
        # new = cv2.rectangle(one_marker*100,(x1,y1),(x2,y2),(0,255,0),2)
        # plt.imshow(new)
        #plt.imshow(masks[-1])
        #plt.show()
        instance_label = instance_label + 1
    return boxes, masks

def get_data(input_path):
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    visualise = True

    with open(input_path,'r') as f:

        print('Parsing annotation files')

        #for line in f:
        #    line_split = line.strip().split('####')
        #    (filename,maskname,class_name) = line_split
        fpath = '/afs/crc.nd.edu/user/h/hwang21/work/keras_my/data/img_train/'
        mpath = '/afs/crc.nd.edu/user/h/hwang21/work/keras_my/data/mask_train/'
        for fname in os.listdir(fpath):
            mname = os.path.splitext(fname.split('/')[-1])[0]+'.png'
            filename = os.path.join(fpath,fname)
            maskname = os.path.join(mpath,mname)
            class_name = 'node'

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print('Found class name with special name bg. Will be treated as a background region (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                print(filename)
                all_imgs[filename] = {}
                img = cv2.imread(filename)


                (rows,cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                all_imgs[filename]['width'] = cols
                all_imgs[filename]['height'] = rows
                boxes,masks = watershed(maskname,class_name,(rows,cols))
                all_imgs[filename]['bboxes'] = boxes
                all_imgs[filename]['masks'] = masks
                if np.random.randint(0,6) > 0:
                    all_imgs[filename]['imageset'] = 'trainval'
                else:
                    all_imgs[filename]['imageset'] = 'test'

            #all_imgs[filename]['bboxes'].append({'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})


        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping)-1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch
    print(class_mapping)
    print('class_mapping')
    return all_data, classes_count, class_mapping


