#! /usr/bin/env python
import os
import cv2
import argparse
from PIL import Image
from PIL import ImageEnhance
import numpy as np

from face_detection import face_detection
from face_points_detection import face_points_detection
from face_swap import warp_image_2d, warp_image_3d, mask_from_points, apply_mask, correct_colours, transformation_from_points


def select_face(im, r=10):
    faces = face_detection(im)

    if len(faces) == 0:
        print('Detect 0 Face !!!')
        exit(-1)

    if len(faces) == 1:
        bbox = faces[0]
    else:
        bbox = faces[0]
#        bbox = []
#        def click_on_face(event, x, y, flags, params):
#            if event != cv2.EVENT_LBUTTONDOWN:
#                return
#
#            for face in faces:
#                if face.left() < x < face.right() and face.top() < y < face.bottom():
#                    bbox.append(face)
#                    break
#        
#        im_copy = im.copy()
#        for face in faces:
#            # draw the face bounding box
#            cv2.rectangle(im_copy, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 1)
#        #cv2.imshow('Click the Face:', im_copy)
#        cv2.setMouseCallback('Click the Face:', click_on_face)
#        while len(bbox) == 0:
#            cv2.waitKey(1)
#        cv2.destroyAllWindows()
#        bbox = bbox[0]

    points = np.asarray(face_points_detection(im, bbox))
    
    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)
    
    x, y = max(0, left-r), max(0, top-r)
    w, h = min(right+r, im_h)-x, min(bottom+r, im_w)-y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y+h, x:x+w]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FaceSwapApp')
    parser.add_argument('--src', required=True, help='Path for source image')
    parser.add_argument('--dst', required=True, help='Path for target image')
    parser.add_argument('--out', required=True, help='Path for storing output images')
    parser.add_argument('--t', required=True, help='Path for storing output images')
    parser.add_argument('--warp_2d', default=False, action='store_true', help='2d or 3d warp')
    parser.add_argument('--correct_color', default=False, action='store_true', help='Correct color')
    parser.add_argument('--no_debug_window', default=False, action='store_true', help='Don\'t show debug window')
    args = parser.parse_args()

    # Read images
    SCALE_FACTOR = 1
    src_img = cv2.imread(args.src)
    src_img = cv2.resize(src_img, (src_img.shape[1] * SCALE_FACTOR,
                         src_img.shape[0] * SCALE_FACTOR))
    dst_img = cv2.imread(args.dst)
    dst_img = cv2.resize(dst_img, (dst_img.shape[1] * SCALE_FACTOR,
                         dst_img.shape[0] * SCALE_FACTOR))

    # Select src face
    src_points, src_shape, src_face = select_face(src_img)
    # Select dst face
    dst_points, dst_shape, dst_face = select_face(dst_img)

    h, w = dst_face.shape[:2]
    
    ### Warp Image
    if not args.warp_2d:
        ## 3d warp
        warped_src_face = warp_image_3d(src_face, src_points[:48], dst_points[:48], (h, w))
    else:
        ## 2d warp
        src_mask = mask_from_points(src_face.shape[:2], src_points)
        src_face = apply_mask(src_face, src_mask)
        # Correct Color for 2d warp
       # if args.correct_color:
        #    warped_dst_img = warp_image_3d(dst_face, dst_points[:48], src_points[:48], src_face.shape[:2])
        #    src_face = correct_colours(warped_dst_img, src_face, src_points)
        # Warp
        warped_src_face = warp_image_2d(src_face, transformation_from_points(dst_points, src_points), (h, w, 3))

    ## Mask for blending
    mask = mask_from_points((h, w), dst_points)
    mask_src = np.mean(warped_src_face, axis=2) > 0
    mask = np.asarray(mask*mask_src, dtype=np.uint8)
    #mask = (cv2.GaussianBlur(mask, (11, 11), 0) > 0) * 1.0
    #mask = cv2.GaussianBlur(mask, (11, 11), 0)
    #cv2.imshow("mask", mask)

    ## Correct color
    #if not args.warp_2d and args.correct_color:
    if  args.correct_color:
        warped_src_face = apply_mask(warped_src_face, mask)
        dst_face_masked = apply_mask(dst_face, mask)
        warped_src_face = correct_colours(dst_face_masked, warped_src_face, dst_points)
    
    ## Shrink the mask
    kernel = np.ones((10, 10), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    ##Poisson Blending
    r = cv2.boundingRect(mask)
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
    
    output = cv2.seamlessClone(warped_src_face, dst_face, mask, center, cv2.NORMAL_CLONE)

    x, y, w, h = dst_shape
    dst_img_cp = dst_img.copy()
    dst_img_cp[y:y+h, x:x+w] = output
    output = dst_img_cp

    dir_path = os.path.dirname(args.out)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(args.out, output)

#    image = Image.open(args.out)
#    enh_con = ImageEnhance.Contrast(image)
#    contrast =1.1
#    image_contrasted = enh_con.enhance(contrast)


#   enh_col = ImageEnhance.Color(image_contrasted)
#    color =1.1
#    image_contrasted = enh_col.enhance(color)

    
#    image_contrasted.save(args.out)

    t = args.t
    url = "https://crmdev.aiyongbao.com/data/img/m_"+t+".jpg" 
    print(url)

    ##For debug
    if not args.no_debug_window:
        cv2.imshow("From", dst_img)
        cv2.imshow("To", output)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()