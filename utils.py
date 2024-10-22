import cv2
import numpy as np
import os

def save_results(input_img, gt_data,density_map,output_dir, fname='results.png'):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[1],input_img.shape[0]))
        gt_data = cv2.resize(gt_data, (input_img.shape[1],input_img.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    cv2.imwrite(os.path.join(output_dir,fname),result_img)
def save_tmp(image,gt,den,outdir,fname='results.png'):
    image = image[0,:,:,:].transpose(1,2,0)
    image_g = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
    den = den[0,0,:,:]
    
    den_map = cv2.resize(den, (image_g.shape[1],image_g.shape[0]))
    den_map = 255*den_map/np.max(den_map)
    
    gt_data = cv2.resize(gt, (image_g.shape[1],image_g.shape[0]))
    gt_data = 255*gt_data/np.max(gt_data)
    
    result_img = np.hstack((image_g,gt_data,den_map))
    cv2.imwrite(os.path.join(outdir,fname),result_img)

def save_density_map(density_map,output_dir, fname='results.png'):    
    density_map = 255*density_map/np.max(density_map)
    density_map= density_map[0][0]
    cv2.imwrite(os.path.join(output_dir,fname),density_map)
    
def display_results(input_img, gt_data,density_map):
    input_img = input_img[0][0]
    gt_data = 255*gt_data/np.max(gt_data)
    density_map = 255*density_map/np.max(density_map)
    gt_data = gt_data[0][0]
    density_map= density_map[0][0]
    if density_map.shape[1] != input_img.shape[1]:
         input_img = cv2.resize(input_img, (density_map.shape[1],density_map.shape[0]))
    result_img = np.hstack((input_img,gt_data,density_map))
    result_img  = result_img.astype(np.uint8, copy=False)
    cv2.imshow('Result', result_img)
    cv2.waitKey(0)