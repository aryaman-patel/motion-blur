# Removing Motion Blur
This project aims to predict the blur kernel that can be used to deblur an images using Non-blind deconvolution. 

![image](https://github.com/aryaman-patel/motion-blur/assets/117113574/44367e43-8bfb-4b9b-bf2a-3c9087eaee9e)

## Reference work : 
> Gong, D., Yang, J., Liu, L., Zhang, Y., Reid, I., Shen, C., Hengel, A., & Shi, Q. (2017). From Motion Blur to Motion Flow: A Deep Learning Solution for Removing Heterogeneous Motion Blur. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR).

[Link to the Project Page](https://donggong1.github.io/blur2mflow)


## Instruction to project: 

The `dataset_old` folder includes an example dataset of an image from the BSD300 dataset. The images have been simulated using the [data generation code](https://github.com/donggong1/motion-flow-syn) from the project page to show the flow and the blur patterns. 

The `model.py` contains the entire FCN architecture as illustrated in the paper: 

![image](https://github.com/aryaman-patel/motion-blur/assets/117113574/017d21e4-8502-480c-91b1-765d8ec67162)

The model can also be trained on the included dataset.

The `preprocess_data.py` folder contains the the code to resize all the images to a standard size of `300x460` for the model to process them. However, the example dataset has already been preprocessed to this size. 

In order to see the results on the example image, run the `flow_field.py` file that produces the optical flow vector field, of the results `result_flowfield.mat` obtained by running inference on an example image `8143_021_blurryimg.png`. 

## Example Result:
![Screen Shot 2023-09-27 at 10 20 58 PM](https://github.com/aryaman-patel/motion-blur/assets/117113574/16445e9f-43e6-44b6-b969-bab2524fcb7e)
