# DCTNet

Codes for ***Discrete Cosine Transform Network for Guided Depth Map Super-Resolution (CVPR 2022 Oral)***

[Zixiang Zhao](https://zhaozixiang1228.github.io/), [Jiangshe Zhang](http://gr.xjtu.edu.cn/web/jszhang), [Shuang Xu](http://shuangxu96.github.io/), [Zudi Lin](https://zudi-lin.github.io/) and [Hanspeter Pfister](https://vcg.seas.harvard.edu/people).

-[*[Paper]*](https://openaccess.thecvf.com/content/CVPR2022/html/Zhao_Discrete_Cosine_Transform_Network_for_Guided_Depth_Map_Super-Resolution_CVPR_2022_paper.html)
-[*[ArXiv]*](https://arxiv.org/abs/2104.06977)

## Citation

```
@InProceedings{Zhao_2022_CVPR,
    author    = {Zhao, Zixiang and Zhang, Jiangshe and Xu, Shuang and Lin, Zudi and Pfister, Hanspeter},
    title     = {Discrete Cosine Transform Network for Guided Depth Map Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5697-5707}
}
```

## Abstract
Guided depth super-resolution (GDSR) is an essential topic in multi-modal image processing, which reconstructs high-resolution (HR) depth maps from low-resolution ones collected with suboptimal conditions with the help of HR RGB images of the same scene. To solve the challenges in interpreting the working mechanism, extracting cross-modal features and RGB texture over-transferred, we propose a novel Discrete Cosine Transform Network (DCTNet) to alleviate the problems from three aspects. First, the Discrete Cosine Transform (DCT) module reconstructs the multi-channel HR depth features by using DCT to solve the channel-wise optimization problem derived from the image domain. Second, we introduce a semi-coupled feature extraction module that uses shared convolutional kernels to extract common information and private kernels to extract modality-specific information. Third, we employ an edge attention mechanism to highlight the contours informative for guided upsampling. Extensive quantitative and qualitative evaluations demonstrate the effectiveness of our DCTNet, which outperforms previous state-of-the-art methods with a relatively small number of parameters. 

## Usage

### Network Architecture

Our DCTNet is implemented in ``model.py``.

### Training

Pretrained model is available in ``'./models/DCTNet_4X.model'``, ``'./models/DCTNet_8X.model'``, ``'./models/DCTNet_16X.model'`` and ``'./models/DCTNet_RealScene.model'``, which are responsible for the tasks of upsampling factors of 4, 8, and 16, and the RGBDD real-world branch task. We train it on NYU v2 (1000 image pairs). In the training phase, all images are resize to 256x256.

If you want to re-train this net, you need to download the original dataset at [https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](), then use the same preprocessing as DKN [16] and FDSR [12] to get a training set like ``./data/NYU_Train_imgsize_256_scale_4.h5``(because the size of this dataset is 10+GB, we cannot upload it). Subsequently, you should run  ``'train.py'`` to retrain.

### Testing

The test images used in the paper have been stored in ``'./RawDatasets/Middlebury'``, ``'./RawDatasets/NYUDepthv2_Test'``, `'./RawDatasets/Lu'`, ``'./RawDatasets/RGBDD'`` and `'./RawDatasets/RGBDD_Test_Realscene'`, respectively.

Unfortunately, since the size of **NYU v2 dataset** is 600+MB and that of **RGBDD in real-world branch** is 100+MB, we only upload **three image pairs** from these two datasets respectively to prove the correctness of our codes. The other datasets contain all the test images.

If you want to inference with our DCTNet and obtain the RMSE results in our paper, please run ``'processing_testsets.py'`` and get the the processed test set in `'./DatasetsAfterProcessing/'`. Then run  ``'test.py'`` to test our method. 

If you use the complete test datasets, the testing results will be printed in the terminal:

```
==============================================
The testing RMSE results of Middlebury Dataset
     X4         X8         X16
----------------------------------------------
[1.09937036 2.04951119 4.19195414]
==============================================
==============================================
The testing RMSE results of NYU V2 Dataset    
     X4         X8         X16
----------------------------------------------
[1.59155273 3.16303039 5.84125805]
==============================================
==============================================
The testing RMSE results of Lu Dataset        
     X4         X8         X16
----------------------------------------------
[0.88223213 1.84769642 4.38759089]
==============================================
==============================================
The testing RMSE results of RGBDD Dataset
     X4         X8         X16
----------------------------------------------
[1.07670105 1.73648119 3.04929352]
==============================================
==============================================
The testing RMSE results in RealScene RGBDD
DCTNet in real-world branch
----------------------------------------------
tensor([7.3676])
==============================================
==============================================
The testing RMSE results in RealScene RGBDD
DCTNet* in real-world branch
----------------------------------------------
tensor([5.4326])
==============================================
```

The above output represents the results of DCTNet in Tab. 2 and Tab. 3 in our paper. The first four parts correspond to the results of the four testsets in Tab. 2, and the last two parts show the RMSE values of ``DCTNet`` and ``DCTNet*`` in Tab. 3.
