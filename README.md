# SART-GAN-for-CT-reconstruction



Readme
=
This project is from Ruiwen Xing’s master thesis: Deep Learning Based CT Image Reconstruction.
 
The project contains following code file:
Main.py                                    the start point
Sinogram.py                            create sinogram from tomography image
Reconstruction.py                    reconstruction process by SART algorithm
imgPainter.py                          output image file
imgFormatConvert.py              adjust shape of image tensor
imgEvaluation.py                     calculate PSNR & SSIM value between two image
NNmodel.py                            create different neural net model
NNstructure.py                        inner structures of neural networks
NNstructure_xxxx.py                 inner structures of other neural networks
Parameters.py                          handle hyper parameters
SAGAN_ops.py                        support block from original SAGAN
SAGAN_util.py                         support block from original SAGAN
Debug.py                                 some functions useful in debug process
createTrainset.py                      to generate noisy or incomplete CT images from clean, high dose CT images
dataLoader.py                          load data
imgTrainset.py                         create paired image trainset



Prepare Dataset
=
In this project we use CT images from cancer image archive.
CT images are created under standard dose scan.
However, this project needs paired image dataset
So, we use ASTRA toolbox to simulate different CT scan and generate low-dose, sparse view, and limited angle sinogram (original scan data)
And then, we use standard SART algorithm to reconstruct these sinogram to CT image.
Because of different setting during simulated scan (low-dose, sparse view, and limited angle), we loss some data in these processes. And when we reconstruct them, we will get noisy or incomplete CT images. These noisy or incomplete images will be paired with original normal-dose clean CT images to train our neural network. 

>$ python createTrainset.py --func [normal or full: full contains data augmentation process] --inFile [input folder path] --outFile [output folder path] --dataType [png or flt] --maxImg [maximum generated image number] --option [sinogram generate option] --iterNum [number of iterations in SART reconstruction] --ns [subset number in SART algorithm] --handelNum [number of images to handle at each time] --startFrom [number of image to start from input folder]


* data augmentation by rotate CT image
* for --option, currently we support "low-dose_1e4", "low-dose_1e5", "low-dose_1e6", "low-dose_2e5", "sparse_view_450", "sparse_view_180", "sparse_view_100", "sparse_view_60", "sparse_view_50", "sparse_view_40", "limited_angle_160", "limited_angle_140", "limited_angle_120", "limited_angle_100"
detail of these options can be seen in parameters.py

For example:
>$ python createTrainset.py \
 --func full \
 --inFile “../NDCT” \
 --outFile “../uiharuDataset/limitedAngle120ns50it20” \
 --dataType png \
 --option limited_angle_140 \
 --iterNum 20 \
 --ns 50 \
 --handelNum 10 \
 --startFrom 0



Prepare environment
=

In this project we use anaconda to create our python environment.
Download anaconda installer from: https://www.anaconda.com/download/#linux
To activate anaconda:
>$ conda activate
To create anaconda environment abc:
>$ conda create --name abc
To activate anaconda environment abc:
>$ conda activate abc
(abc)>$
To install numpy package in conda environment abc:
(abc)>$ conda install numpy
To clone an environment:
>$ conda create –name myclone –clone myenv
To create an environment from yml file
>$ conda env create -f environment.yml
To remove an environment:
>$ conda remove --name myenv --all
To deactivate conda
>$ conda activate
 
See https://docs.conda.io and https://docs.anaconda.com/anaconda/user-guide/getting-started/ for more information.
 
Before run our code, the following python packages need to be installed:
Astra-toolbox               v 1.8.3
Cudatoolkit                 v 8.0
CUDNN                       v 7.1.3
Keras                       v 2.2.4
Matplotlib                  v 3.1.1
Numpy                       v 1.14.2
Pillow                      v 6.0.0
Tensorflow                  v 1.10.0

To install astra toolbox, see http://www.astra-toolbox.com/

We also provide a uiharu-k.yml environment for you.




Training process
=

For environment preparation, please see prepare environment.txt
Some python packages need to be installed
We also provide an environment in uiharu-k.yml

To start a training process, type:

>$ python main.py --function trainNN \
--cleanTrainset [trainset clean img path] \
--cleanTrainsetDataType [png or flt] \
--noisyTrainset [trainset noisy img path] \
--noisyTrainsetDataType [png or flt] \
--cleanTestset [validation set clean img path] \
--cleanTestsetDataType [png or flt] \
--noisyTestset [validation set noisy img path] \
--noisyTestsetDataType [png or flt] \
--checkpointFolder [path to save checkpoint] \
--batchSize [batch size] \
--NNtype [neural net structure in NNmodel.py]

Default hyperparameters are saved in parameters.py
Neural networks type can be seen in NNmodel.py

For example:
>$ python main.py --function trainNN \
--cleanTrainset ../NDCT \
--cleanTrainsetDataType png \
--noisyTrainset “../uiharuDataset/sparseView100ns50iter20;../uiharuDataset/low-dose1e5ns50iter20;../uiharuDataset/limitedAngle140ns50iter20” \
--noisyTrainsetDataType png \
--cleanTestset “../NDCTval” \
--cleanTestsetDataType png \
--noisyTestset “../NDCTval” \
--noisyTestsetDataType png \
--checkpointFolder “./checkpoint” \
--batchSize 1 \
--NNtype NN721



Test process
=

To test the model, type:
>$ python main.py --function autoRecon \
--inputFolder [path of ground truth CT images] \
--dataType [png or flt] \
--sinoFolder [path to save sinogram] \
--noiseOption [scenarios, e.g. limited_angle_120] \
--mnnOrder [order to reconstruct image, e.g. ‘sart5|(gan)./checkpoint|sart20|return’] \
--outputFolder [path to save reconstructed image] \
--ns [subset number of SART]

* --mnnOrder
In this project we designed a simple command system to describe reconstruction process
This command system is defined by reconstruction.py
Here’s the rule:
This system contains three kinds of command, commands are connected by pipeline “|”
So a full mnnOrder should be: “command_1|command_2|command_3|command_4…….|command_n”

The first kind of command is sart command, it includes:
"sart"
Several  iterations of the SART algorithm. “sart15ns20” means 15 iterations of SART with subsetNumber=20.
"TVsart"
Several  iterations of the SART algorithm and optimize the result by reducing total variation. “TVsart15ns20” means 15 iterations of SART with subsetNumber=20, and optimization after each iteration.




The second kind of command is neural net command, it is formed by a neural net type name and a path that save the checkpoint of a trained neural net. It looks like:
(cycleGAN)./checkpoint/cycleGAN

The third kind of command is return command. It simply returns the result.

Thus, a full mnnOrder could be:
“sart30|(simpleGAN)../models/simpleGAN|sart20|(NN721)../models/nn721|sart20|(NN721)../models/nn721|sart20|(NN721)../models/nn721|return”

For example:
>$ python main.py --function autoRecon \
--inputFolder ../NDCTtest \
--dataType png \
--sinoFolder ./sinogram \
--noiseOption limited_angle_120 \
--mnnOrder “sart30|(NN721)../models/simpleGAN|sart20|(NN721)../models/nn721|sart20|(NN721)../models/nn721|sart20|(NN721)../models/nn721|return” \
--outputFolder ./result \
--ns 20
