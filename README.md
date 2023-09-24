# Introduction
This repository is a part of our team's submission for the ML4Earth Physics-aware 2023 hackathon.

![Screenshot (1371)](https://github.com/lisahligono/ML4Earth2023_Physics-aware/assets/72496335/1498bcb5-ab11-4bfe-a924-45462363b670)


## Task Overview

In this hackathon, our goal was to develop a robust flood modeling framework capable of performing at large scales using input and ground truth data from the Pakistan flood in 2022. We focused on applying supervised machine learning methods, specifically utilizing the Linear Regression for Neural Networks model, to solve the 2-D shallow water equations. Our ultimate objective was to create a flood forecast model that can provide valuable insights and predictions in the event of future floods in Pakistan.

## Team

Meet our team members:

- Parinda Pannoon (GitHub:https://github.com/parindapannoon)
- Yanika Dontong (GitHub: https://github.com/YanikaD)
- Lisah Ligono (GitHub: https://github.com/lisahligono)

<h2>Study area</h2>
In the summer monsoon season of 2022, Pakistan experienced a devastating flood event. This flood event impacted approximately one-third of Pakistan's vast population, resulting in the displacement of around 32 million individuals and tragically causing the loss of 1,486 lives, including 530 children. The economic toll of this disaster has been estimated at exceeding $30 billion. The study area encompasses the regions in Pakistan most severely affected by the flood. The Indus River basin, a critical drainage system, plays a pivotal role in this study area's hydrology. (Adapted from https://github.com/zhu-xlab/ML4Earth-Hackathon-2023)

![Screenshot (1373)](https://github.com/lisahligono/ML4Earth2023_Physics-aware/assets/72496335/4e26d4ec-064e-4e91-a788-1f9b1b0c764e)


<h2>Methodology</h2>
<b>Data Preprocessing </b>

**DEM**

Load data from tiff file then selected the values that less than -2,000 and clean missing data.
Precipitation
Load data from tiff file in ordering by name. After that, down sampling the image by 16 to get the image of size 881x440 and unsqueeze to 1x1x881x440

**Manning**

Load data from numpy file then selected the down sampling the image by 16 and unsqueeze the data to be the size of 1x1x881x440

**Trainning and Validation data**

Load data from of tiff files from training and validation path and make the first layer as a ground height. After that, generate grid x, y and t and put them together in tensor list in the order of x, y, t and initial height

**Data for input layer**

Concatenate the tensor of t, dem,manning,rain and height together in the shape of 1x16x881x440x5 


**Model**

The model consist of 3 Linear layers, 4 SpectralConv3d, Modulelist of [5 Conv1d] in the following order

(fc0): Linear(in_features=8, out_features=16, bias=True) 
(sp_convs): ModuleList( (0-3): 4 x SpectralConv3d() ) 
(ws): ModuleList( 

(0): Conv1d(16, 24, kernel_size=(1,), stride=(1,)) 

(1-2): 2 x Conv1d(24, 24, kernel_size=(1,), stride=(1,)) 

(3): Conv1d(24, 32, kernel_size=(1,), stride=(1,)) 

(4): Conv1d(32, 32, kernel_size=(1,), stride=(1,)) ) 

(fc1): Linear(in_features=32, out_features=128, bias=True) 
(fc2): Linear(in_features=128, out_features=1, bias=True)

Which these layers have the following description:
1.	Linear Layer
this layer is to transform data to a linear form: $$ y=xAt+b $$
which input is 
$$ (*,Hin) where  ∗ means any number of dimensions including none and Hin=in_features. $$
and output is
$$ (*,Hout) where  all but the last dimension are the same shapes as the input and Hout=out_features$$
2.	SpectralConv3d Layer
	this layer is applied 3D fourier transform to find Fourier coefficients:
$$ e-i(r1q1+r2q2+r3q3)e-a2(r12+r22+r32)dr1dr2dr3= (a)1.5e-q12+q22+q324a$$
which input is $$(in_channels, out_channels, modes1, modes2, modes3)$$
and output is  
 $$(Fourier_coefficients1,Fourier_coefficients2,Fourier_coefficients3)$$


3.	Conv1d Layer
	this layer is applied to 1D convolution over an input signal composed of several input planes:$$ out(Ni ,Coutj) = bias(Coutj) + k=0Cin-1weight(Cout ,k)*input(Ni ,k) $$
$$where * is the valid cross-correlation operator, N is a Batch size, C is a number of channels and
L is a length of signal sequence$$
which input layer’s size is $$(Ni,Cin,L)$$
and output layer’s size is $$(N,Cout ,Lout)$$




<h2>Results</h2>

<h2>Limitations</h2>

<h2>References</h2>
