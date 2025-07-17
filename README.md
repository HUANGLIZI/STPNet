# STPNet
STPNet: Scale-aware Text Prompt Network for Medical Image Segmentation. [Arxiv](https://arxiv.org/abs/2504.01561) [IEEE Xplore](https://ieeexplore.ieee.org/document/11015682) (IEEE TIP 2025)

### Requirements
This repository is based on PyTorch 1.12.1, CUDA 11.4 and Python 3.8.18; All experiments in our paper were conducted on a single NVIDIA A100 GPU.

Install from the ```requirements.txt``` using:
```angular2html
pip install -r requirements.txt
```
### Usage

### 1. Data Preparation
#### 1.1. QaTa-COVID and MosMedData+ Datasets
The original data can be downloaded in following links:
* COVID-Xray Dataset - [Link (Original)](https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset)

* COVID-CT Dataset - [Link (Original)](https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset)

* Kvasir-SEG Dataset - [Link (Original)](https://datasets.simula.no/downloads/kvasir-seg.zip)

 *(Note: The text annotation of COVID-Xray train and val datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDkm5jsTw5dX_RpuRr?e=uaZq6W).
  The partition of train set and val set of COVID-Xray dataset [download link](https://1drv.ms/f/c/c3143e7c85766728/QihndoV8PhQggMO2rwAAAAAADo5kj33mUee33g).
  The text annotation of COVID-Xray test dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDkj1vvvLt2jDCHqiM?e=954uDF).)*
  
*(Note: The text annotation of COVID-CT train dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDguIIKCRfYB9Z0NL8Dw?e=8rj6rY).
The text annotation of COVID-CT val dataset [download link](https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMGsQAAAAAAtAgZiRQFYfsAjw).
The text annotation of COVID-CT test dataset [download link](https://1drv.ms/x/c/c3143e7c85766728/QShndoV8PhQggMMHsQAAAAAAdHkwXMxGlgU9Tg).)*

#### 1.2. Format Preparation

Then prepare the datasets in the following format for easy use of the code:

```angular2html
├── datasets
    ├── COVID-Xray
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    |   |   ├── Train_text_sentence.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
    └── COVID-CT
        ├── Test_Folder
        │   ├── img
        │   └── labelcol
        ├── Train_Folder
        |   ├── Train_text_sentence.xlsx
        │   ├── img
        │   └── labelcol
        └── Val_Folder
            ├── img
            └── labelcol
    ├── Kvasir-SEG
    │   ├── Test_Folder
    │   │   ├── img
    │   │   └── labelcol
    │   ├── Train_Folder
    |   |   ├── Train_text_sentence.xlsx
    │   │   ├── img
    │   │   └── labelcol
    │   └── Val_Folder
    │       ├── img
    │       └── labelcol
```



### 2. Training


You can train to get your own model.

```angular2html
python train_model.py
```



### 3. Evaluation
#### 3.1. Get Pre-trained Models
Here, we provide pre-trained weights on COVID-Xray, COVID-CT and Kvasir-SEG, if you do not want to train the models by yourself, you can download them in the following links:

*(Note: the pre-trained model will be released in the future.)*

* COVID-Xray: 
* COVID-CT: 
* Kvasir-SEG: 
#### 3.2. Test the Model and Visualize the Segmentation Results
First, change the session name in ```Config.py``` as the training phase. Then run:
```angular2html
python test_model.py
```
You can get the Dice and IoU scores and the visualization results. 



### 4. Results

| Dataset    | 	  Dice (%) | IoU (%) |
| ---------- | ------------------- | -------- |
| COVID-Xray | 80.63      | 71.42   |
| COVID-CT   | 76.18      | 63.41  |
| Kvasir-SEG | 98.19      | 96.45  |
