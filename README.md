# STPNet
STPNet: Scale-aware Text Prompt Network for Medical Image Segmentation

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

  *Note: The text annotation of COVID-Xray train datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDguFoqa9YVfXdadWtsA?e=YpQ1pL).*
  
  *The text annotation of COVID-Xray val datasets [download link](https://1drv.ms/x/s!AihndoV8PhTDguFmGZlojiUgiUK_Bw?e=dCt72d).*
  
  *The text annotation of COVID-Xray test dataset [download link](https://1drv.ms/x/s!AihndoV8PhTDguFndDhGZ_w09BwQCw?e=1CrgtK).*
  
  *(Note: The text annotation of MosMedData+ dataset will be released in the future. And you can email to me for the datasets)*

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
