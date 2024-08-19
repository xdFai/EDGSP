# EDGSP
This is the code of paper 'Beyond Full Label: Single-Point Prompt for Infrared Small Target Label Generation'

Shuai Yuan, Hanlin Qin, Renke Kou, XiangYan, Zechuan Li, Chenxu Peng, Abd-Krim Seghouane [[Paper]](https://www.arxiv.org/abs/2408.08191) [[Weight]](https://drive.google.com/file/d/1zYgTwFDy-cXIfnaaln8fkeW8Z_7-yf4u/view?usp=sharing)

# Chanlleges and inspiration   
![Image text](https://github.com/xdFai/EDGSP/blob/main/Figure/Fig01.png)

# Structure
![Image text](https://github.com/xdFai/EDGSP/blob/main/Figure/Fig02.png)


# Introduction

We present a novel infrared small target label generation (IRSTLG) framework named energy double guided single-point prompt (EDGSP). Experiments on both public (e.g., NUAA-SIRST, NUDT-SIRST, IRSTD-1K) demonstrate the effectiveness of our method. Our main contributions are as follows:

1. To the best of our knowledge, we present the first study of the learning-based IRSTLG paradigm and introduce EDGSP creating a crucial link between label generation and target detection task.

2. We propose target energy initialization (TEI), double prompt embedding (DPE), and bounding box-based matching (BBM) strategies to address insufficient shape evolution, label adhesion, and false alarms.

3. For the first time, three baselines equipped with EDGSP achieve accurate annotation on three datasets. The downstream detection task illustrates that our pseudo label surpasses the full label. Even with coarse point annotated, EDGSP achieves 99.5% performance of full labeling.

## Usage

#### 1. Data

* **Our project has the following structure:**
  ```
  ├──./datasets/
  │    ├── IRSTD-1K
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_IRSTD-1K.txt
  │    │    │    ├── test_IRSTD-1K.txt
  │    ├── NUDT-SIRST
  │    │    ├── images
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── 000001.png
  │    │    │    ├── 000002.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_NUDT-SIRST.txt
  │    │    │    ├── test_NUDT-SIRST.txt
  │    ├── ...
  │    ├── ...
  │    ├── SIRST3
  │    │    ├── images
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── masks
  │    │    │    ├── XDU0.png
  │    │    │    ├── XDU1.png
  │    │    │    ├── ...
  │    │    ├── img_idx
  │    │    │    ├── train_SIRST3.txt
  │    │    │    ├── test_SIRST3.txt
  
  ```
##### 2. Train.
```bash
python train_LG_SCTrans.py
```

#### 3. Test and demo.
```bash
python test_LG_SCTrans_PdFa.py
```

## Results and Trained Models

#### Qualitative Results
![Image text](https://github.com/xdFai/EDGSP/blob/main/Figure/Fig03.png)




#### Quantitative Results on Mixed NUAA-SIRST, NUDT-SIRST, and IRSTD-1K

| Model         | mIoU (x10(-2)) | Pd (x10(-2))| Fat (x10(-2))| Fa (x10(-6))|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| SIRST         | 83.83  |  100.0 | 0 | 0 |
| NUDT-SIRST    | 95.51  |  100.0 | 0 | 0 |
| IRSTD-1K      | 73.80  |  100.0 | 0 | 0 |
| [[Weights]](https://drive.google.com/file/d/1zYgTwFDy-cXIfnaaln8fkeW8Z_7-yf4u/view?usp=sharing)|


## Contact
**Welcome to raise issues or email to [yuansy@stu.xidian.edu.cn](yuansy@stu.xidian.edu.cn) or [yuansy2@student.unimelb.edu.au](yuansy2@student.unimelb.edu.au) for any question regarding our EDGSP.**

