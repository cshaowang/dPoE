### Online Multi-view Anomaly Detection with Disentangled Product-of-Experts Modeling

This repo hosts our dPoE published in ACMMM-2023.

> Roadmap >>

```
dPoE/
├── ckpts/ # Store the trained dPoE.
├── modules/ # Store the proposed modules.
|	├── dPoeModel.py  ===  The proposed dPoE model
|	└── dPoeTraining.py  === Train the proposed model
├── utils/ # Datasets, data processing, model loader, and evalaution.
|	├── DATA/ # Store the datasets
|	├── DataLoader.py === Load data, and generate anomalies
|	├── evaluator.py  === Performance evaluation using AUC
|	└── ModelLoader.py  === Load the trained model
├── main.py === run (training and test) dPoE
└── README.md === THIS file!

```

The dPoE was deployed in the following environments:

- python 3.7.13
- pytorch 1.8.1
- torchvision 0.9.1
- scikit-learn 1.0.2
- scipy 1.7.3

Below is a quick start about how to train and test the proposed dPoE.

Step 1. Generate anomaly
```sh
cd utils
python DataLoader.py --anomaly_type='view'
```

Step 2. Train the proposed dPoE model
```sh
cd dPoE
python main.py --train_mode
```

Step 3. Test the trained dPoE model
```sh
cd dPoE
python main.py --anomaly_type='view'
```

Cite ME:

```
@inproceedings{wang2023debunking,
  title={Debunking free fusion myth: Online multi-view anomaly detection with disentangled product-of-experts modeling},
  author={Wang, Hao and Cheng, Zhi-Qi and Sun, Jingdong and Yang, Xin and Wu, Xiao and Chen, Hongyang and Yang, Yan},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={3277--3286},
  year={2023}
}
```

Should you have any questions, please feel free to concact me (Hao Wang, cshaowang@gmail.com). Thanks!