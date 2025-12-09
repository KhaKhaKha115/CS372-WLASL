WLASL:
============================================================================================

“This project uses the WLASL dataset (Li et al., 2020). Full attribution is listed in ATTRIBUTION.md.”

## 1. What it Does?
-----------------

This project focuses on evaluating and comparing two distinct deep learning architectures for the task of isolated sign language gesture recognition. Using pre-processed human pose and hand keypoint sequences (derived from WLASL data and engineered via Frame Stacking), I benchmark a standard 1D Convolutional Neural Network (1DCNN) against a Temporal Graph Convolutional Network (TGCN) fine-tuned from a pre-trained ASL1000 model. The comparison is conducted over five target glosses ('book', 'friend', 'go', 'yes', 'no') using 5-Fold Cross-Validation to assess architectural robustness, generalization capability, and overall performance in classifying dynamic skeletal movement data. The project implemented data preprocessing and augmentation techniques to enhance model generalization on small datasets. It visualizes training curves and confusion matrices to understand learning behavior and misclassifications. It compares fine-tuning pretrained TGCN weights with training from scratch. Finally, it provides a reproducible evaluation framework with fold-wise metrics and overall results, making it easy to extend to more glosses or even real-time sign language recognition in the future.


## 2. Quick Start
-----------------

To run this project, you must first set up the environment, prepare the K-Fold data, and then execute the training and evaluation scripts in sequence. Full installation and environment details are in **SETUP.md**.

1.  **Install Dependencies:** Set up the required environment using `requirements.txt`.

2.  **Prepare Data:** Run `data_preparer.py` in start_kit folder with raw data inside to generate the 5-Fold cross-validation files (`processed_data_foldX.npz`). This script will preprocess the dataset, generate .npz files, and store results inside processed_data/.

3.  **Train/Finetune Models:** Execute the training scripts (e.g., `train_tgcn_finetune.py` in code/TGCN folder and `model_training.py` in code/1DCNN folder) to generate the saved model weights (into `.h5` and `.pth` files).

4.  **Evaluate:** Run `models_evaluation.py` in code folder to calculate final metrics and generate confusion matrices for both models across all 5 folds. 


## 3. Video Links
-----------------
**Demo Video:** [https://drive.google.com/drive/folders/1smmGOq6GKkfaT79SRj3VIT6mXwzH9DVl?usp=sharing]

**Technical Walkthrough:** [https://drive.google.com/drive/folders/1smmGOq6GKkfaT79SRj3VIT6mXwzH9DVl?usp=sharing]


## 4. Evaluation Results
-----------------
Models were evaluated using 5-Fold Cross-Validation on independent test sets. The final comparison clearly indicates that the 1D CNN significantly outperformed the fine-tuned TGCN model.

**Average Performance** (5-Fold Test Set)
- Average 1DCNN Accuracy: 0.5505 - Average Loss: 1.2346
- Average TGCN Accuracy: 0.3224 - Average Loss: 1.5653
| Metric | 1D CNN | Fine-tuned TGCN |
| :--- | :--- | :--- |
| **Average Accuracy** | **55.05%** | 32.24% |
| **Average F1-Score** | **0.8697** | 0.6116 |
| **Average Latency (ms)** | 70.71ms/sample | **20.33**ms/sample |
| **Average Loss** | **1.2346** | 1.5653 |

**Qualitative Outcomes** (Error Analysis)
  - **1DCNN:** (Stronger) The model showed high recognition accuracy for signs like 'book' and 'friend'. The primary source of error was the systematic misclassification of the 'go' sign into 'book'.'

  - **Fine-tuned TGCN:** (Weaker) The model exhibited poor discriminative ability and a strong classification bias toward one or two classes (e.g., frequently predicting 'go' or 'book' regardless of the true label). This confirms that 1D CNN is the superior architecture for this task and dataset configuration. Except that TGCN is faster than 1DCNN in terms of latency (speed) to process one sample, in other categories, TGCN performs significantly worse than 1DCNN.


**Model Architecture Justification** 
  - 1D CNN (1D Convolutional Neural Network): Chosen as the Baseline to evaluate speed and simplicity. 1D CNNs are excellent at quickly finding patterns in sequential data (the order of frames).Provides a fundamental performance and inference speed benchmark for comparison with the more complex Graph architecture

  - TGCN (Temporal Graph Convolutional Network): Chosen as the Advanced Model to exploit the data's inherent structure. The TGCN explicitly combines two types of understanding: 1. Spatial (Graph): Processes the physical connections between body joints (e.g., shoulder, elbow, hand). 2. Temporal (Time): Tracks the overall movement and sequence of the sign. The goal is to prove whether explicitly modeling the body's spatial structure, in addition to movement over time, gives a significant performance advantage over a standard 1D sequence model (the 1D CNN).



## 5. Individual Contributions (Solo Project)
-----------------
Ha Nguyen did everything for this project.