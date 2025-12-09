## 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

## 2. Set Up Python Environment
<!-- Use Conda -->
```bash
    conda create -n signlang python=3.10
    conda activate signlang
```

<!-- Use Venv -->
```bash
    python3 -m venv signlang
    source signlang/bin/activate  # Linux/Mac
    signlang\Scripts\activate     # Windows
```

## 3. Install dependencies
```bash
    pip install -r requirements.txt
```

## 4. Prepare the data
**A. Data Placement**
Ensure your raw data is structured as follows, based on the configuration in data_preparer.py:
    1/ Keypoint Data: Place the folder containing all raw keypoint files into the expected path:
    ```
    [Your Project Root]/start_kit/keypoints_data/
    ```

    2/ WLASL Metadata: Place the WLASL JSON file:
    ```
    [Your Project Root]/start_kit/WLASL_v0.3.json
    ```

    3/ Raw videos: Place folder containing all raw videos into path:
    ```
    [Your Project Root]/start_kit/raw_videos/
    ```

**B. Pre-trained TGCN Weights**
The TGCN fine-tuning script requires a pre-trained model checkpoint.

    1/ Create the necessary directory:
```bash
mkdir -p TGCN_pretrained_models/asl1000
```

    2/  **ENSURE** the pre-trained weights (`ckpt.pth`) from the original TGCN/ASL source (or its provided link) are placed in this exact location:
    ```
    [Your Project Root]/TGCN_pretrained_models/asl1000/ckpt.pth
    ```
    *(Note: If you have already downloaded this file, you only need to confirm its placement.)*
