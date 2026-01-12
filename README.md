🩺 Multimodal Breast Cancer Detection (BI-RADS Classification)

This project implements a multimodal deep learning system for breast cancer detection and BI-RADS classification using:

🖼️ Multi-view mammogram images (4 views per case)

📝 Radiology text reports

🧠 ResNet-50 for image feature extraction

📄 BERT (bert-base-uncased) for report understanding

🔗 Feature fusion for final BI-RADS prediction (I–VI)

The system supports training, evaluation, single-case prediction, confusion matrix visualization, and automatic AI report generation.

🚀 Features

Multi-view mammogram analysis (L_CC, L_MLO, R_CC, R_MLO)

Automatic BI-RADS label extraction from reports

Balanced multimodal feature fusion (Image + Text)

Detailed evaluation:

Accuracy

Classification report

Confusion matrix

AI-generated clinical-style reports per case

GPU support (CUDA if available)

🧠 BI-RADS Classes Mapping
Class Index	BI-RADS Category
0	BI-RADS I – Negative
1	BI-RADS II – Benign
2	BI-RADS III – Probably Benign
3	BI-RADS IV – Suspicious
4	BI-RADS V – Highly Suggestive of Malignancy
5	BI-RADS VI – Known Malignancy
📁 Dataset Directory Structure (IMPORTANT)

Your dataset must follow this exact structure:

data_dir/
│
├── Case_001/
│   ├── L_CC.jpg
│   ├── L_MLO.jpg
│   ├── R_CC.jpg
│   ├── R_MLO.jpg
│   └── report.txt
│
├── Case_002/
│   ├── L_CC.jpg
│   ├── L_MLO.jpg
│   ├── R_CC.jpg
│   ├── R_MLO.jpg
│   └── report.txt
│
├── Case_003/
│   └── ...

📌 Notes

Each case must be in a separate folder

Image format: .jpg

Image names must be exactly:

L_CC.jpg

L_MLO.jpg

R_CC.jpg

R_MLO.jpg

Missing images will be automatically replaced with zero tensors

The model supports variable report length

📝 Radiology Report Format

The report.txt file should contain BI-RADS information in any common clinical format, for example:

IMPRESSION:
BI-RADS Category: IV

Findings:
Irregular spiculated mass in the upper outer quadrant...


Supported patterns:

BI-RADS: IV

BI RADS 4B

Category V

ACR 3

➡️ The highest BI-RADS value found in the report will be used as the label.

⚙️ Installation
pip install torch torchvision transformers
pip install numpy pandas matplotlib seaborn scikit-learn pillow


Python 3.8+ recommended
CUDA optional but highly recommended

▶️ How to Run

Place your dataset inside data_dir/

Edit the dataset path if needed:

data_dir_path = 'data_dir'


Run the project:

python main.py

📊 Outputs

After execution, the following files will be generated:

File	Description
training_history.png	Training loss & accuracy
confusion_matrix.png	Model confusion matrix
breast_cancer_detector.pth	Trained model
AI_Reports/	AI-generated reports per case

⚠️ Disclaimer

This project is for research and educational purposes only.
It must not be used as a standalone diagnostic tool.
Final diagnosis and clinical decisions must be made by qualified healthcare professionals.

-------------------------------------------------------------------------------------------
<img width="503" height="455" alt="output (1)" src="https://github.com/user-attachments/assets/118063ed-b0cd-4798-ac4a-322fce21ccf5" />

================================================================================
BREAST CANCER DETECTION AI REPORT
================================================================================
Case ID: case_0014
Report Date: 2025-10-25 06:15:25

================================================================================
PREDICTION SUMMARY
================================================================================
AI Assessed Category: **BI-RADS 1 (0): Negative - No significant findings**
AI Confidence Level: 96.13%

================================================================================
PROBABILITY DISTRIBUTION
================================================================================
• BI-RADS 1 (0): 96.13%
• BI-RADS 2 (1): 2.72%
• BI-RADS 3 (2): 0.91%
• BI-RADS 4 (3): 0.10%
• BI-RADS 5 (4): 0.09%
• BI-RADS 6 (5): 0.05%

================================================================================
AI ANALYSIS FINDINGS
================================================================================
Normal fibroglandular architecture. No suspicious masses, microcalcifications, or architectural distortions detected.

================================================================================
ORIGINAL RADIOLOGY FINDINGS
================================================================================
Indication:
MAMMO
Screening Mammogram
Right breast
ACR:I
BIRADS: III( small few indeterminate retroareolar microcalcification).
Left Breast
ACR:I
BIRADS: I
Recommendation
Follow up after 6 months.

================================================================================
CLINICAL RECOMMENDATIONS
================================================================================
• Routine annual screening mammography
• Follow standard screening guidelines

================================================================================
DISCLAIMER
================================================================================
This report is generated by artificial intelligence and is intended to assist
healthcare professionals. It should not replace clinical judgment. The interpreting
physician is responsible for the final diagnosis and management.

Report End Time: 06:15:25
================================================================================
