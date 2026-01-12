import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from transformers import BertTokenizer, BertModel
import warnings

warnings.filterwarnings('ignore')

RESNET_WEIGHTS = models.ResNet50_Weights.IMAGENET1K_V1
IMAGE_SIZE = 1024
BERT_MODEL_NAME = 'bert-base-uncased'
NUM_CLASSES = 6
MAX_REPORT_LENGTH = 512

torch.manual_seed(42)
np.random.seed(42)

class MammogramDataset(Dataset):
    BI_RADS_CONVERSION = {
        '0': 0, 'I': 0, '1': 0,
        'II': 1, '2': 1,
        'III': 2, '3': 2,
        'IV': 3, '4': 3, '4A': 3, '4B': 3, '4C': 3,
        'V': 4, '5': 4,
        'VI': 5, '6': 5
    }

    def __init__(self, data_dir: str, cases: List[str], transform: transforms.Compose = None, max_length: int = MAX_REPORT_LENGTH):
        self.data_dir = data_dir
        self.cases = cases
        self.transform = transform
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
        self.max_length = max_length
        self.labels = self._extract_labels()

    def _extract_labels(self) -> List[int]:
        labels = []
        
        birads_pattern = re.compile(
            r'BI\s*[-]?\s*RADS[^\n:]*[:\s]*\s*([IVX123456]+\s*[ABC]?)',
            re.IGNORECASE | re.DOTALL
        )
        
        global_patterns = re.compile(
            r'(?:ACR|Category|IMPRESSION).*?([IVX123456]+\s*[ABC]?)',
            re.IGNORECASE | re.DOTALL
        )

        for case in self.cases:
            report_path = os.path.join(self.data_dir, case, 'report.txt')
            final_birads = 0  

            try:
                with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                    report_text = f.read().upper()

                matches = birads_pattern.findall(report_text)
                if not matches:
                    matches = global_patterns.findall(report_text)
                
                birads_values = []
                for match in matches:
                    cleaned_value = match.strip().replace('-', '').upper()
                    birads_index = self.BI_RADS_CONVERSION.get(cleaned_value, 0)
                    birads_values.append(birads_index)

                final_birads = max(birads_values) if birads_values else 0

            except Exception as e:
                final_birads = 0  

            labels.append(final_birads)

        return labels

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        case = self.cases[idx]
        case_path = os.path.join(self.data_dir, case)

        images = []
        view_types = ['L_CC', 'L_MLO', 'R_CC', 'R_MLO']  
        
        dummy_tensor_shape = (3, IMAGE_SIZE, IMAGE_SIZE) 
        
        for view in view_types:
            img_path = os.path.join(case_path, f"{view}.jpg")
            try:
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                images.append(image)
            except Exception:
                dummy_img = torch.zeros(dummy_tensor_shape)
                images.append(dummy_img)

        mammogram_tensor = torch.stack(images)

        report_path = os.path.join(case_path, 'report.txt')
        input_ids = torch.zeros(self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(self.max_length, dtype=torch.long)

        try:
            with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
                report_text = f.read()

            encoding = self.tokenizer(
                report_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].squeeze(0) 
            attention_mask = encoding['attention_mask'].squeeze(0)
        except Exception:
            pass

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return mammogram_tensor, input_ids, attention_mask, label

class MultiViewMammogramModel(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.3):
        super(MultiViewMammogramModel, self).__init__()

        self.cnn = models.resnet50(weights=RESNET_WEIGHTS)
        num_features = self.cnn.fc.in_features
        
        for param in self.cnn.parameters():
            param.requires_grad = False
            
        self.cnn.fc = nn.Identity()  

        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
        
        for param in self.bert.parameters():
            param.requires_grad = False
            
        text_features = self.bert.config.hidden_size  

        combined_features = 4 * num_features + text_features

        self.classifier = nn.Sequential(
            nn.Linear(combined_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        image_features = []
        for i in range(images.size(1)):
            view_images = images[:, i, :, :, :]
            features = self.cnn(view_images)
            image_features.append(features)

        image_features = torch.cat(image_features, dim=1)

        text_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        combined_features = torch.cat([image_features, text_features], dim=1)
        combined_features = self.dropout(combined_features)

        output = self.classifier(combined_features)

        return output

class BreastCancerDetector:
    def __init__(self, data_dir: str, batch_size: int = 8, learning_rate: float = 1e-4):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = None
        self.cases = None
        self.train_dataset = None
        self.test_dataset = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None

    def _get_all_cases_and_labels(self) -> Tuple[List[str], List[int]]:
        self.cases = [d for d in os.listdir(self.data_dir)
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not self.cases:
            raise ValueError("No case directories found in the specified data directory.")

        temp_cases = sorted(self.cases)
        temp_dataset = MammogramDataset(self.data_dir, temp_cases, None)
        all_labels = temp_dataset.labels
        
        self.cases = temp_cases
        return all_labels

    def load_data(self):
        all_labels = self._get_all_cases_and_labels()
        
        label_counts = np.bincount(all_labels)
        print("Class distribution (index 0-5 -> BI-RADS I-VI):", label_counts)
        
        train_cases, test_cases, _, _ = train_test_split(
            self.cases, all_labels, 
            test_size=0.2, 
            random_state=42,
            stratify=all_labels if len(np.unique(all_labels)) > 1 and np.min(label_counts) > 1 else None 
        )

        self.train_dataset = MammogramDataset(self.data_dir, train_cases, self.transform)
        self.test_dataset = MammogramDataset(self.data_dir, test_cases, self.transform)

        print(f"Loaded {len(train_cases)} training cases and {len(test_cases)} test cases")

    def create_model(self):
        self.model = MultiViewMammogramModel(num_classes=NUM_CLASSES).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(trainable_params, lr=self.learning_rate)
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)

    def train(self, num_epochs: int = 20):
        if not self.train_dataset:
            self.load_data()
        if not self.model:
            self.create_model()

        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"\nTraining on {self.device} for {num_epochs} epochs...")
        
        train_losses = []
        train_accuracies = []

        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, input_ids, attention_mask, labels) in enumerate(train_loader):
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(images, input_ids, attention_mask)
                loss = self.criterion(outputs, labels)

                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 10 == 0:
                    print(f'Epoch: {epoch+1}/{num_epochs}, '
                          f'Batch: {batch_idx}/{len(train_loader)}, '
                          f'Loss: {loss.item():.4f}', end='\r')

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total

            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)

            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Loss: {epoch_loss:.4f}, '
                  f'Accuracy: {epoch_acc:.2f}% | LR: {self.scheduler.get_last_lr()[0]:.1e}')

            self.scheduler.step()

        self.plot_training_history(train_losses, train_accuracies)

    def evaluate(self):
        if not self.test_dataset or not self.model:
            print("Loading data and model for evaluation...")
            self.load_data()
            self.create_model()

        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print("\nStarting evaluation...")

        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, input_ids, attention_mask, labels in test_loader:
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images, input_ids, attention_mask)
                _, preds = outputs.max(1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"\n{'='*50}\nTEST RESULTS\n{'='*50}")
        print(f"Test Accuracy: {accuracy:.4f}")

        target_names = ['BI-RADS I (0)', 'II (1)', 'III (2)', 'IV (3)', 'V (4)', 'VI (5)']
        if len(set(all_labels)) > 1:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds,
                                        target_names=target_names,
                                        labels=range(NUM_CLASSES),
                                        zero_division=0))
        else:
            print(f"\nAll test cases belong to the same class: {all_labels[0]}")

        self.plot_confusion_matrix(all_labels, all_preds)

    def predict_single_case(self, case_dir: str) -> Tuple[int, np.ndarray]:
        if not self.model:
            raise ValueError("Model not trained or loaded. Please train or load the model first.")

        single_case_path = os.path.join(self.data_dir, case_dir)
        if not os.path.isdir(single_case_path):
            raise ValueError(f"Case directory not found: {single_case_path}")

        single_dataset = MammogramDataset(self.data_dir, [case_dir], self.transform)

        data_loader = DataLoader(single_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        with torch.no_grad():
            for images, input_ids, attention_mask, _ in data_loader:
                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(images, input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()

        return predicted_class, probabilities.cpu().numpy()[0]

    def generate_report(self, case_dir: str, predicted_class: int, probabilities: np.ndarray) -> str:
        birads_categories: Dict[int, str] = {
            0: "BI-RADS 1 (0): Negative - No significant findings",
            1: "BI-RADS 2 (1): Benign findings - Non-cancerous abnormalities",
            2: "BI-RADS 3 (2): Probably Benign - Short-term follow-up recommended",
            3: "BI-RADS 4 (3): Suspicious - Biopsy should be considered",
            4: "BI-RADS 5 (4): Highly Suggestive of Malignancy - Strong cancer suspicion",
            5: "BI-RADS 6 (5): Known Biopsy-Proven Malignancy - Confirmed cancer"
        }
        
        findings = ""
        report_path = os.path.join(self.data_dir, case_dir, 'report.txt')
        if os.path.exists(report_path):
            try:
                with open(report_path, 'r', encoding='utf-8') as f:
                    original_report = f.read()
                    findings_match = re.search(r'Findings:(.*?)Impression:', original_report, re.DOTALL | re.IGNORECASE)
                    if findings_match:
                        findings = findings_match.group(1).strip()
                    else:
                        findings = original_report[:500].strip() + ("..." if len(original_report) > 500 else "")
            except Exception:
                findings = "Could not read original report findings."

        recommendations = {
            0: ["• Routine annual screening mammography", "• Follow standard screening guidelines"],
            1: ["• Routine annual follow-up", "• No immediate intervention required"],
            2: ["• Short-term follow-up in 6 months", "• Additional ultrasound evaluation recommended"],
            3: ["• Tissue diagnosis through biopsy REQUIRED", "• Surgical consultation recommended"],
            4: ["• IMMEDIATE biopsy necessary", "• Surgical oncology consultation URGENTLY", "• Breast MRI for staging and planning"],
            5: ["• IMMEDIATE cancer treatment initiation", "• Multidisciplinary oncology team consultation", "• Treatment planning (surgery, chemotherapy, radiation)"]
        }

        ai_findings = {
            0: "Normal fibroglandular architecture. No suspicious masses, microcalcifications, or architectural distortions detected.",
            1: "AI detected benign findings (e.g., oil cysts, benign calcifications).",
            2: "AI detected a finding with a low but non-zero probability of malignancy, suggesting short-interval follow-up.",
            3: "AI identified suspicious features (e.g., irregular mass, pleomorphic calcifications) with moderate concern for malignancy.",
            4: "AI identified highly suspicious features consistent with malignancy (e.g., spiculated mass, aggressive growth pattern).",
            5: "Prediction is consistent with a known malignancy (e.g., post-biopsy imaging)."
        }

        prob_lines = [
            f'• {birads_categories[i].split(":")[0]}: {prob*100:.2f}%'
            for i, prob in enumerate(probabilities)
        ]
        
        report = f"""
{'='*80}
BREAST CANCER DETECTION AI REPORT
{'='*80}
Case ID: {case_dir}
Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
PREDICTION SUMMARY
{'='*80}
AI Assessed Category: **{birads_categories[predicted_class]}**
AI Confidence Level: {probabilities[predicted_class] * 100:.2f}%

{'='*80}
PROBABILITY DISTRIBUTION
{'='*80}
{chr(10).join(prob_lines)}

{'='*80}
AI ANALYSIS FINDINGS
{'='*80}
{ai_findings.get(predicted_class, 'Analysis not available')}

{'='*80}
ORIGINAL RADIOLOGY FINDINGS
{'='*80}
{findings if findings else 'No original report findings available'}

{'='*80}
CLINICAL RECOMMENDATIONS
{'='*80}
{chr(10).join(recommendations.get(predicted_class, ['• Consultation with radiologist required']))}

{'='*80}
DISCLAIMER
{'='*80}
This report is generated by artificial intelligence and is intended to assist
healthcare professionals. It should not replace clinical judgment. The interpreting
physician is responsible for the final diagnosis and management.

Report End Time: {pd.Timestamp.now().strftime('%H:%M:%S')}
{'='*80}
"""
        return report

    def generate_all_reports(self, output_dir: str = 'AI_Reports'):
        if not self.test_dataset or not self.model:
            print("Loading data and model before generating reports...")
            self.load_data()
            self.create_model()

        os.makedirs(output_dir, exist_ok=True)
        test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False)

        self.model.eval()
        print(f"\nGenerating reports for {len(self.test_dataset)} test cases in '{output_dir}/'...")
        
        with torch.no_grad():
            for i, (images, input_ids, attention_mask, _) in enumerate(test_loader):
                case_dir = self.test_dataset.cases[i]

                images = images.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                outputs = self.model(images, input_ids, attention_mask)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = outputs.argmax(dim=1).item()

                report = self.generate_report(case_dir, predicted_class, probabilities.cpu().numpy()[0])

                filename = f"{case_dir}_AI_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
                filepath = os.path.join(output_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report)

                print(f"Report generated: {filepath}", end='\r')
            print("\nAll reports generated successfully.")

    def plot_training_history(self, losses: List[float], accuracies: List[float]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(losses, label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(accuracies, label='Training Accuracy', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved to training_history.png")

    def plot_confusion_matrix(self, true_labels: List[int], pred_labels: List[int]):
        try:
            cm = confusion_matrix(true_labels, pred_labels, labels=range(NUM_CLASSES))
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['I', 'II', 'III', 'IV', 'V', 'VI'],
                        yticklabels=['I', 'II', 'III', 'IV', 'V', 'VI'])
            plt.xlabel('Predicted BI-RADS Category')
            plt.ylabel('True BI-RADS Category')
            plt.title('Confusion Matrix')
            plt.savefig('confusion_matrix.png')
            print("Confusion matrix plot saved to confusion_matrix.png")
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")

    def save_model(self, path: str = 'breast_cancer_detector.pth'):
        if self.model is None:
             raise ValueError("No model to save. Please train the model first.")
             
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str = 'breast_cancer_detector.pth'):
        if not os.path.exists(path):
             raise FileNotFoundError(f"Model file not found at {path}")
             
        if not self.model:
            self.create_model()

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.model.eval()
        print(f"Model loaded from {path}")

def main():
    data_dir_path = 'data_dir' 
    
    detector = BreastCancerDetector(data_dir=data_dir_path, batch_size=4, learning_rate=1e-5)
    
    try:
        detector.load_data()
        detector.create_model()
        
        detector.train(num_epochs=15)
        detector.evaluate()
        detector.save_model()
        
        detector.generate_all_reports()
        
    except ValueError as e:
        print(f"Startup Error: {e}. Please check your data directory structure.")
    except FileNotFoundError as e:
        print(f"File Error: {e}. Check if required model or data files exist.")
    except Exception as e:
        print(f"An unexpected error occurred during execution: {e}")

if __name__ == "__main__":
    main()
