**Comprehensive Documentation: Structuring Labels for Multilabel Classification**

---

### **Overview**
Multilabel classification tasks involve assigning multiple labels to a single data instance. This document provides a clear and structured explanation of how to organize label files for multilabel classification to ensure easy understanding and reproducibility for collaborators and stakeholders.

---

### **Key Components of a Multilabel Classification Dataset**
To effectively structure the dataset for a multilabel classification task, the following components are essential:

#### **1. Image Files**
- Store all image files in a dedicated directory, ensuring proper naming conventions and easy access.
- Example directory structure:

```
project_directory/
|
|-- images/                     # Directory containing all image files
|   |-- image1.jpg
|   |-- image2.jpg
|   |-- ...
|
|-- labels.csv                  # CSV file containing multilabel annotations
|
|-- scripts/                    # Scripts for data processing and model training
|-- checkpoints/                # Folder for saving trained model checkpoints
|-- README.md                   # Documentation file
```

#### **2. Labels File (CSV Format)**
The `labels.csv` file is the backbone of the dataset, containing mappings between the data instances (e.g., images) and their associated labels.

##### **Structure of `labels.csv`:**
1. **Image_Name**: This column contains the filenames of the images (e.g., `image1.jpg`).
2. **Binary Label Columns**: Each class has a dedicated column, where:
   - `1` indicates that the class applies to the image.
   - `0` indicates that the class does not apply to the image.

##### **Example `labels.csv`:**
| Image_Name   | Class_A | Class_B | Class_C | Class_D |
|--------------|---------|---------|---------|---------|
| image1.jpg   | 1       | 0       | 1       | 0       |
| image2.jpg   | 0       | 1       | 0       | 1       |
| image3.jpg   | 1       | 1       | 0       | 0       |
| image4.jpg   | 0       | 0       | 1       | 1       |

---

### **How to Create and Structure the Labels File**

#### **Step 1: Define Classes**
- Identify all the possible labels (classes) that an instance can belong to.
- Example: In a stroke creaking phenomena classification task, the classes might include:
  - **Stroke_Type1**
  - **Stroke_Type2**
  - **Noise**
  - **Crack_Visible**
  - **Deformation**

#### **Step 2: Annotate the Data**
- For each data instance (e.g., image), identify which labels apply.
- Assign `1` for applicable labels and `0` for non-applicable labels.

#### **Step 3: Create the CSV File**
- Use a spreadsheet or a Python script to create a CSV file.
- Each row should correspond to an image, and columns represent the labels.

#### **Step 4: Validate the Labels File**
- Verify that the `Image_Name` column correctly matches the image filenames.
- Ensure there are no missing or inconsistent labels.

---

### **Best Practices for Label Files**

#### **1. Ensure Consistency**
- The `Image_Name` column should directly match the filenames of the images in the directory.
- Labels should be binary (0 or 1) and consistently formatted.

#### **2. Handle Missing Data**
- If a label is unknown or not applicable, explicitly assign `0` rather than leaving it blank.

#### **3. Keep Classes Balanced**
- If possible, ensure a balanced distribution of labels across the dataset to avoid bias.

#### **4. Use Descriptive Class Names**
- Class names should be self-explanatory to avoid confusion (e.g., use `Crack_Visible` instead of `C_V`).

#### **5. Maintain a README**
- Include a `README.md` file in the project directory to describe:
  - The purpose of the dataset.
  - The meaning of each label.
  - How the labels were annotated.

---

### **Example Scenario: Stroke Creaking Phenomena**
For the stroke creaking phenomena classification, the dataset might look like this:

#### **Directory Structure:**
```
stroke_creaking_dataset/
|
|-- images/
|   |-- stroke1.jpg
|   |-- stroke2.jpg
|   |-- ...
|
|-- labels.csv
|-- README.md
```

#### **Sample `labels.csv`:**
| Image_Name   | Stroke_Type1 | Stroke_Type2 | Noise | Crack_Visible | Deformation |
|--------------|--------------|--------------|-------|---------------|-------------|
| stroke1.jpg  | 1            | 0            | 1     | 1             | 0           |
| stroke2.jpg  | 0            | 1            | 0     | 0             | 1           |
| stroke3.jpg  | 1            | 0            | 1     | 0             | 0           |

---

### **Using the Labels File in PyTorch**
To load and use this file, create a PyTorch dataset class that dynamically loads the images and their labels. For example:

```python
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch

class MultiLabelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        labels = self.data.iloc[idx, 1:].values.astype('float32')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels)
```

---

### **Conclusion**
A well-structured multilabel classification dataset ensures clarity, reproducibility, and compatibility with machine learning frameworks. By organizing images, creating a properly formatted `labels.csv` file, and following best practices, you can streamline the training and evaluation process for multilabel classifiers. This documentation serves as a guide to ensure others can easily understand and utilize your dataset.

