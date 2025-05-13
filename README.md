# 🚗 Car Evaluation Decision Tree Classifier (Weka + Java)

This project implements a **Decision Tree Classifier** using the **Weka Java API** to analyze the [Car Evaluation Dataset](https://archive.ics.uci.edu/dataset/19/car+evaluation). The goal is to compare two models (M1 and M2) based on different training/testing splits, evaluate their performance, and visualize the resulting decision trees.

---

## 📂 Project Structure

project2/
│
├── Ai.java # Main Java file - trains & evaluates models
├── car_evaluation.arff # Dataset (UCI Car Evaluation Dataset)
├── decision_tree_M1.dot # DOT file for Model M1 decision tree
├── decision_tree_M2.dot # DOT file for Model M2 decision tree
├── results_M1.txt # Evaluation results for Model M1
├── results_M2.txt # Evaluation results for Model M2
├── M1_train_distribution.txt # Class distribution in M1 training data
├── M1_test_distribution.txt # Class distribution in M1 test data
├── M2_train_distribution.txt # Class distribution in M2 training data
├── M2_test_distribution.txt # Class distribution in M2 test data
└── README.md # Project documentation (this file)

markdown
Copy
Edit

---

## 📌 Objectives

This project addresses the following tasks:

1. **Preprocessing**: Randomizes the dataset before splitting.
2. **Model M1**: Uses a 70/30 train-test split.
3. **Model M2**: Uses a 50/50 train-test split.
4. **Training**: Uses the J48 (C4.5 implementation) classifier from Weka.
5. **Evaluation**:
   - Accuracy and F1-score
   - Confusion matrix
6. **Visualization**: Exports the decision tree to DOT files for both models.
7. **Distribution Analysis**: Computes the target class distribution for each split.

---

## 🧪 How to Run

> ✅ Prerequisites: Java and [Weka jar library](https://waikato.github.io/weka-wiki/use_weka_in_your_java_code/)

1. Place `weka.jar` in your project directory.
2. Compile the Java file:

```bash
javac -cp .;weka.jar Ai.java
Run the classifier:

bash
Copy
Edit
java -cp .;weka.jar Ai
📊 Results Summary
🎯 Model M1 (70% Train / 30% Test)
Accuracy: See results_M1.txt

F1-Score: See results_M1.txt

Decision Tree: decision_tree_M1.dot

🔍 Model M2 (50% Train / 50% Test)
Accuracy: See results_M2.txt

F1-Score: See results_M2.txt

Decision Tree: decision_tree_M2.dot

📈 Class Distribution
Files:

M1_train_distribution.txt

M1_test_distribution.txt

M2_train_distribution.txt

M2_test_distribution.txt

📦 Dataset Info
Source: UCI Car Evaluation Dataset

Attributes:

Buying, Maintenance, Doors, Persons, Lug Boot, Safety

Class (Target): Acceptability (unacc, acc, good, vgood)

✍️ Authors
Qosai Badaha --1210469


📌 References
UCI Dataset Description

Weka Java Integration Guide

Graphviz (for DOT tree visualization)

🧠 Notes
The .dot files can be visualized using tools like Graphviz to display the decision tree structure.

The project is implemented using pure Java and Weka, providing full control over data flow and evaluation.

🎓 This project is part of COMP338: Artificial Intelligence — Fall 2024/2025
