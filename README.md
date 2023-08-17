
CH3 Project: Predicting Spatial DNA Methylation from Sequence and Gene Expression using Deep Learning with Attention
Welcome to the CH3 project repository! This repository showcases our work on predicting spatial DNA methylation patterns from sequence data and gene expression using deep learning techniques enhanced with attention mechanisms.

Overview
The primary objective of this project is to reproduce and extend the findings presented in the research paper titled "Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention" by Levy-Jurgenson et al. Our work is organized into three main parts:

1. Data Preparation
In this phase, we meticulously reconstructed the data used for training and validating the deep learning model as detailed in the paper. The core datasets utilized include raw human genome data, CpG location data, and gene location data. Leveraging this raw information, we generated sequences centered around each CpG site and compiled a file of distances between CpG sites and genes. Additionally, we processed BRCA and LUAD methylation level data to derive gene expression and methylation level data per sample and CpG site.

For an in-depth understanding of our data preparation process, including associated documents and code, refer to the Data Preparation Documentation.

2. Model Implementation
The heart of our project involves implementing the deep learning model introduced in the aforementioned paper by Levy-Jurgenson et al. However, we chose to implement this model using PyTorch, a departure from the original TensorFlow implementation. This decision was guided by our familiarity with PyTorch and our goal to achieve equivalent results to the original model.

Accompanying our PyTorch implementation, we have established a comprehensive training and testing environment. To access the model code and instructions, please visit the Model Code Repository.

3. Dilution Test: Evaluating Spatial Methylation Prediction
In the third phase, we devised a novel dilution test to gauge the potential success of our spatial methylation prediction approach. Through this test, we simulated gene expression data typical of spatial samples. We introduced a random dilution process to the gene expression test data, wherein the strength of gene expression serves as the probability of a gene's inclusion in the sample's diluted data.

We conducted dilution tests at various working points, including 100, 1000, 5000, 10,000, 15,000, and 20,000. Each working point was subjected to 10 iterations, and we derived accuracy statistics based on these iterations. As anticipated, better performance was achieved with a higher retention of gene expression data and decreased performance as we approached complete removal of gene expressions.

For a comprehensive insight into the dilution test results and associated performance graphs, please explore the Dilution Test Results.

Thank you for visiting our repository and delving into the complexities of predicting spatial DNA methylation patterns through deep learning with attention. We hope our work adds value to the ongoing research in this dynamic field.

For inquiries or collaboration opportunities, please reach out to your-contact-email@example.com.
