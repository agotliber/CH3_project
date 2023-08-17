# CH3 Project Reproducing and Extending: Predicting Spatial DNA Methylation from Sequence and Gene Expression using Deep Learning with Attention.

Welcome to the CH3 project repository! This repository showcases our work on predicting DNA methylation from sequence data and gene expression data at different retention levels, utilizing deep learning techniques enhanced with attention mechanisms. Our project builds upon the framework presented in the paper by [Levy-Jurgenson et al.](https://link.springer.com/chapter/10.1007/978-3-030-18174-1_13)

## Overview

The primary objective of this project is to reproduce and extend the findings presented in the research paper titled "Predicting Methylation from Sequence and Gene Expression Using Deep Learning with Attention" by Levy-Jurgenson et al. Our work is organized into three main parts, with the first two parts focusing on reproducing the original results, while the third part serves as an extension, laying the groundwork for predicting Spatial DNA Methylation.

### 1. Data Preparation
In this initial phase, we reconstructed the datasets required for training and validating our model, adhering closely to the methodology outlined in the paper. The core datasets utilized include gene expression and methylation level data from patients with BRCA and LUAD conditions, coupled with essential human genome, CpG locations, and gene locations data.
Leveraging this raw information, we generated four data files: 1. sequences centered around each CpG site 2. distances between CpG sites and genes. 3. gene expression per subject 4. methylation level data per sample and CpG site.
For an in-depth understanding of our data preparation process, including associated documents and code, refer to the [Data Preparation Documentation](link-to-data-prep-documentation).

## 1. Data Preparation

In this initial phase, we embarked on the meticulous reconstruction of datasets essential for the training and validation of our model, meticulously aligning our approach with the methodology outlined in the paper. The foundational datasets employed encompass gene expression and methylation level data sourced from patients with BRCA and LUAD conditions. Complementing these datasets are pivotal components: human genome data, CpG locations, and gene locations data.

Harnessing the raw information at hand, we diligently generated four distinct data files:

1. Sequences Centered around Each CpG Site
2. Distances between CpG Sites and Genes
3. Gene Expression Per Subject
4. Methylation Level Data Per Sample and CpG Site

These meticulously prepared datasets form the bedrock of our project, poised to underpin subsequent model implementation and predictive analyses. For a comprehensive grasp of our data preparation methodology, as well as access to pertinent documentation and code, we invite you to explore the [Data Preparation Documentation](link-to-data-prep-documentation).


### 2. Model Implementation

The heart of our project involves implementing the deep learning model introduced in the aforementioned paper by Levy-Jurgenson et al. However, we chose to implement this model using PyTorch, a departure from the original TensorFlow implementation. This decision was guided by our familiarity with PyTorch and our goal to achieve equivalent results to the original model.

Accompanying our PyTorch implementation, we have established a comprehensive training and testing environment. To access the model code and instructions, please visit the [Model Code Repository](link-to-model-code).

### 3. Dilution Test: Evaluating Spatial Methylation Prediction

In the third phase, we devised a novel dilution test to gauge the potential success of our spatial methylation prediction approach. Through this test, we simulated gene expression data typical of spatial samples. We introduced a random dilution process to the gene expression test data, wherein the strength of gene expression serves as the probability of a gene's inclusion in the sample's diluted data.

We conducted dilution tests at various working points, including 100, 1000, 5000, 10,000, 15,000, and 20,000. Each working point was subjected to 10 iterations, and we derived accuracy statistics based on these iterations. As anticipated, better performance was achieved with a higher retention of gene expression data and decreased performance as we approached complete removal of gene expressions.

For a comprehensive insight into the dilution test results and associated performance graphs, please explore the [Dilution Test Results](link-to-dilution-results).

Thank you for visiting our repository and delving into the complexities of predicting spatial DNA methylation patterns through deep learning with attention. We hope our work adds value to the ongoing research in this dynamic field.

For inquiries or collaboration opportunities, please reach out to [your-contact-email@example.com](mailto:your-contact-email@example.com).
