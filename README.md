# stellar-classification-ml
# 基于机器学习的恒星类型预测

## 项目简介

本项目为机器学习课程设计，基于 Kaggle 比赛“预测恒星类型”完成天体类别多分类任务。  
任务目标是根据天体观测特征预测其类别，类别包括 GALAXY、QSO、STAR。

本项目只使用机器学习模型 LightGBM，未使用深度学习模型。

## 数据来源

Kaggle Playground Series - Predicting Stellar Class

数据文件包括：

- train.csv
- test.csv
- sample_submission.csv

由于数据来源于 Kaggle 比赛平台，运行时请先在 Kaggle Notebook 中添加比赛数据集。

## 使用模型

- LightGBM
- 5折分层交叉验证
- 特征工程
- balanced accuracy 评价指标
- 混淆矩阵与分类报告分析

## 运行环境

```bash
pip install -r requirements.txt
