# MetaFace: Meta-Learning for Debiasing Face Encoders

## Overview
MetaFace is a research project exploring meta-learning approaches to mitigate bias in face recognition systems. By leveraging episodic training and multiple demographic attributes, we aim to develop face encoders that maintain high recognition accuracy while reducing demographic bias.

## 🎯 Objectives
- Implement and compare different meta-learning approaches for face recognition
- Reduce demographic bias in face encoders while maintaining performance
- Provide a flexible framework for experimenting with various debiasing strategies
- Create reproducible benchmarks for evaluating fairness in face recognition

## Key training parameters (Prototypical Models)
--dataset_root      # Path to dataset
--experiment_root   # Output directory
--epochs           # Number of training epochs
--learning_rate    # Initial learning rate
--cuda             # Use GPU acceleration
--classes_per_it_tr # Number of classes per training episode
--num_support_tr   # Number of support samples per class
--num_query_tr     # Number of query samples per class
