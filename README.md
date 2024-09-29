# Binary Image Classification using Convolutional Neural Network (CNN)
This project is aimed at solving a binary image classification problem using a Convolutional Neural Network (CNN). The dataset, titled Dahlias, is used for training and testing the model. The following README outlines the steps taken in the project, from data preprocessing and augmentation to model architecture design, evaluation, and final conclusions.

## Project Objectives
1. **Data Preprocessing and Augmentation:**
   - Resize all images to a resolution of 64 x 64.
   - Apply data augmentation techniques such as rotation, flipping, or zooming to increase the variety of the training dataset and improve model generalization.
2. **Baseline CNN Architecture:**
   
![Architecture](https://github.com/user-attachments/assets/6df38666-7413-457c-a65b-702b21737f38)

   - Implement the baseline CNN architecture as outlined in Figure 1, using the following specifications:
      - Use ReLU as the activation function for each hidden layer.
      - Use Softmax as the activation function in the output layer for binary classification.
      - Visualize training loss and validation loss over epochs.
    - Analyze whether the model is overfitting, underfitting, or performing optimally based on the loss curves.
        
3. **Architecture Improvement:**
   - Modify the baseline CNN architecture by introducing additional techniques such as:
        - Dropout: To prevent overfitting.
        - Batch Normalization: To improve the model's training stability and performance.
    - Justify why these modifications lead to a better-performing model compared to the baseline.
4. **Model Evaluation:**
    - Evaluate the performance of the modified architecture using the test set.
    - Predict ground truth values based on the model's output and explain the results in terms of classification metrics such as accuracy, precision, recall, and F1-score.
5. **Code Walkthrough and Explanation:**
    - Record a video explaining the code, including the architecture, results, and an analysis of the performance metrics.
    - Share insights on how well the model performed and your opinion on the evaluation results.

## Dataset Description
The dataset contains labeled images for binary classification. Each image belongs to one of two classes and is resized to 64 x 64 for input into the CNN. Data augmentation is applied to increase the diversity of the training data, ensuring better generalization.

## Project Workflow
1. **Data Preprocessing and Augmentation:**
    - Resize all images to 64x64 resolution.
    - Apply data augmentation techniques like random flips, rotations, zooming, etc., to the training data to prevent overfitting and improve generalization.
2. **Baseline CNN Model:**
    - Build a baseline CNN model following the architecture provided in Figure 1:
        - Convolutional and MaxPooling layers
        - Fully connected Dense layers
        - Use ReLU activation in hidden layers and Softmax activation in the output layer.
    - Train the model and plot the training loss and validation loss graphs over epochs.
    - Analyze the performance of the model and determine if it is overfitting, underfitting, or performing optimally.
3. **Modified CNN Model:**
    - Improve the baseline CNN by adding:
        - Dropout layers to reduce overfitting.
        - Batch Normalization layers to stabilize and speed up training.
        - Experiment with different architectures and hyperparameters to achieve better performance.
    - Explain why these modifications lead to a better-performing model.
4. **Model Evaluation:**
    - Use the modified model to evaluate its performance on the test dataset.
    - Generate predictions and compare them with the ground truth.
    - Use evaluation metrics such as accuracy, precision, recall, and F1-score to assess the model's classification performance.
5. **Video Explanation:**
    - Record a video walkthrough explaining the entire process, including the architecture design, training, evaluation, and results.
    - Share your insights on the model’s performance and any key takeaways.
      
## Requirements
- Python 3.x
- Libraries:
  - `TensorFlow`
  - `Pandas`
  - `NumPy`
  - `Scikit-learn`
  - `Matplotlib`
  - `Seaborn`
 
## Results
- Baseline CNN Model: The initial architecture was trained and evaluated, with training and validation loss graphs plotted to analyze the model's performance.
- Modified CNN Model: The architecture was optimized using dropout and batch normalization, achieving better performance compared to the baseline model.

## Conclusion
This project demonstrates the process of building, training, and optimizing a CNN for binary image classification. The modifications introduced to the baseline model (dropout, batch normalization) significantly improved performance, resulting in higher accuracy and better generalization.
