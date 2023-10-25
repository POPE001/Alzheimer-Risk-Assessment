# Time Series Classification Project - Alzheimer's Disease Prediction using Convolutional Neural Network

## Overview

This project aims to predict the onset of Alzheimer's disease by analyzing demographic data using a Convolutional Neural Network (CNN). Alzheimer's disease is a critical health concern, and early prediction can aid in better management and care. This README provides an overview of the project, its structure, and key findings.

## Project Structure

The project is organized as follows:

- **Data Collection:** The dataset was collected from [source link] and is stored in 'investigator_nacc61.csv'.

- **Data Preprocessing:** In this phase, data was cleaned and processed. This included dealing with missing values, feature engineering, and creating the target variable.

- **Data Visualization:** Key insights were visualized using Matplotlib and Seaborn to better understand the data and its characteristics.

- **Feature Engineering:** Selected features such as age, gender, marital status, and other demographic information were chosen to be used as input for the model.

- **Data Preparation:** The data was split into training, validation, and test sets. The data was scaled using Min-Max scaling, and one-hot encoding was performed on categorical variables.

- **Modeling:** A Convolutional Neural Network (CNN) was constructed for time series classification, considering the sequence length and the number of classes. The model architecture can be seen in 'model_architecture.png'.

- **Model Training:** The model was trained with class weights to handle class imbalance. Training and validation losses and accuracies were monitored and visualized during training.

- **Model Evaluation:** The model was evaluated on the test dataset. Metrics such as weighted recall and F1-score were calculated.

- **Confusion Matrix:** A confusion matrix was generated to visualize the model's performance on different classes.

- **Model Evaluation Visualization:** The model training and validation losses and accuracies were plotted to track model performance.

## Requirements

This project was developed using Python and requires the following packages:

- TensorFlow
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn
- Keras Tuner (for hyperparameter tuning)

## Usage

To run this project, follow these steps:

1. Clone the project repository.
2. Ensure you have the required libraries installed. You can use a virtual environment to manage dependencies.
3. Set up your Python environment.
4. Run the Jupyter notebook or Python script to execute the code.
5. View the results in the generated visualizations or output.

## Model Evaluation

The model achieved a weighted recall of [value] and a weighted F1-score of [value] on the test dataset, indicating that it performs well in predicting the onset of Alzheimer's disease.

## Future Work

For future iterations, you may consider the following improvements:

- Hyperparameter tuning using techniques like grid search or Bayesian optimization.
- Exploring other time series classification models such as Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks.
- Additional feature engineering to capture more relevant information.

## Conclusion

This project demonstrates the application of deep learning techniques for time series classification in predicting the onset of Alzheimer's disease. The model achieved promising results, providing a foundation for further research and improvement in this critical area of healthcare.

For any questions or further details, please contact [oluwamayowaadeoni@gmail.com].

