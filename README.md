# Predicting-breast-cancer-using-machine-learning-algorithm

# Breast Cancer prediction:
I download the dataset from https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset

Objective:
- Understand the Dataset & cleanup (We use the only the feature_mean for our model).
- Build classification models to predict whether the cancer type is Malignant or Benign.
- Also fine-tune the hyperparameters & compare the evaluation metrics of various classification algorithms.


Breast cancer is the most common cancer amongst women in the world. It accounts for 25% of all cancer cases, and affected over 2.1 Million people in 2015 alone. It starts when cells in the breast begin to grow out of control. These cells usually form tumors that can be seen via X-ray or felt as lumps in the breast area.

The key challenge against its detection is how to classify tumors into malignant (cancerous, labeled 1) or benign(non-cancerous, labeled 0).

|         |   B   |    M   |
|---------|-------|--------|
|diagnosis| 357   |   212  |

# Reading the data: 
the data contain the following feature :
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean
- concave points_mean
- symmetry_mean
- fractal_dimension_mean

after ploting the dataset we find some outliners we have to deal with to make sure that we receive the best results
![download](https://user-images.githubusercontent.com/93203143/182454765-41f7c906-d952-4758-bc83-e7db966f0f64.png)

# building models:

I use two model to predict the status and compare each one to fide the optimal model I use KNN model and Decision Tree model and to make sure I use the best tree I build Tree with Grid search.
![download (1)](https://user-images.githubusercontent.com/93203143/182584483-85471edd-225f-42a1-9a44-cb15ababbdaf.png)


# results:
train_test_full_error for the Decision Tree Classifier:

|         |   train   |    test   |
|---------|-------|--------|
|accuracy| 1.0   |   0.914894  |
|precision| 1.0   |   0.849315  |
|recall| 1.0   |   0.925373  |
|f1| 1.0   |   0.885714  |

train_test_full_error for the GridSearchCV:

|         |   train   |    test   |
|---------|-------|--------|
|accuracy| 0.947507   |   0.888298  |
|precision| 0.925170  |   0.810811 |
|recall| 0.937931   |   0.895522 |
|f1| 0.931507  |  0.851064 |

train_test_full_error for the KNN:

|         |   train   |    test   |
|---------|-------|--------|
|accuracy| 1.0   |   0.952128 |
|precision| 1.0  |   	0.939394 |
|recall| 1.0   |   0.925373 |
|f1| 1.0 |  0.932331 |

- The KNN model gave us the best results.
- Grid search did not help to improve the results from our decision tree.
