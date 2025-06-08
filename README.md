## Deep Learning Challenge ##
# Module 21 #

# Background #
The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures.
 With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.
From Alphabet Soup’s business team, you have received access to a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
  1. EIN and NAME—Identification columns
  2. APPLICATION_TYPE—Alphabet Soup application type
  3. AFFILIATION—Affiliated sector of industry
  4. CLASSIFICATION—Government organization classification
  5. USE_CASE—Use case for funding
  6. ORGANIZATION—Organization type
  7. STATUS—Active status
  8. INCOME_AMT—Income classification
  9. SPECIAL_CONSIDERATIONS—Special considerations for application
  10. ASK_AMT—Funding amount requested
  11. IS_SUCCESSFUL—Was the money used effectively


  # Instructions #

    # Step 1: Preprocess the Data #
    Using your knowledge of Pandas and scikit-learn’s StandardScaler(), you’ll need to preprocess the dataset. This step prepares you for Step 2, 
    where you'll compile, train, and evaluate the neural network model.
    Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
      1. From the provided cloud URL, read in the charity_data.csv to a Pandas DataFrame, and be sure to identify the following in your dataset:
          * What variable(s) are the target(s) for your model?
          * What variable(s) are the feature(s) for your model?
      2.  Drop the EIN and NAME columns.
      3. Determine the number of unique values for each column.
      4. For columns that have more than 10 unique values, determine the number of data points for each unique value.
      5. Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, 
         Other, and then check if the replacement was successful.
      6. Use pd.get_dummies() to encode categorical variables.
      7. Split the preprocessed data into a features array, X, and a target array, y. Use these arrays and the train_test_split function to split the data into training and testing datasets.
      8. Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

    # Step 2  Compile, Train, and Evaluate the Model #
    Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an 
    Alphabet Soup-funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining 
    the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.
      1. Continue using the file in Google Colab in which you performed the preprocessing steps from Step 1.
      2. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
      3. Create the first hidden layer and choose an appropriate activation function.
      4. If necessary, add a second hidden layer with an appropriate activation function.
      5. Create an output layer with an appropriate activation function.
      6. Check the structure of the model.
      7. Compile and train the model
      8. Create a callback that saves the model's weights every five epochs.
      9. Evaluate the model using the test data to determine the loss and accuracy.
      10. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity.h5.

    # Step 3 Optimize the Model #
    Using your knowledge of TensorFlow, optimize your model to achieve a target predictive accuracy higher than 75%.
    Use any or all of the following methods to optimize your model:
        Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
          * Dropping more or fewer columns.
          * Creating more bins for rare occurrences in columns.
          * Increasing or decreasing the number of values for each bin.
          * Add more neurons to a hidden layer.
          * Add more hidden layers.
          * Use different activation functions for the hidden layers.
          * Add or reduce the number of epochs to the training regimen.
      1. Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb
      2. Import your dependencies and read in the charity_data.csv to a Pandas DataFrame from the provided cloud URL.
      3. Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
      4. Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
      5. Save and export your results to an HDF5 file. Name the file AlphabetSoupCharity_Optimization.h5.


    # Step 4 Write a Report on the Neural Network Model #
    For this part of the assignment, you’ll write a report on the performance of the deep learning model you created for Alphabet Soup.
    The report should contain the following:
      1. Overview of the analysis: Explain the purpose of this analysis.
      2. Results: Using bulleted lists and images to support your answers, address the following questions:
            2.a Data Preprocessing
                * What variable(s) are the target(s) for your model?
                * What variable(s) are the features for your model?
                * What variable(s) should be removed from the input data because they are neither targets nor features?
            2.b Compiling, Training, and Evaluating the Model
                * How many neurons, layers, and activation functions did you select for your neural network model, and why?
                * Were you able to achieve the target model performance?
                * What steps did you take in your attempts to increase model performance?
       3. Summary: Summarize the overall results of the deep learning model. Include a recommendation for 
       how a different model could solve this classification problem, and then explain your recommendation.

       # Step 5: Copy Files Into Your Repository # 
     

########### Step 4 Answers ##########

# 1. Overview
        The goal of this analysis was to build a deep learning model that predicts the likelihood of success for applicants to the Alphabet Soup charity organization. This enables the organization to make more informed decisions and allocate resources more effectively to the most promising candidates.
# 2. Results
  #    2.a Data Preprocessing
            Target Variable: `IS_SUCCESSFUL` (Binary: 1 if venture was successful, 0 otherwise)
            Feature Variables: All remaining columns after dropping `EIN` and `NAME` were used, including categorical variables transformed using one-hot encoding. 
                               These features represented applicant demographics, financials, and application details.
            Removed Variables:`EIN` and `NAME` were removed as they do not contribute to the predictive capabilities of the model.

  #   2.b Compiling, Training, and Evaluating the Model
       a. Initial Neural Network Model: 'Starter_Code.ipynb'**
       b. Layers: 2 hiddden layers  
       c. Neurons: 80 in the first layer, 30 in the second layer  
       d. Activation Functions: RELU for hidden layers, SIGMOID for output  
       e. Accuracy: 73%   
     This baseline configuration used a common activation function RELU for handling non-linear relationships and a SIGMOID activation in the output layer for binary classification.

  #  3.Summary
        To further enhance the model's performance, Keras Tuner with Hyperband optimization was employed for fine-tuning. Three model variations were tested, focusing on key hyperparameters and architectural adjustments.
        The number of neurons in each layer was carefully adjusted, and a variety of activation functions were explored, including ReLU, Tanh, and Sigmoid. Beyond these, the architecture was expanded to include larger layer sizes and additional activation options such as SELU and Leaky ReLU, providing greater flexibility in capturing complex data patterns.
        Regularization techniques were incorporated to reduce overfitting and improve generalization. Dropout layers were added with rates of 0.2, 0.3, and 0.5, significantly enhancing the model's robustness. Furthermore, learning rates were tuned with values of 0.001, 0.005, and 0.01, enabling the optimizer to converge effectively without overshooting the minimum.
        This comprehensive fine-tuning process not only improved the model's accuracy but also contributed to its overall stability and reliability, ensuring that it remains a powerful tool for Alphabet Soup charity's decision-making framework.
        To further improve predictive performance and model interpretability, it is advisable to explore alternative machine learning models that may be better suited to the structure and characteristics of the dataset:
        Random Forest Classifier is ensemble method effectively handles categorical variables and provides clear insights into feature importance, supporting transparency and interpretability in the decision-making process.
        XGBoost Classifier is known for its high accuracy on structured data, XGBoost is particularly adept at managing class imbalances—making it a strong candidate for enhancing prediction reliabilit

