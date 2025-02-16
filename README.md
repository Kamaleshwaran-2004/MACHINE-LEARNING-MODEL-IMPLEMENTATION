 MACHINE-LEARNING-MODEL-IMPLEMENTATION

cOMPANY; CODETECH IT SOLUTIONS

NAME; KAMALESHWARAN.R

INTERS ID; CT08PPJ

DOMAIN; PYTHON

DURATION; 4 weeks  from January 25th, 2025 to February 25th, 2025.  

MENTOR; NEELA SANTOSH

DESCRIPTION EXPLANATION;

# Text preprocessing function

Data Preprocessing
1. Loading data: The script loads a CSV file named "spam.csv" containing SMS messages labeled as either "ham" or "spam".
2. Text preprocessing: The preprocess_text function converts messages to lowercase, removes numbers and punctuation, and removes stopwords.

Model Training and Evaluation
1. Splitting data: The data is split into training (80%) and testing sets (20%).
2. TF-IDF vectorization: The text data is converted into numerical features using TF-IDF vectorization.
3. Naive Bayes classification: A Naive Bayes classifier is trained on the training data.
4. Model evaluation: The model is evaluated on the testing data using accuracy score, classification report, and confusion matrix.

Model Testing
1. Testing with new messages: The predict_message function takes a new message, preprocesses it, and uses the trained model to predict whether it's "ham" or "spam".

Main Function
1. Loading data and training model: The main function loads the data, trains the model, and evaluates its performance.
2. Testing with a new message: The main function tests the model with a new message and prints the prediction result.

The script uses the following libraries:

- pandas for data manipulation
- numpy for numerical computations
- nltk for text preprocessing
- re for regular expressions
- string for string manipulation
- matplotlib and seaborn for visualization
- sklearn for machine learning tasks
- 
