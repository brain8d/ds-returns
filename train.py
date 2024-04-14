import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, f1_score, ConfusionMatrixDisplay, classification_report


def train():
    '''function to train model for predicting returns'''
    
    filtered_df = pd.read_parquet("transactions-filtered-v02.parquet")

    # Set seed for reproducibility
    seed = 2020
    X = filtered_df.drop(["proportion_not_returned", "Returned"], axis = 1)
    y = filtered_df["Returned"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=seed, test_size=0.2)

    # Select relevant columns
    numeric_features = [0,1,3,5,6,7,11]
    categorical_features = [2,4,8,9,10]

    # Make pipeline with preprocessing
    numeric_transformer = make_pipeline(StandardScaler())
    categorical_transformer = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

    # Transform appropriate columns
    Preprocessor = ColumnTransformer(
        transformers=[
            ("numerical_transformer", numeric_transformer, numeric_features),
            ("categorical transformer", categorical_transformer, categorical_features)
        ], 
    )

    # Train model 1
    bst = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=1, objective='binary:logistic')
    bst_pipe = make_pipeline(Preprocessor, bst)
    bst_pipe.fit(X_train, y_train)

    # Get some metrics
    model = bst_pipe
    model_name = "XGBoost"
    print("Score", model.score(X_test, y_test))
    y_pred = model.predict(X_test)
    print("classification report", model_name)
    print(classification_report(y_test, y_pred))

    # Save the model
    joblib.dump(model, f"./model/{model_name}.joblib")