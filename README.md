This project is an automated machine learning program designed to work with unknown datasets. It supports three tasks: classification, regression, and clustering.

The program automatically analyzes the dataset, handles missing values, removes unnecessary columns, processes numerical and categorical features, and tests different machine learning models to select the best-performing one. It also includes a time limit to help meet the hackathon's 60-second constraint.

Requirements: Python, pandas, NumPy, and scikit-learn.

To install the required libraries, open a terminal in the project folder and run:

pip install pandas numpy scikit-learn

How to run the program:

Place the CSV dataset in the same folder as automl.py, open a terminal in that folder, and run the appropriate command.

For classification, specify the dataset and the column to predict.

python automl.py --csv "file.csv" --task classification --target Target --time-limit 52

For regression, use:

python automl.py --csv "file.csv" --task regression --target Target --time-limit 52

For clustering, specify the required number of clusters:

python automl.py --csv "file.csv" --task clustering --clusters 3 --time-limit 52

Replace 3 with actual number wanted
Replace file.csv with the dataset filename and Target with the exact target column name.”

The program displays the selected model, execution time, and evaluation results. Classification reports accuracy, balanced accuracy, and macro F1. Regression reports R², RMSE, and MAE. Clustering reports cluster sizes and available silhouette scores.

To save the predictions and dataset profile in a JSON file, add --output results.json to the command.

For classification and regression, these terminal commands automatically split the dataset into 80% training and 20% testing. For separate training and evaluation datasets, the run() function can be imported into another Python script.

The program uses a default internal time budget of 52 seconds, although total execution time may be slightly longer due to startup and file loading.