import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_data(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(data['year'], data['days'], marker='o', linestyle='-')
    plt.title('Year vs. Number of Frozen Days')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.grid(True)
    plt.savefig("plot.jpg")

def plot_data(filename):
    # Load the dataset
    data = pd.read_csv(filename)
    
    # Prepare the data for linear regression
    X = np.array([[1, x] for x in data['year']]).astype(np.int64)  # Q3a
    Y = data['days'].values.astype(np.int64)  # Q3b
    Z = X.T @ X  # Q3c
    I = np.linalg.inv(Z)  # Q3d
    PI = I @ X.T  # Q3e
    beta_hat = PI @ Y  # Q3f
    
    # Plotting the data and the linear regression model
    plt.figure(figsize=(10, 6))
    plt.plot(data['year'], data['days'], marker='o', linestyle='-', label='Actual Data')
    
    # Generating predictions for the linear regression line
    predictions = X @ beta_hat
    plt.plot(data['year'], predictions, label='Linear Regression', color='red')
    
    plt.title('Year vs. Number of Frozen Days with Linear Regression')
    plt.xlabel('Year')
    plt.ylabel('Number of Frozen Days')
    plt.grid(True)
    plt.legend()
    plt.savefig("plot.jpg")
    
    # Prediction for winter 2022-23
    xtest = 2022
    ytest_hat = beta_hat[0] + beta_hat[1] * xtest  # Q4: Prediction
    
    #Q5
    beta1_sign_symbol = ">" if beta_hat[1] > 0 else "<" if beta_hat[1] < 0 else "="
    interpretation = "If >, more ice days over years. If <, fewer ice days over years. If =, no change."
    
    # Predicting the year Lake Mendota will no longer freeze
    x_star = -beta_hat[0] / beta_hat[1] if beta_hat[1] != 0 else 'undefined'  # Q6a
    
    # Model Limitation Discussion
    if x_star != 'undefined':
        answer = "Based on the current model, Lake Mendota is predicted to stop freezing by the year {:.0f}, which may not be reliable due to extrapolation far beyond the data range and ignoring other environmental factors.".format(x_star)
    else:
        answer = "Prediction is undefined due to a zero or near-zero slope, indicating no significant trend in freezing days over the years."

    # Printing the matrices, vectors, and prediction
    print("Q3a:")
    print(X)
    print("Q3b:")
    print(Y)
    print("Q3c:")
    print(Z.astype(np.int64))
    print("Q3d:")
    print(I)
    print("Q3e:")
    print(PI)
    print("Q3f:")
    print(beta_hat)
    print("Q4: " + str(ytest_hat))  # Output the prediction
    print("Q5a: " + beta1_sign_symbol)
    print("Q5b: " + interpretation)
    print("Q6a: " + str(x_star))
    print("Q6b: " + answer)

if __name__ == "__main__":
    # Check if filename argument is provided
    if len(sys.argv) < 2:
        print("Usage: python3 hw5.py filename.csv")
        sys.exit(1)
    
    filename = sys.argv[1]
    plot_data(filename)
