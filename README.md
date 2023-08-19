import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from bokeh.plotting import figure, show
from bokeh.io import output_notebook

output_notebook()

train_data = pd.read_csv("train.csv")
ideal_data = pd.read_csv("ideal.csv")
test_data = pd.read_csv("test.csv")

print(train_data)
print(ideal_data)
print(test_data)

train_x = train_data[['x']]
train_y = train_data[['y1']]

print(train_x)
print(train_y)

# Training a linear regression model
linear_reg = LinearRegression()
linear_reg.fit(train_x, train_y)

# Training an SVR model
svr_reg = SVR()
svr_reg.fit(train_x, train_y.values.ravel())

# Using the mean squared error (MSE), mean absolute error (MAE), and R2 score to evaluate the models.
train_y_pred_linear = linear_reg.predict(train_x)
train_y_pred_svr = svr_reg.predict(train_x)

linear_mse = mean_squared_error(train_y, train_y_pred_linear)
svr_mse = mean_squared_error(train_y, train_y_pred_svr)

linear_mae = mean_absolute_error(train_y, train_y_pred_linear)
svr_mae = mean_absolute_error(train_y, train_y_pred_svr)

linear_r2 = r2_score(train_y, train_y_pred_linear)
svr_r2 = r2_score(train_y, train_y_pred_svr)

# Printing the models' MSE, MAE, and R2 scores.
print("Linear Regression:")
print("MSE:", linear_mse)
print("MAE:", linear_mae)
print("R2 Score:", linear_r2)

print("\nSVR:")
print("MSE:", svr_mse)
print("MAE:", svr_mae)
print("R2 Score:", svr_r2)

# Visualizing the linear regression
linear_plot = figure(title='Linear Regression', x_axis_label='x', y_axis_label='y')
linear_plot.scatter(train_x['x'], train_y['y1'], color='blue', legend_label='Actual')
linear_plot.line(train_x['x'], train_y_pred_linear.flatten(), color='red', legend_label='Linear Regression')
show(linear_plot)

# Visualizing the SVR
svr_plot = figure(title='SVR', x_axis_label='x', y_axis_label='y')
svr_plot.scatter(train_x['x'], train_y['y1'], color='blue', legend_label='Actual')
svr_plot.line(train_x['x'], train_y_pred_svr.flatten(), color='green', legend_label='SVR')
show(svr_plot)
