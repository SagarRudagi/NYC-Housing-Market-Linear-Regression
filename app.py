import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Set the app title
st.title("Simple Linear Regression")
st.subheader("NYC Housing Data")

# Write some text
# st.write("Let's predict the price of the house based on the area (sqft) of the property.")
# st.markdown('#')# Add an interactive slider
# slider_value = st.slider("Select a value", 0, 100, 50)
# st.write("The selected value is:", slider_value)

# Sample data for illustration
df = pd.read_csv("NY-Housing-SimpleData.csv")
df.columns = ["propertysqft","price"]
df.drop([1,7,304,823,2146,2148,3130,4623], axis=0, inplace=True) # Dropping outliers


# Create the Streamlit app

def play(params):
    slope = float(params[1])
    intercept = float(params[0])
    
    X = df["propertysqft"]
    y = df["price"]
    slope_min, slope_max = 0.0, 0.0
    intercept_min, intercept_max = 0.0, 0.0
    if slope < 0:
        slope_min  = 2 * slope 
    else:
        slope_max = 2 * slope

    if intercept < 0:
        intercept_min  = 2 * intercept 
    else:
        intercept_max = 2 * intercept

    slope_slider = st.slider("Slope", min_value=slope_min, max_value=slope_max, value=slope, step=0.1)
    slope_text = st.text_input("Slope (text)", value=str(slope_slider))
    intercept_slider = st.slider("Intercept", min_value=intercept_min, max_value=intercept_max, value=intercept, step=0.1)
    intercept_text = st.text_input("Intercept (text)", value=str(intercept_slider))

    # Update slider values from text boxes
    if slope_text:
        slope_slider = float(slope_text)
    if intercept_text:
        intercept_slider = float(intercept_text)

    # Update text box values from sliders
    slope_text = str(slope_slider)
    intercept_text = str(intercept_slider)

    # Predict y values
    predicted_y = slope_slider * X + intercept_slider

    # Calculate and display MSE
    mse = mean_squared_error(y, predicted_y)
    st.write(f"Mean Squared Error: {mse:.2f}")

    # Plot the data and predicted line
    fig, ax = plt.subplots()
    ax.scatter(X, y, label='True Data')
    ax.plot(X, predicted_y, color='red', label='Predicted Line')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    st.pyplot(fig)


def main():
    # Display the dataset
    
    st.write('Dataset:')
    st.dataframe(df)
    

    # Prepare the data for the model
    X = df['propertysqft'].values.reshape(-1, 1)
    y = df['price'].values

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    y_pred = model.predict(X)

    # Visualize the model
    fig, ax = plt.subplots()
    ax.scatter(X, y, color='blue', label='Actual Data')
    ax.plot(X, y_pred, color='red', label='Predicted Line')
    ax.set_xlabel("Area (SQFT)")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    # Display model parameters
    st.write('Model Parameters:')
    st.write('Intercept:', model.intercept_)
    st.write('Coefficient:', model.coef_[0])
    return [model.intercept_, model.coef_[0]]


if __name__ == '__main__':
    params = main()
    play(params)