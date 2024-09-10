from flask import Flask, render_template, request  
import plotly.graph_objs as go  
import pandas as pd  
import numpy as np  
from sklearn.preprocessing import MinMaxScaler  
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense, LSTM, Dropout  
from statsmodels.tsa.arima.model import ARIMA  
  
app = Flask(__name__)  
  
def determine_season_from_date(date):  
   month = date.month  
   if month in [12, 1, 2]:  
      return 'Winter'  
   elif month in [3, 4, 5]:  
      return 'Autumn'  
   elif month in [6, 7, 8]:  
      return 'Summer'  
   elif month in [9, 10, 11]:  
      return 'Monsoon'  
  
# Load the dataset  
df = pd.read_csv('commodity_prices.csv')  
df['date'] = pd.to_datetime(df['date'])  
df.set_index('date', inplace=True)  
  
# Add seasonality feature  
df['month'] = df.index.month  
df['season'] = df.index.map(determine_season_from_date)  
  
# Function to preprocess data for LSTM  
def preprocess_data_lstm(data, n_steps):  
   # Normalize the numerical features (like 'close' price)  
   scaler = MinMaxScaler(feature_range=(0, 1))  
   numerical_data = scaler.fit_transform(data[['close']])  
  
   # One-Hot Encode the categorical 'season' feature  
   encoded_seasons = pd.get_dummies(data['season']).values  
  
   # Combine the numerical data and the encoded season data  
   combined_data = np.hstack((numerical_data, encoded_seasons))  
  
   X, y = [], []  
   for i in range(n_steps, len(combined_data)):  
      X.append(combined_data[i - n_steps:i])  # Sequence of n_steps (past data)  
      y.append(combined_data[i, 0])  # The close price we are trying to predict  
  
   X, y = np.array(X), np.array(y)  
  
   # Reshape X to be [samples, time_steps, features]  
   X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))  
   return X, y, scaler  
  
# Function to predict prices using LSTM  
def predict_prices_lstm(data, n_steps, days_to_predict):  
   # Preprocess data for LSTM  
   X_train, y_train, scaler = preprocess_data_lstm(data, n_steps)  
  
   # Build and train the LSTM model  
   model = Sequential()  
   model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))  
   model.add(Dropout(0.2))  
   model.add(LSTM(units=50))  
   model.add(Dropout(0.2))  
   model.add(Dense(1))  
   model.compile(loss='mean_squared_error', optimizer='adam')  
   model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)  
  
   # Make predictions  
   predicted_prices = []  
   last_n_days = data[-n_steps:]  
  
   for _ in range(days_to_predict):  
      # Prepare the last n_steps as input to the model  
      X_input = np.reshape(scaler.transform(last_n_days[['close']]), (1, n_steps, 1))  
      pred_price = model.predict(X_input)  
      predicted_prices.append(pred_price[0][0])  
  
      # Update the last n_days for the next prediction (use the predicted value as the new last day)  
      new_data = pd.DataFrame({'close': [pred_price[0][0]], 'month': [last_n_days['month'].iloc[-1]], 'season': [last_n_days['season'].iloc[-1]]})  
      last_n_days = pd.concat([last_n_days.iloc[1:], new_data], ignore_index=True)  
  
   return predicted_prices  
  
# ARIMA model prediction function  
def predict_prices_arima(data, days_to_predict):  
   model = ARIMA(data, order=(5, 1, 0))  # ARIMA(5,1,0) is just an example, adjust as needed  
   model_fit = model.fit()  
  
   forecast = model_fit.forecast(steps=days_to_predict)  
   return forecast  
  
@app.route('/')  
def home():  
   commodities = df['commodity'].unique()  
   return render_template('index.html', commodities=commodities)  
  
@app.route('/forecast', methods=['GET', 'POST'])  
def forecast():  
   if 'commodity' in request.args:  
      commodity = request.args.get('commodity')  
      forecast_days = int(request.args.get('forecast_days'))  
      threshold = float(request.args.get('threshold'))  
      prediction_type = request.args.get('prediction_type')  # "day", "month", "season"  
      model_type = request.args.get('model_type')  # "LSTM", "ARIMA"  
  
      # Data for model (including season feature)  
      commodity_data = df[df['commodity'] == commodity][['close', 'month', 'season']]  
      commodity_data = commodity_data.dropna()  
  
      n_steps = 500  
  
      if model_type == 'LSTM':  
        if len(commodity_data) > n_steps:  
           forecasted_prices = predict_prices_lstm(commodity_data, n_steps, forecast_days)  
        else:  
           return "Error: Not enough data for forecasting after cleaning."  
  
      elif model_type == 'ARIMA':  
        if len(commodity_data) > n_steps:  
           forecasted_prices = predict_prices_arima(commodity_data['close'], forecast_days)  
        else:  
           return "Error: Not enough data for ARIMA model."  
  
      # Generate forecast dates  
      last_date = commodity_data.index[-1]  
      forecast_dates = pd.date_range(last_date, periods=forecast_days + 1)  
  
      # Assign seasons to forecast dates  
      forecast_seasons = [determine_season_from_date(date) for date in forecast_dates]  
  
      # Create a DataFrame for forecasted prices and seasons  
      forecast_df = pd.DataFrame({  
        'date': forecast_dates,  
        'predicted_price': forecasted_prices,
        'season': forecast_seasons   
      })   
   
      # Plot the forecasted prices   
      fig = go.Figure(data=[go.Scatter(x=forecast_df['date'], y=forecast_df['predicted_price'])])   
      fig.update_layout(title='Forecasted Prices for ' + commodity, xaxis_title='Date', yaxis_title='Price')   
   
      # Render the plot   
      return render_template('forecast.html', commodity=commodity, forecast_days=forecast_days, threshold=threshold, prediction_type=prediction_type, model_type=model_type, plot_div=fig.to_html(full_html=False))   
   
   else:   
      return "Error: Please select a commodity."   
   
if __name__ == '__main__':   
   app.run(debug=True)