# =========================
# app.py
# =========================

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    try:
        Model = request.form['Model']
        Variant = request.form['Variant']
        Fuel_Type = request.form['Fuel_Type']
        City_Mileage = float(request.form['City_Mileage'])
        Gears = float(request.form['Gears'])
        Front_Brakes = request.form['Front_Brakes']
        Rear_Brakes = request.form['Rear_Brakes']
        Wheelbase = float(request.form['Wheelbase'])
        Power_Steering = request.form['Power_Steering']
        Engine_Type = request.form['Engine_Type']
        Cruise_Control = request.form['Cruise_Control']
        Android_Auto = request.form['Android_Auto']
        Rain_Sensing_Wipers = request.form['Rain_Sensing_Wipers']
        Automatic_Headlamps = request.form['Automatic_Headlamps']
        Battery = request.form['Battery']
        Electric_Range = float(request.form['Electric_Range'])

        input_data = [[
            Model,
            Variant,
            Fuel_Type,
            City_Mileage,
            Gears,
            Front_Brakes,
            Rear_Brakes,
            Wheelbase,
            Power_Steering,
            Engine_Type,
            Cruise_Control,
            Android_Auto,
            Rain_Sensing_Wipers,
            Automatic_Headlamps,
            Battery,
            Electric_Range
        ]]

        prediction = model.predict(input_data)[0]

        predicted_price = round(prediction, 2)

        return render_template(
            'index.html',
            prediction_text=f'Estimated Car Price: ₹ {predicted_price}'
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text=f'Error: {str(e)}'
        )

if __name__ == "__main__":
    app.run(debug=True)