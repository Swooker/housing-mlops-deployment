import gradio as gr
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load("model.pkl")

# Define prediction function
def predict_price(area, bedrooms, bathrooms):
    input_data = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms]
    })
    prediction = model.predict(input_data)
    return f"Predicted Price: ${prediction[0]:,.2f}"

# Create Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (sq ft)"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms")
    ],
    outputs="text",
    title="Housing Price Predictor",
    description="Enter house details to predict price using a pre-trained linear regression model."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
