import streamlit as st
import requests
import base64
import time
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_TOKEN')

st.title("Face Swap App")

# Function to convert file to base64
def file_to_base64(file):
    return base64.b64encode(file.read()).decode('utf-8')

# Upload swap image
swap_image = st.file_uploader("Upload Swap Image", type=["jpg", "jpeg", "png"])
# Upload target video
target_video = st.file_uploader("Upload Target Video", type=["mp4"])

if swap_image and target_video:
    swap_image_base64 = file_to_base64(swap_image)
    target_video_base64 = file_to_base64(target_video)

    if st.button("Submit"):
        st.write("Processing...")

        # Make the prediction request
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            json={
                "version": "11b6bf0f4e14d808f655e87e5448233cceff10a45f659d71539cafb7163b2e84",
                "input": {
                    "swap_image": f"data:image/jpeg;base64,{swap_image_base64}",
                    "target_video": f"data:video/mp4;base64,{target_video_base64}"
                }
            },
            headers={
                "Authorization": f"Token {REPLICATE_API_TOKEN}",
                "Content-Type": "application/json"
            }
        )

        if response.status_code == 200:
            prediction = response.json()
            prediction_id = prediction['id']
            get_url = prediction['urls']['get']
            
            # Polling the status of the prediction
            status = prediction['status']
            while status not in ["succeeded", "failed", "canceled"]:
                time.sleep(5)  # Wait for 5 seconds before checking the status again
                status_response = requests.get(get_url, headers={
                    "Authorization": f"Token {REPLICATE_API_TOKEN}",
                }, verify=False)  # Disable SSL verification for status polling
                status_data = status_response.json()
                status = status_data['status']

            if status == "succeeded":
                output_url = status_data['output'][0]  # Assuming the output is a URL
                st.video(output_url)
            else:
                st.error("Error: " + status_data.get('error', 'Unknown error occurred.'))
        else:
            st.error("Error: " + response.text)
