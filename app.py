import streamlit as st
import requests
import replicate
import time

# Load the REPLICATE_API_TOKEN from secrets.toml
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

st.title("Face Swap App")

# Function to upload file to tmpfiles.org and return the URL
def upload_to_tmpfiles(file):
    files = {'file': file}
    response = requests.post("https://tmpfiles.org/api/v1/upload", files=files)
    if response.status_code == 200:
        return response.json().get('data', {}).get('url')
    else:
        st.error("Failed to upload file to tmpfiles.org")
        return None

# Upload swap image
st.header("Upload Swap Image")
swap_image = st.file_uploader("Choose a JPG, JPEG, or PNG image", type=["jpg", "jpeg", "png"])

# Upload target video
st.header("Upload Target Video")
target_video = st.file_uploader("Choose an MP4 video", type=["mp4"])

# Display uploaded files if available
if swap_image and target_video:
    st.header("Preview")
    swap_image_url = upload_to_tmpfiles(swap_image)
    target_video_url = upload_to_tmpfiles(target_video)

    if swap_image_url and target_video_url:
        st.image(swap_image, caption="Swap Image", use_column_width=True)
        st.video(target_video, format='video/mp4', start_time=0)

        if st.button("Submit"):
            st.write("Processing... This may take a while.")
            st.spinner("Performing face swap...")

            # Function to perform the prediction request with retries
            def perform_prediction(input_data, retries=3, delay=5):
                for attempt in range(retries):
                    try:
                        output = replicate.run(
                            "arabyai-replicate/roop_face_swap:11b6bf0f4e14d808f655e87e5448233cceff10a45f659d71539cafb7163b2e84",
                            input=input_data
                        )
                        return output
                    except Exception as e:
                        st.error(f"Attempt {attempt + 1} failed: {e}")
                        time.sleep(delay)
                st.error("All attempts failed. Please try again later.")
                return None

            # Prepare input data
            input_data = {
                "swap_image": swap_image_url,
                "target_video": target_video_url
            }

            # Perform prediction with retries
            output_video_url = perform_prediction(input_data)

            if output_video_url:
                st.header("Output")
                st.video(output_video_url, format='video/mp4', start_time=0)

# Show a warning if no files are uploaded
if not swap_image and not target_video:
    st.warning("Please upload a swap image and a target video to get started.")
