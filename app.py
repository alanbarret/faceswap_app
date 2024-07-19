import streamlit as st
import requests
import replicate
from PIL import Image

# Function to upload file to tmpfiles.org and return the URL
def upload_file(file):
    files = {'file': file}
    response = requests.post('https://tmpfiles.org/api/v1/upload', files=files)
    
    if response.status_code == 200:
        response_data = response.json()
        download_url = response_data['data']['url'].replace('tmpfiles.org/', 'tmpfiles.org/dl/')
        return download_url
    else:
        return None

# Function to check if the uploaded file is a valid image
def is_valid_image(image_file):
    try:
        img = Image.open(image_file)
        img.verify()
        return True
    except Exception as e:
        st.error(f"Invalid image file: {e}")
        return False

# Function to perform prediction
def perform_prediction(input_data):
    try:
        output = replicate.run(
            "arabyai-replicate/roop_face_swap:11b6bf0f4e14d808f655e87e5448233cceff10a45f659d71539cafb7163b2e84",
            input=input_data
        )
        return output
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Streamlit app
st.title('Face Swap App')

# Upload swap image
st.header("Upload Swap Image")
uploaded_image = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])

# Upload target video
st.header("Upload Target Video")
uploaded_video = st.file_uploader('Choose a video', type=['mp4', 'mov', 'avi'])

image_url = None
video_url = None

if uploaded_image is not None:
    if is_valid_image(uploaded_image):
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        image_url = upload_file(uploaded_image)
        if image_url:
            st.success('Image uploaded successfully!')
            st.write('Image Download URL:', image_url)
        else:
            st.error('Failed to upload the image.')

if uploaded_video is not None:
    st.video(uploaded_video)
    video_url = upload_file(uploaded_video)
    if video_url:
        st.success('Video uploaded successfully!')
        st.write('Video Download URL:', video_url)
    else:
        st.error('Failed to upload the video.')

if image_url and video_url:
    st.header("Preview")
    st.image(uploaded_image, caption="Swap Image", use_column_width=True)
    st.video(uploaded_video, format='video/mp4', start_time=0)

    if st.button("Submit"):
        with st.spinner("Performing face swap..."):
            # Prepare input data
            input_data = {
                "swap_image": image_url,
                "target_video": video_url
            }

            # Perform prediction
            output_video_url = perform_prediction(input_data)

            if output_video_url:
                st.header("Output")
                st.video(output_video_url, format='video/mp4', start_time=0)
            else:
                st.error("Face swap prediction failed. Please check the input files and try again.")
else:
    if uploaded_image or uploaded_video:
        st.warning("Please upload valid files for both the swap image and the target video.")
