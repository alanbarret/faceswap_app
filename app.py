import streamlit as st
import replicate
import base64

# Load the REPLICATE_API_TOKEN from secrets.toml
REPLICATE_API_TOKEN = st.secrets["REPLICATE_API_TOKEN"]

st.title("Face Swap App")

# Function to convert file to base64 URL
def file_to_base64_url(file):
    if file:
        encoded_file = base64.b64encode(file.read()).decode('utf-8')
        return f"data:{file.type};base64,{encoded_file}"
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
    swap_image_url = file_to_base64_url(swap_image)
    target_video_url = file_to_base64_url(target_video)

    if swap_image_url and target_video_url:
        st.image(swap_image, caption="Swap Image", use_column_width=True)
        st.video(target_video, format='video/mp4', start_time=0)

        if st.button("Submit"):
            st.write("Processing...")

            # Make the prediction request using replicate.run()
            input_data = {
                "swap_image": swap_image_url,
                "target_video": target_video_url
            }

            try:
                output = replicate.run(
                    "arabyai-replicate/roop_face_swap:11b6bf0f4e14d808f655e87e5448233cceff10a45f659d71539cafb7163b2e84",
                    input=input_data
                )

                output_video_url = output  # Assuming output is a video URL
                st.header("Output")
                st.video(output_video_url, format='video/mp4', start_time=0)

            except Exception as e:
                st.error(f"Error: {e}")

# Show a warning if no files are uploaded
if not swap_image and not target_video:
    st.warning("Please upload a swap image and a target video to get started.")
