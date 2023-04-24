import streamlit as st
import os
import shutil
import subprocess
import time

# from streamlit_option_menu import option_menu


def main():
    cwd = os.getcwd()
    st.image("../website/banner.png", use_column_width=True)
    st.title("Food Image Recognition")  # <- change this align to center

    # st.subheader("Food Image Recognition")

    st.write("### AI Model That Recongnizes Food & Recommends Recipes")
    menu = ["Yolo", "RNN", "CNN"]
    choice = st.sidebar.selectbox("Algorithm", menu)

    col1, col2 = st.columns(2)

    if choice == "Yolo":
        # st.subheader("Upload an image")
        image_file = st.file_uploader(
            "Please upload an image", type=["jpg", "png", "jpeg"])
        if image_file is not None:
            col1.write("Original Image :camera:")
            col1.image(image_file, use_column_width=True,
                       caption="Uploaded Image")
            with open(os.path.join(f'{cwd}/user_uploads', image_file.name), "wb") as f:
                f.write(image_file.getbuffer())
                st.success("Saved image_file")

            # progress_text = "Operation in progress. Please wait."
            # my_bar = st.progress(0, text=progress_text)

            # for percent_complete in range(100):
            #     time.sleep(0.1)
            #     my_bar.progress(percent_complete + 1, text=progress_text)

            classified = make_inference(image_file)

            # add loading animation before displaying the image

            col2.write("Classified Image :hammer_and_wrench:")
            col2.image(classified, use_column_width=True,
                       caption="detected Image")

        folderPath = f'{cwd}/inferenced_imgs'
        # Check Folder is exists or Not
        if os.path.exists(folderPath):
            # Delete Folder code
            shutil.rmtree(folderPath)


def make_inference(SOURCE_PATH):
    cwd = os.getcwd()
    progress_text = "Inference in progress. Please wait."
    my_bar = st.progress(0)
    with st.spinner(progress_text):
        subprocess.run(
            ['python3', f'{cwd}/yolov5/detect.py', '--source',
             f'{cwd}/user_uploads', '--weights',
             f'{cwd}/yolov5/runs/train/exp2/weights/best.pt', '--project',
             f'{cwd}', '--name', 'inferenced_imgs'])
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1)
    infer_img = os.path.join(f'{cwd}/inferenced_imgs/', SOURCE_PATH.name)
    return infer_img

    cwd = os.getcwd()
    subprocess.run(
        ['python3', f'{cwd}/yolov5/detect.py', '--source',
         f'{cwd}/user_uploads', '--weights',
         f'{cwd}/yolov5/runs/train/exp2/weights/best.pt', '--project',
         f'{cwd}', '--name', 'inferenced_imgs'])
    infer_img = os.path.join(f'{cwd}/inferenced_imgs/', SOURCE_PATH.name)
    return infer_img


if __name__ == '__main__':
    main()
