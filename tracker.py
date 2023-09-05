import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict  # Add this import
import tempfile
import streamlit as st
import shutil
import base64



def run_detection_on_video(video_file):
    # 1. Initialize session state
    if 'stop_button_clicked' not in st.session_state:
        st.session_state.stop_button_clicked = False

    # 2. Streamlit button to stop detection
    if st.button("Stop Detection"):
        st.session_state.stop_button_clicked = True

    CONFIDENCE_THRESHOLD = 0.8

    # Initialize dictionaries and lists
    class_count = {}
    #class_time_series = {}
    class_time_series = defaultdict(list)

    time_list = []
    count_list = []

    #colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(1000)]
    colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    #tfile = tempfile.NamedTemporaryFile(delete=False)
    #tfile.write(video_file.read())

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
        print(f"Saved video to {tfile.name}")  # Debug line

        video_cap = cv2.VideoCapture(tfile.name)
        if not video_cap.isOpened():
            print("Could not open video")
            return


    #cap = cv2.VideoCapture(tfile.name)

    #video_cap = cv2.VideoCapture(video_file)
    #video_cap = cv2.VideoCapture(0)
    writer = create_video_writer(video_cap, "video_output.mp4")

    model = YOLO("yolov8x.pt")
    tracker = DeepSort(max_age=50)

    # Create a VideoCapture object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('plot_video.mp4', fourcc, 1, (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))



    # Initialize Streamlit layout
    #col1, col2 = st.columns(2)  # or st.columns(2) in newer Streamlit versions

    # Initialize your Matplotlib figure and axis
    #fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    while True:
        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        if not ret:
            st.warning("Failed to get video frame.")
            break

        detections = model(frame)[0]
        results = []
        # Use defaultdict to automatically handle new classes
        class_count = defaultdict(int)
        obj_count = 0

        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = map(int, data[:4])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

            obj_count += 1
            class_name = model.names[class_id]

            normalized_class_name = class_name.rstrip('s')
            #class_count[normalized_class_name] = class_count.get(normalized_class_name, 0) + 1
            #class_count[class_name] += 1

                # either this
            class_count[normalized_class_name] = class_count.get(normalized_class_name, 0) + 1


        # Update the time series data
        for class_name, count in class_count.items():
            if class_name not in class_time_series:
                class_time_series[class_name] = []
            class_time_series[class_name].append(count)

            class_time_series[normalized_class_name].append(class_count.get(normalized_class_name, 0))

        time_list.append(start)
        count_list.append(obj_count)


        ax[0].cla()
        ax[0].plot(time_list, count_list)
        ax[0].set_title("Object Count Over Time")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Count")

        ax[1].cla()
        # Plot new data
        for idx, (label, timeseries) in enumerate(class_time_series.items()):
            ax[1].plot(timeseries, label=label, color=colors[idx % len(colors)])

        ax[1].legend()
        ax[1].set_title("Object Count By Class Type Over Time")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Count")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        # Update your graph and display it in column 2
        #ax.plot([0, 1], [0, 1])  # Dummy data

        #col1.image(frame, channels='BGR', use_column_width=True)
        #col2.pyplot(fig)
        col1, col2 = st.columns([2,2])  # Define columns
        with col1:
            st.image(frame, channels='BGR', use_column_width=True)  # Show video frame
        with col2:
            st.pyplot(fig)  # Show plot


        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)


        #imS = cv2.resize(frame, (1000, 1000))                # Resize image
        #cv2.imshow("Frame", imS)
        cv2.imshow("Frame", frame)
        writer.write(frame)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        video_writer.write(img)

        if cv2.waitKey(1) == ord("q"):
            break

        if st.session_state.stop_button_clicked:
            st.session_state.stop_button_clicked = False  # Reset for future runs
            break


    video_cap.release()
    # Clear the figure
    plt.close(fig)
    writer.release()
    cv2.destroyAllWindows()

     # Only show the download links if the stop button was clicked
    if st.session_state.get('stop_button_clicked', False):
        def get_video_download_link(video_path, download_name):
            with open(video_path, 'rb') as f:
                video_file = f.read()
            video_b64 = base64.b64encode(video_file).decode()
            href = f'<a href="data:file/mp4;base64,{video_b64}" download="{download_name}.mp4">Download {download_name}.mp4</a>'
            return href

        # Provide the download link for the video output
        st.markdown(get_video_download_link('camera_output.mp4', 'video_output'), unsafe_allow_html=True)

        # Provide the download link for the plot video
        st.markdown(get_video_download_link('plot_video.mp4', 'plot_video'), unsafe_allow_html=True)





def run_detection_on_camera():
    CONFIDENCE_THRESHOLD = 0.8

    # Initialize session state
    if 'camera_index' not in st.session_state:
        st.session_state.camera_index = None  # Default to None to force user selection

    if 'start_tracking' not in st.session_state:
        st.session_state.start_tracking = False

    if 'stop_button_clicked' not in st.session_state:
        st.session_state.stop_button_clicked = False


    # Message to instruct the user
    st.text("Please select a camera index from the dropdown below.")

    # Dropdown for camera index selection
    camera_index = st.selectbox("Select Camera Index:", ["", 0, 1, 2, 3, 4, 5])

    if camera_index != "":
        st.session_state.camera_index = int(camera_index)

    # Button to start tracking
    if st.button("Start Tracking"):
        if st.session_state.camera_index is not None:
            st.session_state.start_tracking = True
        else:
            st.warning("Please select a camera index before starting tracking.")

    if st.session_state.start_tracking:
        if st.session_state.camera_index is not None:
            try:
                video_cap = cv2.VideoCapture(st.session_state.camera_index)
                if not video_cap.isOpened():
                    st.error("Failed to open camera. Try another index.")
                    return
            except Exception as e:
                st.error(f"An error occurred: {e}")
                return
    # Initialize dictionaries and lists
    class_count = {}
    #class_time_series = {}
    class_time_series = defaultdict(list)

    time_list = []
    count_list = []

    #colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(1000)]
    colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

    plt.ion()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)

    video_cap = cv2.VideoCapture(camera_index)
    #video_cap = cv2.VideoCapture(0)
    writer = create_video_writer(video_cap, "camera_output.mp4")

    model = YOLO("yolov8x.pt")
    tracker = DeepSort(max_age=50)

    # Create a VideoCapture object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('plot_video.mp4', fourcc, 1, (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))

    if st.button("Stop Detection"):
        st.session_state.stop_button_clicked = True


    col1, col2 = st.columns(2)  # or st.columns(2) in newer Streamlit versions
    while True:

        start = datetime.datetime.now()
        ret, frame = video_cap.read()
        if not ret:
            break

        detections = model(frame)[0]
        results = []
        # Use defaultdict to automatically handle new classes
        class_count = defaultdict(int)
        obj_count = 0

        for data in detections.boxes.data.tolist():
            confidence = data[4]
            if float(confidence) < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = map(int, data[:4])
            class_id = int(data[5])
            results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, class_id])

            obj_count += 1
            class_name = model.names[class_id]

            normalized_class_name = class_name.rstrip('s')
            #class_count[normalized_class_name] = class_count.get(normalized_class_name, 0) + 1
            #class_count[class_name] += 1

                # either this
            class_count[normalized_class_name] = class_count.get(normalized_class_name, 0) + 1


        # Update the time series data
        for class_name, count in class_count.items():
            if class_name not in class_time_series:
                class_time_series[class_name] = []
            class_time_series[class_name].append(count)

            class_time_series[normalized_class_name].append(class_count.get(normalized_class_name, 0))

        time_list.append(start)
        count_list.append(obj_count)

        ax[0].cla()
        ax[0].plot(time_list, count_list)
        ax[0].set_title("Object Count Over Time")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("Count")

        ax[1].cla()
        # Plot new data
        for idx, (label, timeseries) in enumerate(class_time_series.items()):
            ax[1].plot(timeseries, label=label, color=colors[idx % len(colors)])

        ax[1].legend()
        ax[1].set_title("Object Count By Class Type Over Time")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Count")

        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)

        col1.image(frame, channels='BGR', use_column_width=True)
        col2.pyplot(fig)

        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            ltrb = track.to_ltrb()
            xmin, ymin, xmax, ymax = map(int, ltrb)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        end = datetime.datetime.now()
        fps = f"FPS: {1 / (end - start).total_seconds():.2f}"
        cv2.putText(frame, fps, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 8)


        #imS = cv2.resize(frame, (1000, 1000))                # Resize image
        #cv2.imshow("Frame", imS)
        cv2.imshow("Frame", frame)
        writer.write(frame)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        video_writer.write(img)

        if cv2.waitKey(1) == ord("q"):
            break

        if st.session_state.stop_button_clicked:
            st.session_state.stop_button_clicked = False  # Reset for future runs
            break

    video_cap.release()
    writer.release()
    cv2.destroyAllWindows()

     # Only show the download links if the stop button was clicked
    if st.session_state.get('stop_button_clicked', False):
        def get_video_download_link(video_path, download_name):
            with open(video_path, 'rb') as f:
                video_file = f.read()
            video_b64 = base64.b64encode(video_file).decode()
            href = f'<a href="data:file/mp4;base64,{video_b64}" download="{download_name}.mp4">Download {download_name}.mp4</a>'
            return href

        # Provide the download link for the video output
        st.markdown(get_video_download_link('camera_output.mp4', 'video_output'), unsafe_allow_html=True)

        # Provide the download link for the plot video
        st.markdown(get_video_download_link('plot_video.mp4', 'plot_video'), unsafe_allow_html=True)

