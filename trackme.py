import datetime
from ultralytics import YOLO
import cv2
from helper import create_video_writer
from deep_sort_realtime.deepsort_tracker import DeepSort
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict  # Add this import
import streamlit as st
import base64
import tempfile
import uuid


def get_video_download_link(video_path):
    with open(video_path, 'rb') as f:
        video_file = f.read()
    b64 = base64.b64encode(video_file).decode()
    href = f'<a href="data:file/mp4;base64,{b64}" download="{video_path.split("/")[-1]}">Download {video_path.split("/")[-1]}</a>'
    return href

def track():

    st.title('Object Detection and Tracking')
    # Initialize session state
    if 'start_button_clicked' not in st.session_state:
        st.session_state['start_button_clicked'] = False

    # Initialize session state
    if 'start_button_clicked' not in st.session_state:
        st.session_state['start_button_clicked'] = False

    # Streamlit UI Elements
    option = st.selectbox('Choose an option', ['Upload Video File', 'Select Camera Index'])

    if option == 'Upload Video File':
        uploaded_file = st.file_uploader("Upload your video file", type=["mp4"])
        if uploaded_file:
            st.session_state['start_button_clicked'] = st.button("Start", key='start_button')
            if st.session_state['start_button_clicked']:
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                video_cap = cv2.VideoCapture(tfile.name)

    elif option == 'Select Camera Index':
        camera_index = st.selectbox('Choose a camera index', list(range(0, 6)))
        st.session_state['start_button_clicked'] = st.button("Start", key='start_button')
        if st.session_state['start_button_clicked']:
            video_cap = cv2.VideoCapture(camera_index)



    # Main Loop
    if st.session_state.get('start_button_clicked', False):  # Initialize state if not done
        col1, col2 = st.columns([2,2])  # Define columns
        stop_button_placeholder = st.empty()
        CONFIDENCE_THRESHOLD = 0.8
        # Initialize dictionaries and lists
        class_count = {}
        class_time_series = {}
        time_list = []
        count_list = []

        colors = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(1000)]

        plt.ion()
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))
        plt.subplots_adjust(hspace=0.4)
        #frame_placeholder = st.empty()

        #video_cap = cv2.VideoCapture("test-video.mp4")
        #video_cap = cv2.VideoCapture(0)
        writer = create_video_writer(video_cap, "output.mp4")

        model = YOLO("yolov8m.pt")
        tracker = DeepSort(max_age=50)

        # Create a VideoCapture object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter('plot_video.mp4', fourcc, 1, (fig.canvas.get_width_height()[0], fig.canvas.get_width_height()[1]))

        # Main Loop
        if st.session_state['start_button_clicked']:
            frame_placeholder = col1.empty()
            plot_placeholder = col2.empty()  # Added this line
        stop_button_created = False
        while True:

            start = datetime.datetime.now()
            ret, frame = video_cap.read()
            if not ret:
                st.warning('End of Video.')
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
            #frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            ax[1].cla()
            #frame_placeholder.image(frame, channels="BGR", use_column_width=True)

            # Plot new data
            for idx, (label, timeseries) in enumerate(class_time_series.items()):
                ax[1].plot(timeseries, label=label, color=colors[idx % len(colors)])

            ax[1].legend()
            ax[1].set_title("Object Count By Class Type Over Time")
            ax[1].set_xlabel("Frame")
            ax[1].set_ylabel("Count")

            plt.tight_layout()
            plt.draw()
            #plt.pause(0.01)
            #st.pyplot(fig)
            #plot_placeholder.pyplot(fig)  # Added this line


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
            #cv2.imshow("Frame", frame)
            writer.write(frame)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            video_writer.write(img)

            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            plot_placeholder.image(img, channels="BGR", use_column_width=True)  # Added this line
            #col1.image(frame, channels="BGR", use_column_width=True)
            #col2.image(img, channels="BGR", use_column_width=True)  # Added this line


            #if cv2.waitKey(1) == ord("q"):
            #    break
            # Add a unique key to the stop button
            if not stop_button_created:
                stop_button = stop_button_placeholder.button('Stop', key="unique_stop_button_key")
                stop_button_created = True
            if stop_button:
                st.warning('Stopping the execution.')
                video_cap.release()
                writer.release()
                video_writer.release()
                cv2.destroyAllWindows()
                break
