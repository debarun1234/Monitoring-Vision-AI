import streamlit as st
import tensorflow as tf
import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_option_menu import option_menu
import pytz
from datetime import date
from email.message import EmailMessage
import ssl
import smtplib
import time

def check_custom_timeout():
    motion_start_time = None
    custom_timeout_threshold = 60

    if motion_start_time is not None:
        elapsed_time = time.time() - motion_start_time
        if elapsed_time > custom_timeout_threshold:
            print(f"Custom timeout triggered after {elapsed_time:.3f} seconds. Exiting.")
            return True
    return False

def save_video(frames, video_filename, fps):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

def save_frames(frames, output_directory, time_variable):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Create a subfolder inside the specified directory
    # Use the time_variable as the subfolder name
    time_variable = time_variable.replace(":", "_").replace(" ", "_")
    frames_folder_path = os.path.join(output_directory, time_variable)
    os.makedirs(frames_folder_path, exist_ok=True)

    for i, frame in enumerate(frames):
        # Assuming frame is in OpenCV format
        # If frames are in PIL format, you can skip the cv2 conversion
        frame_path = os.path.join(frames_folder_path, f"frame_{i}.png")
        cv2.imwrite(frame_path, frame)
        

def status_choose(mo_status, cnn_status, flag):
    if mo_status == cnn_status:
        status = mo_status
    else:
        status = "Running"
    
    return status


def get_log(video):
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 8
    machine_status = None
    status_chk = 0
    status_chk_f = 0
    status1 = ""
    status = ""
    chk_time = 0
    flag = 0
    alert_flag = 0
    chk_time_1 = 100
    status_temp = "Loading ..."
    clear = st.empty()
    st_count = 0
    count = 0
    i = 0
    IST = pytz.timezone('Asia/Kolkata') 
    record_frames = []
    rec_frames_count = 0
    rec_on = 0
    # Set parameters for motion detection
    motion_detected = True
    motion_threshold = 1000  # Adjust based on your environment 

    sender_email = "alertmintergraph@gmail.com"
    sender_password = "uqfq sjou kfxt qarb"
    recipient = ["instadatahelp@hotmail.com", "debarun.ghosh.121@gmail.com"] #for the alert message / mail sending email address

    # Create VideoCapture object using local video file path
    video_capture = cv2.VideoCapture(video)

    # Initialize the first frame
    ret, prev_frame = video_capture.read()
    prev_frame = cv2.resize(prev_frame, (512, 512))
    
    # Open a connection to the webcam (0 represents the default webcam)
    cap = cv2.VideoCapture(video)

    # Get the frames per second (fps) of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for capturing screenshots in seconds
    screenshot_interval = 2  # 1-second interval
    screenshot_frames = int(screenshot_interval * fps)  # Number of frames to wait for 1 second

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame1 = frame
            frame_high = frame
            frame_high1 = frame_high

            # Resize the frame to match the input size of the model
            frame = cv2.resize(frame, (512, 512))

            # Preprocess the frame
            img_array = image.img_to_array(frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make predictions
            prediction = model.predict(img_array, verbose=0)

            # Capture a screenshot at the specified interval
            if frame_count % screenshot_frames == 0:
                datetime_ist = datetime.now(IST)
                current_time = datetime_ist.strftime("%H:%M:%S")
                today = date.today()
                current_day = today.strftime("%A")
                screenshot_filename = f"screenshot_{current_time}.png_{current_day}"
                cv2.imwrite(screenshot_filename, frame)
                # Convert frames to grayscale
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate absolute difference between the current frame and the previous frame
                frame_diff = cv2.absdiff(gray_prev, gray_frame)

                # Apply threshold to the difference frame
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

                # Count the number of non-zero pixels in the thresholded frame
                count = cv2.countNonZero(thresh)

                # Motion detection logic
                if count > motion_threshold:
                    # Motion detected
                    if motion_detected:
                        mo_status = "Running"
                    motion_detected = True
                    motion_start_time = time.time()
                elif not motion_detected:
                    # No motion detected, but motion was detected previously
                    if check_custom_timeout():
                        motion_detected = False  # Reset motion detection status
                        mo_status = "Not Running"
                else:
                    # No motion detected
                    mo_status = "Not Running"

                if prediction > 0.98:
                    cnn_status = "Running"
                else:
                    cnn_status = "Not Running"

                # Update the previous frame
                prev_frame = frame

                status = status_choose(mo_status, cnn_status, flag)

                st.write(f"{current_time} - Screenshot captured: {screenshot_filename}, Machine {status}")
                
                if status == status1:
                    status_chk += 1
                else:
                    status_chk = 0
                    status_chk_f = 0
                    status_chk_rec = 0
                    alert_flag += 1
                status1 = status

            if st_count == 0:
                st_count = 1
            # Check for consecutive frames and update machine status
            if status_chk >= consecutive_frames_threshold:
                status_main = status
                        
            frame_count += 1

            if status_main == "Not Running":
                chk_time = int(current_time[3:5])
                if chk_time == chk_time_1 + 0 + i:
                    subject = "⚠️ Alert! Machine is not running ⚠️"
                    message = f"The Machine was observed not running first at {chk_time_1_act} IST. It's been {0+i} minutues, and we have observed the machine is still not running. Current time is {current_time} IST."
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                        smtp.login(sender_email, sender_password)
                        for rec in recipient:
                            em = EmailMessage()
                            em['From'] = sender_email
                            em['Subject'] = subject
                            em['To'] = rec
                            em.set_content(message)
                            smtp.sendmail(sender_email, rec, em.as_string())
                    
                    i += 5
                if status_chk == 0:
                    chk_time_1 = chk_time
                    chk_time_1_act = current_time

            if alert_flag >= 2:
            	if status_main == "Running":
                	if status_chk_f == 0:
                            subject = "⚠️ Alert! Machine has started running ⚠️"
                            message = f"The Machine was observed running first at {current_time} IST."
                            context = ssl.create_default_context()
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                                smtp.login(sender_email, sender_password)
                                for rec in recipient:
                                        em = EmailMessage()
                                        em['From'] = sender_email
                                        em['Subject'] = subject
                                        em['To'] = rec
                                        em.set_content(message)
                                        smtp.sendmail(sender_email, rec, em.as_string())
                            status_chk_f = 1
                    
                    elif status_chk_rec == 0 and rec_frames_count < fps*14:
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif status_chk_rec == 0 and rec_frames_count == fps*14:
                            rec_on = 0
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif rec_on == 0:
                            chk_time_2_act = current_time
                            output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine starts running"
                            save_frames(record_frames, output_directory, chk_time_2_act)
                            record_frames = []
                            rec_frames_count = 0
                            rec_on = 1
                            status_chk_rec = 1
                
            if status == "Not Running":
                if status_chk_rec == 0 and rec_frames_count < fps*14:
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif status_chk_rec == 0 and rec_frames_count == fps*14:
                    rec_on = 0
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif rec_on == 0:
                    chk_time_2_act = current_time
                    output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine stops"
                    save_frames(record_frames, output_directory, chk_time_2_act)
                    record_frames = []
                    rec_frames_count = 0
                    rec_on = 1
                    status_chk_rec = 1

        
        else:
            # Capture a screenshot at the specified interval
            if frame_count % screenshot_frames == 0:
                datetime_ist = datetime.now(IST)
                current_time = datetime_ist.strftime("%H:%M:%S")
                today = date.today()
                current_day = today.strftime("%A")
                screenshot_filename = f"screenshot_{current_time}.png_{current_day}"
                status = status
                frame = frame1
                frame = frame1
                frame_high = frame_high1


                st.write(f"{current_time} - Screenshot captured: {screenshot_filename}, Machine {status}")
                
                if status == status1:
                    status_chk += 1
                else:
                    status_chk = 0
                    status_chk_f = 0
                    alert_flag += 1
                status1 = status

            if st_count == 0:
                st_count = 1
            # Check for consecutive frames and update machine status
            if status_chk >= consecutive_frames_threshold:
                status_main = status
                noti_flag = 1
                    
            count = 1
            frame_count += 1

            if status_main == "Not Running":
                chk_time = int(current_time[3:5])
                if chk_time == chk_time_1 + 0 + i:
                    subject = "⚠️ Alert! Machine is not running ⚠️"
                    message = f"The Machine was observed not running first at {chk_time_1_act} IST. It's been {0+i} minutues, and we have observed the machine is still not running. Current time is {current_time} IST."
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                        smtp.login(sender_email, sender_password)
                        for rec in recipient:
                            em = EmailMessage()
                            em['From'] = sender_email
                            em['Subject'] = subject
                            em['To'] = rec
                            em.set_content(message)
                            smtp.sendmail(sender_email, rec, em.as_string())
                    
                    i += 5
                if status_chk == 0:
                    chk_time_1 = chk_time
                    chk_time_1_act = current_time

            if alert_flag >= 2:
            	if status_main == "Running":
                	if status_chk_f == 0:
                            subject = "⚠️ Alert! Machine has started running ⚠️"
                            message = f"The Machine was observed running first at {current_time} IST."
                            context = ssl.create_default_context()
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                                smtp.login(sender_email, sender_password)
                                for rec in recipient:
                                        em = EmailMessage()
                                        em['From'] = sender_email
                                        em['Subject'] = subject
                                        em['To'] = rec
                                        em.set_content(message)
                                        smtp.sendmail(sender_email, rec, em.as_string())
                            status_chk_f = 1
                    
                    elif status_chk_rec == 0 and rec_frames_count < fps*14:
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif status_chk_rec == 0 and rec_frames_count == fps*14:
                            rec_on = 0
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif rec_on == 0:
                            chk_time_2_act = current_time
                            output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine starts running"
                            save_frames(record_frames, output_directory, chk_time_2_act)
                            record_frames = []
                            rec_frames_count = 0
                            rec_on = 1
                            status_chk_rec = 1
                
            if status == "Not Running":
                if status_chk_rec == 0 and rec_frames_count < fps*14:
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif status_chk_rec == 0 and rec_frames_count == fps*14:
                    rec_on = 0
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif rec_on == 0:
                    chk_time_2_act = current_time
                    output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine stops"
                    save_frames(record_frames, output_directory, chk_time_2_act)
                    record_frames = []
                    rec_frames_count = 0
                    rec_on = 1
                    status_chk_rec = 1

    # Release the webcam capture object and close the OpenCV window
    cap.release()



def get_machine_status(video): 
    frame_count = 0
    running_frames = 0
    not_running_frames = 0
    skip_frames = 2  # Skip 2 frames in between each prediction
    consecutive_frames_threshold = 8
    machine_status = None
    status_chk = 0
    status_chk_f = 0
    status1 = ""
    status = ""
    chk_time = 0
    flag = 0
    alert_flag = 0
    chk_time_1 = 100
    status_temp = "Loading ..."
    clear = st.empty()
    st_count = 0
    count = 0
    i = 0
    IST = pytz.timezone('Asia/Kolkata') 
    record_frames = []
    rec_frames_count = 0
    rec_on = 0
    # Set parameters for motion detection
    motion_detected = True
    motion_threshold = 1000  # Adjust based on your environment

    sender_email = "alertmintergraph@gmail.com"
    sender_password = "uqfq sjou kfxt qarb"
    recipient = ["instadatahelp@hotmail.com", "deendayalpandey1@gmail.com"]
    #recipient = ["chandrapauldas01@gmail.com"]
    
    # Create VideoCapture object using local video file path
    video_capture = cv2.VideoCapture(video)

    # Initialize the first frame
    ret, prev_frame = video_capture.read()
    prev_frame = cv2.resize(prev_frame, (512, 512))
    
    # Open a connection to the webcam (0 represents the default webcam)
    cap = cv2.VideoCapture(video)

    # Get the frames per second (fps) of the webcam
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set the interval for capturing screenshots in seconds
    screenshot_interval = 2  # 1-second interval
    screenshot_frames = int(screenshot_interval * fps)  # Number of frames to wait for 1 second


    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            frame_high = frame
            frame_high1 = frame_high
            # Skip frames
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue

            # Resize the frame to match the input size of the model
            frame = cv2.resize(frame, (512, 512))
            frame1 = frame

            # Preprocess the frame
            img_array = image.img_to_array(frame)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make predictions
            prediction = model.predict(img_array, verbose=0)

            # Capture a screenshot at the specified interval
            if frame_count % screenshot_frames == 0:
                datetime_ist = datetime.now(IST)
                current_time = datetime_ist.strftime("%H:%M:%S")
                today = date.today()
                current_day = today.strftime("%A")
                screenshot_filename = f"screenshot_{current_time}.png"
                cv2.imwrite(screenshot_filename, frame)
                # Convert frames to grayscale
                gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Calculate absolute difference between the current frame and the previous frame
                frame_diff = cv2.absdiff(gray_prev, gray_frame)

                # Apply threshold to the difference frame
                _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

                # Count the number of non-zero pixels in the thresholded frame
                count = cv2.countNonZero(thresh)

                # Motion detection logic
                if count > motion_threshold:
                    # Motion detected
                    if motion_detected:
                        mo_status = "Running"
                    motion_detected = True
                    motion_start_time = time.time()
                elif not motion_detected:
                    # No motion detected, but motion was detected previously
                    if check_custom_timeout():
                        motion_detected = False  # Reset motion detection status
                        mo_status = "Not Running"
                else:
                    # No motion detected
                    mo_status = "Not Running"

                if prediction > 0.98:
                    cnn_status = "Running"
                else:
                    cnn_status = "Not Running"

                # Update the previous frame
                prev_frame = frame

                status = status_choose(mo_status, cnn_status, flag)

                if status == status1:
                    status_chk += 1
                else:
                    status_chk = 0
                    status_chk_f = 0
                    status_chk_rec = 0
                    alert_flag += 1
                status1 = status

            if st_count == 0:
                with clear.container():
                    st.write(f"Machine Status: {status_temp}")
                st_count = 1
            # Check for consecutive frames and update machine status
            if status_chk >= consecutive_frames_threshold:
                status_main = status
                with clear.container():
                    st.write(f"Machine Status: {status_main}")
                        
            frame_count += 1

            if status_main == "Not Running":
                chk_time = int(current_time[3:5])
                if chk_time == chk_time_1 + 0 + i:
                    subject = "⚠️ Alert! Machine is not running ⚠️"
                    message = f"The Machine was observed not running first at {chk_time_1_act} IST. It's been {0+i} minutues, and we have observed the machine is still not running. Current time is {current_time} IST."
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                        smtp.login(sender_email, sender_password)
                        for rec in recipient:
                            em = EmailMessage()
                            em['From'] = sender_email
                            em['Subject'] = subject
                            em['To'] = rec
                            em.set_content(message)
                            smtp.sendmail(sender_email, rec, em.as_string())
                    
                    i += 5
                if status_chk == 0:
                    chk_time_1 = chk_time
                    chk_time_1_act = current_time

            if alert_flag >= 2:
            	if status_main == "Running":
                	if status_chk_f == 0:
                            subject = "⚠️ Alert! Machine has started running ⚠️"
                            message = f"The Machine was observed running first at {current_time} IST."
                            context = ssl.create_default_context()
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                                smtp.login(sender_email, sender_password)
                                for rec in recipient:
                                        em = EmailMessage()
                                        em['From'] = sender_email
                                        em['Subject'] = subject
                                        em['To'] = rec
                                        em.set_content(message)
                                        smtp.sendmail(sender_email, rec, em.as_string())
                            status_chk_f = 1
                    
                    elif status_chk_rec == 0 and rec_frames_count < fps*14:
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif status_chk_rec == 0 and rec_frames_count == fps*14:
                            rec_on = 0
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif rec_on == 0:
                            chk_time_2_act = current_time
                            output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine starts running"
                            save_frames(record_frames, output_directory, chk_time_2_act)
                            record_frames = []
                            rec_frames_count = 0
                            rec_on = 1
                            status_chk_rec = 1
                
            if status == "Not Running":
                if status_chk_rec == 0 and rec_frames_count < fps*14:
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif status_chk_rec == 0 and rec_frames_count == fps*14:
                    rec_on = 0
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif rec_on == 0:
                    chk_time_2_act = current_time
                    output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine stops"
                    save_frames(record_frames, output_directory, chk_time_2_act)
                    record_frames = []
                    rec_frames_count = 0
                    rec_on = 1
                    status_chk_rec = 1

        else:
            # Capture a screenshot at the specified interval
            if frame_count % screenshot_frames == 0:
                datetime_ist = datetime.now(IST)
                current_time = datetime_ist.strftime("%H:%M:%S")
                today = date.today()
                current_day = today.strftime("%A")
                screenshot_filename = f"screenshot_{current_time}.png"
                status = status
                frame = frame1
                frame_high = frame_high1
                if status == status1:
                    status_chk += 1
                else:
                    status_chk = 0
                    status_chk_f = 0
                    alert_flag += 1
                status1 = status

            if st_count == 0:
                with clear.container():
                    st.write(f"Machine Status: {status_temp}")
                st_count = 1
            # Check for consecutive frames and update machine status
            if status_chk >= consecutive_frames_threshold:
                status_main = status
                noti_flag = 1
                with clear.container():
                    st.write(f"Machine Status: {status_main}")
                    
            count = 1
            frame_count += 1

            if status_main == "Not Running":
                chk_time = int(current_time[3:5])
                if chk_time == chk_time_1 + 0 + i:
                    subject = "⚠️ Alert! Machine is not running ⚠️"
                    message = f"The Machine was observed not running first at {chk_time_1_act} IST. It's been {0+i} minutues, and we have observed the machine is still not running. Current time is {current_time} IST."
                    context = ssl.create_default_context()
                    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                        smtp.login(sender_email, sender_password)
                        for rec in recipient:
                            em = EmailMessage()
                            em['From'] = sender_email
                            em['Subject'] = subject
                            em['To'] = rec
                            em.set_content(message)
                            smtp.sendmail(sender_email, rec, em.as_string())
                    
                    i += 5
                if status_chk == 0:
                    chk_time_1 = chk_time
                    chk_time_1_act = current_time

            if alert_flag >= 2:
            	if status_main == "Running":
                	if status_chk_f == 0:
                            subject = "⚠️ Alert! Machine has started running ⚠️"
                            message = f"The Machine was observed running first at {current_time} IST."
                            context = ssl.create_default_context()
                            with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as smtp:
                                smtp.login(sender_email, sender_password)
                                for rec in recipient:
                                        em = EmailMessage()
                                        em['From'] = sender_email
                                        em['Subject'] = subject
                                        em['To'] = rec
                                        em.set_content(message)
                                        smtp.sendmail(sender_email, rec, em.as_string())
                            status_chk_f = 1

                    elif status_chk_rec == 0 and rec_frames_count < fps*14:
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif status_chk_rec == 0 and rec_frames_count == fps*14:
                            rec_on = 0
                            record_frames.append(frame_high)
                            rec_frames_count += 1
                    elif rec_on == 0:
                            chk_time_2_act = current_time
                            output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine starts running"
                            save_frames(record_frames, output_directory, chk_time_2_act)
                            record_frames = []
                            rec_frames_count = 0
                            rec_on = 1
                            status_chk_rec = 1
            
            if status_main == "Not Running":
                if status_chk_rec == 0 and rec_frames_count < fps*14:
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif status_chk_rec == 0 and rec_frames_count == fps*14:
                    rec_on = 0
                    record_frames.append(frame_high)
                    rec_frames_count += 1
                elif rec_on == 0:
                    chk_time_2_act = current_time
                    output_directory = "C:/Users/Dell/Downloads/MachineRun-main (1)/MachineRun-main/Recordings/Machine stops"
                    save_frames(record_frames, output_directory, chk_time_2_act)
                    record_frames = []
                    rec_frames_count = 0
                    rec_on = 1
                    status_chk_rec = 1

    # Release the webcam capture object and close the OpenCV window
    cap.release()

# Set the layout
st.set_page_config(page_title="Machine Status App", page_icon="🤖", layout="wide")

# Main title
st.title("Machine Status Monitoring App")

# Load the saved model
model = tf.keras.models.load_model('machine_model_5jan10pm.h5') #type the name of the model you have trained .h5
video_path = "rtsp://admin:Admin@123@125.19.34.95:554/cam/realmonitor?channel=1&subtype=0"
#video_path = r"video_output_path_local (video-converter.com).mp4"

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Machine Status", "Machine runtime log"],
        icons = ["lightning-charge-fill", "list-columns"],
        default_index = 0)


if selected == "Machine Status":
    # Display the machine status
    st.subheader("Machine Status")
    # Call the function to get the machine status and log
    get_machine_status(video_path)

if selected == "Machine runtime log":
    # Display the machine status
    st.subheader("Machine runtime log")
    get_log(video_path)
