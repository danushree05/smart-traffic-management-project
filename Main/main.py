import cv2
import time
import ast
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import pyautogui
from customtkinter import *

#YOLO trained Model
model = YOLO('../Models/best.pt')
    
# Global Declarations 
area = []
count_lst = []
window_width_opencv = 900
window_height_opencv = 600

# Tkinter GUI
gui = CTk()
gui.title("User Confirmation")

# Setting the Tkinter window size
window_width = 400
window_height = 285
gui.geometry(f"{window_width}x{window_height}")

# Getting the screen width and height (For Tkinter GUI)
screen_width = gui.winfo_screenwidth()
screen_height = gui.winfo_screenheight()

# Calculating the window position to center it on the screen (For Tkinter GUI)
x_pos = (screen_width - window_width) // 2
y_pos = (screen_height - window_height) // 2

# Get the screen size (For Open-CV Window)
screen_width_opencv, screen_height_opencv = pyautogui.size()

# Calculate the window position to center it on the screen (For Open-CV Window)
x_pos_opencv = (screen_width_opencv - window_width_opencv) // 2
y_pos_opencv = (screen_height_opencv - window_height_opencv) // 2


# Setting the window position
gui.geometry(f"+{x_pos}+{y_pos}")

# Setting the Style
style = ttk.Style()
style.configure("TButton", padding=10, relief="flat")
#style.configure("TLabel", font=("Helvetica", 14), foreground="red", padding=10)

areas = ['../Data/Areas/area.txt', '../Data/Areas/area2.txt', '../Data/Areas/area3.txt', '../Data/Areas/area4.txt']
video_path = ['../Data/Videos/Video_1.mp4', '../Data/Videos/Video_2.mp4', '../Data/Videos/Video_3.mp4', '../Data/Videos/Video_4.mp4']


probability_flag = 1



def time_allocation():
    #Time Allocation
    time = int(str(np.random.randint(1, 10, 1)[0]) + '000')
    return time



# Mouse event callback function
def get_coordinates(event, x, y, flags, param):
    image = param['image']  # Get the image from the parameter
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(area) <= 4:
            area.append((x, y))
            # Draw a red dot at the clicked point
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image - Select a Rectangular Area using Mouse Clicks", image)
        if(len(area) == 4):
            cv2.destroyAllWindows()
        if(len(area) < 4):
            cv2.waitKey(0)        

            
def capture():
    # Open the video file
    global video_path
    frames = []
    for video in video_path:
        cap = cv2.VideoCapture(video)
        
        frame_time_ms = time_allocation()

        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time_ms)

        # Read the frame at the specified time
        ret, frame = cap.read()

        # Check if the frame was successfully read
        if ret:
            frames.append(frame)
    
    return frames


def defining_area():
    cnt = 0
    global areas
    global area
    # Load the frames
    frame_list = capture()
    i = 0
    for frame in frame_list:
        image = cv2.resize(frame, (900, 600))

        # Create a window and set the mouse callback
        cv2.namedWindow("Image - Select a Rectangular Area using Mouse Clicks")
        cv2.setMouseCallback("Image - Select a Rectangular Area using Mouse Clicks", get_coordinates, {'image': image})

        # Move the Open-CV window to the calculated position
        cv2.moveWindow("Image - Select a Rectangular Area using Mouse Clicks", x_pos_opencv, y_pos_opencv)

        # Display the image
        cv2.imshow("Image - Select a Rectangular Area using Mouse Clicks", image)

        # Wait for the user to click on the image
        key = cv2.waitKey(0)
        if key == 27:
            messagebox.showerror("Alert", "Complete this step First")
            
        with open(areas[i], 'w') as f:
            f.write(str(area))
        i += 1
        cnt += 1
        if cnt < 4:
            area = []

            
def count_vehicles():
    model_frames = []
    global count_lst
    global areas
    total_vehicle_count = []
    total_time_count = []
    time_to_sleep = []
    i = 0
    result = []
    my_file = open('../Data/classes.txt', 'r')
    classes = my_file.read()
    class_lst = classes.split('\n')
    
    for file in areas:
        with open(file, 'r') as f:
            result.append(ast.literal_eval(f.readline()))
    
    frame_list = capture()
            
    for fram in frame_list:
        image = cv2.resize(fram, (900, 600))

        frame = model.predict(image)
        results = frame[0].boxes.data
        pixels = pd.DataFrame(results).astype('float')

        # Define the points for the polygon
        point1 = result[i][0]  # Top-left corner
        point2 = result[i][1]  # Top-right corner
        point3 = result[i][2]  # Bottom-right corner
        point4 = result[i][3]  # Bottom-left corner
        points = np.array([point1, point2, point3, point4], np.int32)

        for index, row in pixels.iterrows():

            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])

            d = int(row[5])
            c = class_lst[d]

            cx = int(x1 + x2) // 2
            cy = int(y1 + y2) // 2

            ret = cv2.pointPolygonTest(points, (cx, cy), False)

            if ret >= 0:
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 3, (255, 0, 255), -1)
                cv2.putText(image, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 2)

                count_lst.append(c)

        total_count = 0.0

        for vehicle in count_lst:
            if vehicle == 'car':
                total_count += 1.0
            elif vehicle == 'motorbike':
                total_count += 0.8
            elif vehicle == 'three wheelers -CNG-':
                total_count += 1.5
            elif vehicle == 'pickup':
                total_count += 2.0
            elif vehicle == 'auto rickshaw':
                total_count += 1.5
            elif vehicle == 'minivan':
                total_count += 2.0
            elif vehicle == 'bus':
                total_count += 2.0
            elif vehicle == 'bicycle':
                total_count += 1.6
            elif vehicle == 'truck':
                total_count += 2.5
            elif vehicle == 'van':
                total_count += 1.1
            elif vehicle == 'ambulance':
                total_count += 1.1

        
        cv2.polylines(image, [points], True, (255, 255, 0), 2)

        total_vehicles = str("Total Number of Vehicles : {}".format(len(count_lst)))
        time_allocated = str("Time Allocated : {0} seconds".format(int(total_count)))
        time_to_sleep.append(int(total_count))
        
        total_vehicle_count.append(total_vehicles)
        total_time_count.append(time_allocated)
        
        model_frames.append(image)
        
        count_lst = []
        i += 1
    
    image1 = cv2.resize(model_frames[0], (400, 300))
    image2 = cv2.resize(model_frames[1], (400, 300))
    image3 = cv2.resize(model_frames[2], (400, 300))
    image4 = cv2.resize(model_frames[3], (400, 300))
    
    # Combine the four images into one frame
    first_row = cv2.hconcat([image1, image2])
    second_row = cv2.hconcat([image3, image4])
    final_frame = cv2.vconcat([first_row, second_row])

    return final_frame, total_vehicle_count, total_time_count, time_to_sleep


def result_frame(frame):
    cv2.namedWindow("Result")
    cv2.moveWindow("Result", x_pos_opencv, y_pos_opencv)
    cv2.imshow("Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
        
def oldone(ambulance_flag):
    global screen_width
    global screen_height
    Result_final_frame, Result_total_vehicles, Result_time_allocated, time_for_sleep = count_vehicles()
    #gui.destroy()
    
    gui2 = CTk()
    
    # Setting the Style
    style = ttk.Style()
    style.theme_use("vista")
    style.configure("TButton", padding=10, relief="flat")
    
    gui2.title("Result Window")
    
    # Setting the Tkinter window size
    gui2_window_width = 740
    gui2_window_height = 460
    gui2.geometry(f"{gui2_window_width}x{gui2_window_height}")
    

    # Calculating the window position to center it on the screen (For Tkinter GUI)
    gui2_x_pos = (screen_width - gui2_window_width) // 2
    gui2_y_pos = (screen_height - gui2_window_height) // 2
    
    # Setting the window position
    gui2.geometry(f"+{gui2_x_pos}+{gui2_y_pos}")




    label1 = CTkLabel(gui2, text="Result for Frame 1", font=("Times", 24), text_color=('red'))
    label1.grid(row=0, column=1, padx=20, pady=20)
    
    vehicle_count_result_1 = CTkLabel(gui2, text=Result_total_vehicles[0], font=("Times", 18))
    vehicle_count_result_1.grid(row=1, column=1, padx=20, pady=5)
    
    vehicle_time_result_1 = CTkLabel(gui2, text=Result_time_allocated[0], font=("Times", 18))
    vehicle_time_result_1.grid(row=2, column=1, padx=20, pady=5)
    

    
    label2 = CTkLabel(gui2, text="Result for Frame 2", font=("Times", 24), text_color=('red'))
    label2.grid(row=0, column=3, padx=20, pady=20)
    
    vehicle_count_result_2 = CTkLabel(gui2, text=Result_total_vehicles[1], font=("Times", 18))
    vehicle_count_result_2.grid(row=1, column=3, padx=20, pady=5)
    
    vehicle_time_result_2 = CTkLabel(gui2, text=Result_time_allocated[1], font=("Times", 18))
    vehicle_time_result_2.grid(row=2, column=3, padx=20, pady=5)
    
    
    
    label_temp = CTkLabel(gui2, text="")
    label_temp.grid(row=3, column=1)
    label_temp = CTkLabel(gui2, text="")
    label_temp.grid(row=4, column=1)
    
    
    
    label3 = CTkLabel(gui2, text="Result for Frame 3", font=("Times", 24), text_color=('red'))
    label3.grid(row=5, column=1, padx=20, pady=20)
    
    vehicle_count_result_3 = CTkLabel(gui2, text=Result_total_vehicles[2], font=("Times", 18))
    vehicle_count_result_3.grid(row=6, column=1, padx=20, pady=5)
    
    vehicle_time_result_3 = CTkLabel(gui2, text=Result_time_allocated[2], font=("Times", 18))
    vehicle_time_result_3.grid(row=7, column=1, padx=20, pady=5)
    
    
    
    label4 = CTkLabel(gui2, text="Result for Frame 4", font=("Times", 24), text_color=('red'))
    label4.grid(row=5, column=3, padx=10, pady=20)
    
    vehicle_count_result_4 = CTkLabel(gui2, text=Result_total_vehicles[3], font=("Times", 18))
    vehicle_count_result_4.grid(row=6, column=3, padx=20, pady=5)
    
    vehicle_time_result_4 = CTkLabel(gui2, text=Result_time_allocated[3], font=("Times", 18))
    vehicle_time_result_4.grid(row=7, column=3, padx=20, pady=5)
    
    
    
    label_temp = CTkLabel(gui2, text="")
    label_temp.grid(row=11, column=1)
    label_temp = CTkLabel(gui2, text="")
    label_temp.grid(row=12, column=1)
    
    
    result_btn = CTkButton(gui2, text="Show Resultant Frames", command=lambda: result_frame(Result_final_frame), width=60, height=40, border_width=2, border_color='#FFCC70', font=('Times', 16))
    result_btn.grid(row=13, column=2, padx=20, pady=10)


    vehicle_count_results = [vehicle_count_result_1, vehicle_count_result_2, vehicle_count_result_3, vehicle_count_result_4]
    vehicle_time_results = [vehicle_time_result_1, vehicle_time_result_2, vehicle_time_result_3, vehicle_time_result_4]
    labels_to_change = [label1, label2, label3, label4]


    


    def update_labels(to_display, to_call):

        def counter(to_display, time):
            if time == 0:
                vehicle_time_results[to_display].configure(text="Time Alloted : 0 seconds", font=("Times", 18))
                gui2.after(500, lambda d=to_display+1, c=to_call+1: update_labels(d, c))
            else:
                text_display = "Time Alloted : " + str(time) + " seconds"
                vehicle_time_results[to_display].configure(text=text_display, font=("Times", 18))
                gui2.after(1000, lambda d=to_display, c=time-1: counter(d, c))

        global Result_final_frame, Result_total_vehicles, Result_time_allocated, time_for_sleep
        if(to_call == 4):
            Result_final_frame, Result_total_vehicles, Result_time_allocated, time_for_sleep = count_vehicles()
            update_labels(0, 0)

        elif to_display < 4:
            vehicle_count_results[to_display].configure(text=Result_total_vehicles[to_display], font=("Times", 18))
            vehicle_time_results[to_display].configure(text=Result_time_allocated[to_display], font=("Times", 18))
            labels_to_change[to_display].configure(text="GO", font=("Times", 24), text_color=('green'))
            for i in range(to_display, 4):
                if i != to_display:
                    vehicle_count_results[i].configure(text="Total Number of Vehicles : ----", font=("Times", 18))
                    vehicle_time_results[i].configure(text="Time Allocated :  : --------------", font=("Times", 18))
            if to_display == 0:
                labels_to_change[3].configure(text="Result for Frame 4", font=("Times", 24), text_color=('red'))
            else:
                labels_to_change[to_display-1].configure(text="Result for Frame {0}".format(to_display), font=("Times", 24), text_color=('red'))
            if(to_display == 3):
                counter(to_display, time_for_sleep[to_display])
            else:
                counter(to_display, time_for_sleep[to_display])


    def temp(probable_lane, p1, p2):
        global probability_flag
        for i in range(0, 4):
            if i != probable_lane:
                labels_to_change[i].configure(text="STOP", font=("Times", 24), text_color=('red'))

        labels_to_change[probable_lane].configure(text="AMBULANCE", font=("Times", 24), text_color=('green'))
        probability_flag = 0
        gui2.after(8000, lambda d=p1, c=p2: update_labels_for_ambulance(d, c))



    def update_labels_for_ambulance(to_display, to_call):

        def counter(to_display, time, flag, exceptional_time_for_ambulance):
            if flag == 1:
                if exceptional_time_for_ambulance > time-5:
                    text_display = "Time Alloted : " + str(exceptional_time_for_ambulance) + " seconds"
                    vehicle_time_results[to_display].configure(text=text_display, font=("Times", 18))
                    gui2.after(1000, lambda d=to_display, c=time: counter(d, c, 1, exceptional_time_for_ambulance-1))
                else:
                    gui2.after(500, lambda probable_lane=probable_lane, p1=to_display, p2=to_display: temp(probable_lane, p1, p2))
            else:
                if time == 0:
                    vehicle_time_results[to_display].configure(text="Time Alloted : 0 seconds", font=("Times", 18))
                    gui2.after(500, lambda d=to_display+1, c=to_call+1: update_labels_for_ambulance(d, c))
                else:
                    text_display = "Time Alloted : " + str(time) + " seconds"
                    vehicle_time_results[to_display].configure(text=text_display, font=("Times", 18))
                    gui2.after(1000, lambda d=to_display, c=time-1: counter(d, c, 0, 0))


        global Result_final_frame, Result_total_vehicles, Result_time_allocated, time_for_sleep
        global probability_flag
        probability_of_lane = {5: 0, 6: 3, 7: 2, 8: 3}

        if(to_call == 4):
            Result_final_frame, Result_total_vehicles, Result_time_allocated, time_for_sleep = count_vehicles()
            update_labels_for_ambulance(0, 0)

        elif to_display < 4:
            if to_display == 1 and probability_flag == 1:
                probable_lane = probability_of_lane[ambulance_lane_probability()]
                vehicle_count_results[to_display].configure(text=Result_total_vehicles[to_display], font=("Times", 18))
                vehicle_time_results[to_display].configure(text=Result_time_allocated[to_display], font=("Times", 18))
                labels_to_change[to_display].configure(text=f"GO", font=("Times", 24), text_color=('green'))
                
                labels_to_change[to_display-1].configure(text=f"Result for Frame {to_display}", font=("Times", 24), text_color=('red'))

                probability_flag = 0

                counter(to_display, time_for_sleep[to_display], 1, time_for_sleep[to_display])

            else:
                for i in range(to_display, 4):
                    if i != to_display:
                        vehicle_count_results[i].configure(text="Total Number of Vehicles : ----", font=("Times", 18))
                        vehicle_time_results[i].configure(text="Time Allocated :  : --------------", font=("Times", 18))
                        labels_to_change[i].configure(text="Result for Frame {}".format(i+1), font=("Times", 24), text_color=('red'))

                vehicle_count_results[to_display].configure(text=Result_total_vehicles[to_display], font=("Times", 18))
                vehicle_time_results[to_display].configure(text=Result_time_allocated[to_display], font=("Times", 18))
                labels_to_change[to_display].configure(text="GO", font=("Times", 24), text_color=('green'))
                for i in range(to_display, 4):
                    if i != to_display:
                        vehicle_count_results[i].configure(text="Total Number of Vehicles : ----", font=("Times", 18))
                        vehicle_time_results[i].configure(text="Time Allocated :  : --------------", font=("Times", 18))
                if to_display == 0:
                    labels_to_change[3].configure(text="Result for Frame 4", font=("Times", 24), text_color=('red'))
                else:
                    labels_to_change[to_display-1].configure(text="Result for Frame {0}".format(to_display), font=("Times", 24), text_color=('red'))
                if to_display == 3:
                    counter(to_display, time_for_sleep[to_display], 0, 0)
                elif to_display == 1:
                    counter(to_display, time_for_sleep[to_display] - 4, 0, 0)
                else:
                    counter(to_display, time_for_sleep[to_display], 0, 0)

                probability_flag = 1
    
    if ambulance_flag != 1:
        update_labels(0, 4)
    else:
        update_labels_for_ambulance(0, 4)

    gui2.mainloop()


def ambulance_lane_probability():
    ambulance_lane = int(np.random.randint(5, 9, 1)[0])
    return ambulance_lane


def newone():
    with open('../Data/Videos/area.txt', 'w') as file:
        pass
    while(len(area) < 4):
        defining_area()
    oldone(0)



label = CTkLabel(gui, text="Want to Test on New Frames or Old Frames", font=('Times', 21), text_color=('#FFCC70'))
new = CTkButton(gui, text="New", corner_radius=32, fg_color='#4158D0', hover_color='#6850C0', border_color='#FFCC70', border_width=2, height=50, width=150, font=('Times', 20), command=newone)
old = CTkButton(gui, text="Old", corner_radius=32, fg_color='#4158D0', hover_color='#6850C0', border_color='#FFCC70', border_width=2, height=50, width=150, font=('Times', 20), command=lambda: oldone(0))

new.grid(row=0, column=0, padx=10, pady=40)
old.grid(row=0, column=1, padx=10, pady=40)
label.grid(row=1, columnspan=2, padx=16, pady=5)

ambulance = CTkButton(gui, text="Emergency vehicles Demo", corner_radius=32, fg_color='#4158D0', hover_color='#6850C0', border_color='#FFCC70', border_width=2, height=50, width=150, font=('Times', 20), command=lambda: oldone(1))
ambulance.grid(row=3, columnspan=2, padx=16, pady=50)

gui.mainloop()