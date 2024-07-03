import cv2
import numpy as np
import datetime
import os
import socket
import firebase_script
import csv
from recording_script import record_video
from firebase_script import FirebaseObserver
import threading

stop_event = threading.Event()

# Start the recording thread and pass the stop event to it
recording_thread = threading.Thread(target=record_video, args=(stop_event,))
recording_thread.start()
video_dir = "/home/vaffa/Desktop/AI (copy)/recordings"  # Replace with your path to video directory
csv_dir = "/home/vaffa/Desktop/AI (copy)/csv"  # Replace with your path to CSV directory
ssid_to_check = "ORBI"  # Replace with your SSID name

# Create CSV directory if it doesn't exist
os.makedirs(csv_dir, exist_ok=True)

# Start observing in a separate thread
#path_to_watch = "/media/vaffa/KingstonUSB/Mozee2/AI (copy)"
firebase_observer = FirebaseObserver(video_dir, csv_dir, ssid_to_check)
firebase_observer.start()

# Function to generate a new CSV filename with timestamp
def get_csv_filename():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(csv_dir, f'classification_results_{timestamp}.csv')

# Function to write a single batch of face data to CSV
def write_faces_to_csv(faces, csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Gender', 'Age'])  # Write header
        for face in faces:
            writer.writerow(face)


def check_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check internet connection by trying to connect to a host.
    Google's DNS (8.8.8.8).
    """
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError:
        pass
    return False

def get_current_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Initialize webcam for camera 1
cap1 = cv2.VideoCapture(0)
if not cap1.isOpened():
    print("Error: could not open camera 1")
    exit()


# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('/home/vaffa/Desktop/AI (copy)/models/haarcascade_frontalface_default.xml')

age_model_path = "/home/vaffa/Desktop/AI (copy)/models/deploy_age.prototxt"
age_weights_path = "/home/vaffa/Desktop/AI (copy)/models/age_net.caffemodel"
gender_model_path = "/home/vaffa/Desktop/AI (copy)/models/deploy_gender.prototxt"
gender_weights_path = "/home/vaffa/Desktop/AI (copy)/models/gender_net.caffemodel"

# Load pre-trained models for age and gender detection
age_net = cv2.dnn.readNetFromCaffe(age_model_path, age_weights_path)
gender_net = cv2.dnn.readNetFromCaffe(gender_model_path, gender_weights_path)


# Define age and gender categories
AGE_BUCKETS=["(0-4)","(5-9)","(10-14)","(15-19)","(20-24)","(25-29)","(30-34)","(35-39)","(40-44)","(45-49)","(50-54)","(55-59)","(60-64)","(65-69)","(70-74)","(75-79)","(80-84)","(85-89)","(90-94)","(95-99)","(100+)"]
GENDERS = ['Male', 'Female']

script_directory = os.path.dirname(__file__)


#import pdb; pdb.set_trace()

def add_timestamp(frame):
    current_time = datetime.datetime.now()
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

#Added this - Ryan



class Person():
    def __init__(self, box, frame):
        self.box = box  # face location
        self.tracker = cv2.TrackerKCF_create()  # creates a tracker
        self.tracker.init(frame, box)  # and initializes it with the frame and bbox
        self.gender_list = []  # Here is where the gender predictions are going to go
        self.age_list = []  # Here's where the age predictions will go
        self.lost = False   # If true, then this face is lost from the frame
        self.lost_frame = -1 #Here is where we'll track when the face was lost

    def update_position(self, box, frame):
        self.box = box
    def mode(self, arr):
        if (arr is None):
            return "None was used as an input instead of an array"
        if len(arr) == 0:
            return "There are no elements in this array"
        return max(set(arr), key=arr.count)

    def get_gender(self): # Gets gender of passenger by taking mode of gender predictions
        return self.mode(self.gender_list)
    def get_age(self): # Gets age of passenger by taking mode of age predictions
        return self.mode(self.age_list)

class Person_List():
    def __init__(self, age_net, gender_net):
        self.current_persons = [] #People we're tracking
        self.lost_persons = [] #People we've lost track of momentarily
        self.total_persons = [] #People that
        
        #I've added the age buckets and genders here since they aren't very large
        self.AGE_BUCKETS=["(0-4)","(5-9)","(10-14)","(15-19)","(20-24)","(25-29)","(30-34)","(35-39)","(40-44)","(45-49)","(50-54)","(55-59)","(60-64)","(65-69)","(70-74)","(75-79)","(80-84)","(85-89)","(90-94)","(95-99)","(100+)"]
        self.GENDERS = ['Male', 'Female']

        #Added the age and gender nets to Person_list
        self.age_net = age_net
        self.gender_net = gender_net

        #Added frame number
        self.frame_num = 0

        #Here's the data we can add to Firebase
        self.batch = []

    def addPerson(self, p):
        self.current_persons.append(p)
    def addLostPerson(self, p, frame):
        self.lost_persons.append(p)
        p.lost = True
        p.lost_frame = frame
    def last_person(self):
        return self.current_persons[-1]

    def check_if_gone(self): #work on this 11/20
        # Checks if the face is mising for a long time and if it is, remove the face from the lost faces file

        curr_frame = self.frame_num

        wait_frames = 10  # How many frames to wait for the person to return

        for i, person in reversed(list(enumerate(self.lost_persons))):
            if (curr_frame - person.lost_frame) > wait_frames:
                per = self.lost_persons.pop(i)
                self.batch.append([get_current_datetime(), per.get_gender(), per.get_age()])
    def update_frame(self, frame_count):
        self.frame_num = frame_count
    def make_prediction(self, blob, person, type):
        if self.frame_num % 10 != 0: #Limits measurements to every 20 or certain number of frames
            return
        if type == "gender":
        	self.make_gender_prediction(blob, person)
        if type == "age":
        	self.make_age_prediction(blob, person)
        
    def make_gender_prediction(self, blob, person):
        # Perform gender detection
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDERS[gender_preds[0].argmax()]
        person.gender_list.append(gender)

    def make_age_prediction(self, blob, person):
        # Perform age detection
        self.age_net.setInput(blob)
        age_preds = age_net.forward()
        i = age_preds[0].argmax()
        age = AGE_BUCKETS[i] # make changes here
        person.age_list.append(age)



    def in_range(self, face_id):#work on this 11/20



        d = 50 #named it 'd' bc when i named the variable distance, the compiler confused the variable and the function
        for i, person in reversed(list(enumerate(self.lost_persons))):
            if distance(center(face_id), center(person.box)) < d:
                p = self.lost_persons.pop(i)
                p.lost = False
                p.lost_frame = -1
                self.current_persons.append(p)
                return True

        return False

    '''def in_range_v2(self, face_id):#work on this 11/20
        #for i, person in reversed(list(enumerate(Person_Handler.current_persons))): # come back

        distance = 50
        for i, person in reversed(list(enumerate(self.lost_persons))):
            if distance((200,200), center(person.box)) < distance:
                p = self.lost_persons.pop(i)
                p.lost = False
                p.lost_frame = -1
                self.current_persons.append(p)
                return True

        return False'''
    

        
def distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return ((y2 - y1)**2 + (x2 - x1) **2)**0.5

def center(box):
    (x,y,w,h) = box
    return (x + w / 2, y + h / 2)

#Added this -Ryan

# Flag for checking internet connection periodically
last_checked_time = None
check_interval = 60  # in seconds, adjust as needed

face_count = 0
batch_faces = []

Person_Handler = Person_List(age_net, gender_net) #List of person objects -ryan
frame_count = 0 #Added this -ryan
total_faces = 0

face_blob = None

while True:
    t_stamp_0 = datetime.datetime.now()
    Person_Handler.update_frame(frame_count)#Adds the current frame number to the object so it can be accessed from within the object
    Person_Handler.check_if_gone() #ryan
	
    frame_count +=1

    ret1, frame1 = cap1.read()

    

    if not ret1:
        print("Error: could not read frame from both cameras")
        break
  
    add_timestamp(frame1)

  
    # Perform face detection on frame1
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    faces1 = face_cascade.detectMultiScale(gray1, 1.2, 10) # changed min Neighbors from 5 to 10


    #ryan
    t_stamp_1 = datetime.datetime.now()
    for i, person in reversed(list(enumerate(Person_Handler.current_persons))): # Updates the trackers


        success, new_bbox = person.tracker.update(frame1)

        if success:
            x, y, w, h = map(int, new_bbox)
            #The line below just draws the box around the face
            #cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2) #Commented out the rectangle
        else:
            lost_person = Person_Handler.current_persons.pop(i)
            Person_Handler.addLostPerson(lost_person, frame_count)

    t_stamp_2 = datetime.datetime.now()
    for (x, y, w, h) in faces1:
        face_blob = cv2.dnn.blobFromImage(frame1[y:y+h, x:x+w], 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False, crop=False)

        #The current person
        cur_person = None

  	    # Add detected face data to the current batch
        #Comment the line below out
        #batch_faces.append([get_current_datetime(), gender, age])
        face_count += 1

        # Write to a new CSV file if the batch size reaches 10
        if len(Person_Handler.batch) == 3: #lowered this to 4 it was 10
            csv_filename = get_csv_filename()
            write_faces_to_csv(Person_Handler.batch, csv_filename)
            face_count = 0
            Person_Handler.batch = []

        #-ryan
        face_box = (x, y, w, h)

        # I want to check if the face had already been tracked
        face_found = False

        # Here I'm going through the faces I have presently and trying to find a face that matches
        for count, person in enumerate(Person_Handler.current_persons):
            if face_found:#If I find the face, I don't want to waste processing power by ruinning the code below unnecessarily
                continue
            if distance(center(face_box), center(person.box)) < 75: #previously 100
                face_found = True
                person.box = (x, y, w, h)
                cur_person = person


        # Check if this face is already being tracked
        if not face_found:
            if Person_Handler.in_range(face_box):
                cur_person = Person_Handler.last_person()
                 #Added the continue so that I won't create a new person and add em to Person Handler (if face is found in lost faces)
                    
            else:
                total_faces += 1  # This isn't critical,but I may need the in_range

                new_person = Person(face_box, frame1)  # creating the person object and initializing trackers
                cur_person = new_person
                Person_Handler.addPerson(new_person)
        
        #Here's where we'll make the age and gender predictions #Commented out age and gender predictions to see how it affects performance
        Person_Handler.make_prediction(face_blob, cur_person, "gender")
        Person_Handler.make_prediction(face_blob, cur_person, "age")

        # Display the results on the frame -maybe move this down
        text = f"{cur_person.get_gender()}, {cur_person.get_age()}"
        cv2.putText(frame1, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 0, 0), 2)


    #combined_frame = cv2.hconcat([frame1, frame2])
    cv2.imshow("Camera Feed", frame1)
  
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(total_faces)
        	
        break
    
    t_stamp_3 = datetime.datetime.now()




# When you want to stop the recording, set the stop event
stop_event.set()

# Wait for the recording thread to finish
recording_thread.join()
firebase_observer.stop()

cap1.release()
cv2.destroyAllWindows()

