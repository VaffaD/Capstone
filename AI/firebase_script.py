import os
import threading
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import pyrebase
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import NetworkManager
import dbus.mainloop.glib
from gi.repository import GLib

class FirebaseObserver:
    def __init__(self, video_dir, csv_dir, mac_to_check):
        self.video_dir = video_dir
        self.csv_dir = csv_dir
        self.mac_to_check = mac_to_check
        self.observer = Observer()
        self.initialize_firebase()
        self.db = firestore.client()

    def start(self):
        event_handler = self.UploadHandler(self.db, self.mac_to_check, self.video_dir, self.csv_dir)
        self.observer.schedule(event_handler, path=self.video_dir, recursive=False)
        self.observer.schedule(event_handler, path=self.csv_dir, recursive=False)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def initialize_firebase(self):
        # Firebase configuration
        config = {
	    "apiKey": "AIzaSyDIwDlBu0LeriJWo2r7stDa_U_TxfSdabk",
	    "authDomain": "passengersense.firebaseapp.com",
	    "databaseURL": "https://passengersense-default-rtdb.firebaseio.com",
	    "projectId": "passengersense",
	    "storageBucket": "passengersense.appspot.com",
	    "messagingSenderId": "183736169378",
	    "appId": "1:183736169378:web:488745d943e49a851338fb",
	    "measurementId": "G-N5TTQ339NH",
	    "serviceAccount": "serviceAccountKey.json",
	    "databaseURL": 'https://passengersense-default-rtdb.firebaseio.com'
        }
        cred = credentials.Certificate('serviceAccountKey.json')
        firebase_admin.initialize_app(cred)
        global firebase
        firebase = pyrebase.initialize_app(config)

    class UploadHandler(FileSystemEventHandler):
        def __init__(self, db, mac_to_check, video_dir, csv_dir):
            self.db = db
            self.mac_to_check = mac_to_check
            self.storage = firebase.storage()
            self.video_dir = video_dir
            self.csv_dir = csv_dir

        def on_created(self, event):
            if not event.is_directory:
                file_path = event.src_path
                filename = os.path.basename(file_path)
                if file_path.startswith(self.video_dir) and filename.lower().endswith('.mp4'):
                    if self.check_mac_address(self.mac_to_check):
                        self.upload_file(file_path)
                elif file_path.startswith(self.csv_dir) and filename.lower().endswith('.csv'):
                    self.process_csv(file_path, filename)

        def upload_file(self, file_path):
            try:
                self.storage.child(os.path.basename(file_path)).put(file_path)
                print(f"Uploaded file: {file_path}")
            except Exception as e:
                print(f"Error uploading file: {str(e)}")

        def process_csv(self, file_path, filename):
            try:
                df = pd.read_csv(file_path)
                data = self.calculate_age_gender_data(df)
                self.update_firestore_data(filename, data)
            except Exception as e:
                print(f"Error processing CSV file: {str(e)}")

        def calculate_age_gender_data(self, df):
        # Age and gender data calculation
            data = {
                'male': len(df[df['Gender'] == 'Male']),
                'female': len(df[df['Gender'] == 'Female']),
                'age_0_4': len(df[df['Age'] == '(0-4)']),
                'age_5_9': len(df[df['Age'] == '(5-9)']),
                'age_10_14': len(df[df['Age'] == '(10-14)']),
                'age_15_19': len(df[df['Age'] == '(15-19)']),
                'age_20_24': len(df[df['Age'] == '(20-24)']),
                'age_25_29': len(df[df['Age'] == '(25-29)']),
                'age_30_34': len(df[df['Age'] == '(30-34)']),
                'age_35_39': len(df[df['Age'] == '(35-39)']),
                'age_40_44': len(df[df['Age'] == '(40-44)']),
                'age_45_49': len(df[df['Age'] == '(45-49)']),
                'age_50_54': len(df[df['Age'] == '(50-54)']),
                'age_55_59': len(df[df['Age'] == '(55-59)']),
                'age_60_64': len(df[df['Age'] == '(60-64)']),
                'age_65_69': len(df[df['Age'] == '(65-69)']),
                'age_70_74': len(df[df['Age'] == '(70-74)']),
                'age_75_79': len(df[df['Age'] == '(75-79)']),
                'age_80_84': len(df[df['Age'] == '(80-84)']),
                'age_85_89': len(df[df['Age'] == '(85-89)']),
                'age_90_94': len(df[df['Age'] == '(90-94)']),
                'age_95_99': len(df[df['Age'] == '(95-99)']),
                'age_100_plus': len(df[df['Age'] == '(100+)'])
            }
            return data



        def update_firestore_data(self, filename, data):
            try:
                doc_ref = self.db.collection('files').document(filename)
                doc = doc_ref.get()
                if doc.exists:
                    doc_ref.set(data, merge=True)
                    print(f"Updated Firestore document: {filename}")
                else:
                    doc_ref.set(data)
                    print(f"Created new Firestore document: {filename}")
            except Exception as e:
                print(f"Error updating Firestore: {str(e)}")

        def check_mac_address(self, mac_to_check):
            dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
            loop = GLib.MainLoop()
            for conn in NetworkManager.NetworkManager.ActiveConnections:
                settings = conn.Connection.GetSettings()
                # Check for wireless and ethernet settings as the MAC address can be in either
                if '802-11-wireless' in settings and 'mac-address' in settings['802-11-wireless']:
                    mac_address = settings['802-11-wireless']['mac-address']
                    if mac_address.lower() == mac_to_check.lower():
                        return True
                elif '802-3-ethernet' in settings and 'mac-address' in settings['802-3-ethernet']:
                    mac_address = settings['802-3-ethernet']['mac-address']
                    if mac_address.lower() == mac_to_check.lower():
                        return True
            return False





