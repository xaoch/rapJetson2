
import random
import csv
import io
import os
import threading
import time
import logging
import sys
from sys import platform
from datetime import datetime
try: #Depende de la version de python
   from Queue import Queue, Empty
except:
   from queue import Queue, Empty
import numpy as np
import cv2
from configs import CONFIGS, CAMERA, SERVER_URL
from datetime import datetime
import subprocess
import json
import math
import requests
from requests import ConnectionError
from glob import glob
import shutil

WEBSERVER=CONFIGS["rap_server"]

SAVE_CSV_URL = "http://"+WEBSERVER+"/resultados/save_csv/"
SAVE_IMAGES_URL = "http://"+WEBSERVER+"/resultados/save_images/"
FFMPEG_BIN="/usr/bin/ffmpeg"

#Antropometric constant values of the human head.
#Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
#X-Y-Z with X pointing forward and Y on the left.
#The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = np.float32([-100.0, -77.5, -5.0]) #0
P3D_GONION_RIGHT = np.float32([-110.0, -77.5, -85.0]) #4
P3D_MENTON = np.float32([0.0, 0.0, -122.7]) #8
P3D_GONION_LEFT = np.float32([-110.0, 77.5, -85.0]) #12
P3D_LEFT_SIDE = np.float32([-100.0, 77.5, -5.0]) #16
P3D_FRONTAL_BREADTH_RIGHT = np.float32([-20.0, -56.1, 10.0]) #17
P3D_FRONTAL_BREADTH_LEFT = np.float32([-20.0, 56.1, 10.0]) #26
P3D_SELLION = np.float32([0.0, 0.0, 0.0]) #27
P3D_NOSE = np.float32([21.1, 0.0, -48.0]) #30
P3D_SUB_NOSE = np.float32([5.0, 0.0, -52.0]) #33
P3D_RIGHT_EYE = np.float32([-20.0, -65.5,-5.0]) #36
P3D_RIGHT_TEAR = np.float32([-10.0, -40.5,-5.0]) #39
P3D_LEFT_TEAR = np.float32([-10.0, 40.5,-5.0]) #42
P3D_LEFT_EYE = np.float32([-20.0, 65.5,-5.0]) #45
#P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
#P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = np.float32([10.0, 0.0, -75.0]) #62
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)
HAND_MID_SPINE_THRESHOLD=100
HAND_DISTANCE_THRESHOLD=80

cam_w = 600
cam_h = 480
c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / np.tan(60/2 * np.pi / 180)
f_y = f_x

camera_matrix = np.float32([[f_x, 0.0, c_x],
                               [0.0, f_y, c_y],
                               [0.0, 0.0, 1.0] ])

camera_distortion = np.float32([0.0, 0.0, 0.0, 0.0, 0.0])

landmarks_3D = np.float32([P3D_RIGHT_SIDE,
                                  P3D_GONION_RIGHT,
                                  P3D_MENTON,
                                  P3D_GONION_LEFT,
                                  P3D_LEFT_SIDE,
                                  P3D_FRONTAL_BREADTH_RIGHT,
                                  P3D_FRONTAL_BREADTH_LEFT,
                                  P3D_SELLION,
                                  P3D_NOSE,
                                  P3D_SUB_NOSE,
                                  P3D_RIGHT_EYE,
                                  P3D_RIGHT_TEAR,
                                  P3D_LEFT_TEAR,
                                  P3D_LEFT_EYE,
                                  P3D_STOMION])


FPS = CAMERA['framerate']

logger = logging.getLogger("Camera")
logger.setLevel(logging.DEBUG)


class VideoRecorder:
    """
    VideoRecorder que utiliza opencv para leer datos de camara usb.
    """

    def __init__(self, on_error):
        """
        on_error: callback
        """
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            raise IOError("Error al reconocer la camara USB")
        self.set_camera_params()
        # print self.camera.get(cv2.CAP_PROP_FRAME_WIDTH), self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        #self.channel = grpc.insecure_channel(SERVER_URL)
        self.record_channel = None
        # self.grpc_stub = FeatureExtractionApi_pb2_grpc.FeatureExtractionStub(channel)
        self.recording_stop = True
        self.image_queue = Queue()
        self.count = 0
        self.sent_count = 0
        self.grabbing = False
        self.on_error = on_error
        # Starting Camera
        logger.debug("Camera started")

    def set_camera_params(self):
        self.camera.set(3,600)
        self.camera.set(4,480)

    def capture_continuous(self, filename):
        """
        Captura frames en un loop y los encola en image_queue
        """
        logger.debug("capture continuous")
        self.count = 1
        self.grabbing = True
        self.filename = filename
        self.createFolders(filename)
 	self.csv_file=open(str(filename)+'/result.csv', mode='w')
	self.resultfile = csv.writer(self.csv_file, delimiter=',')
        self.resultfile.writerow(["frame","looks","positions"])
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        self.videoFile = cv2.VideoWriter(filename+'/video.avi',fourcc, 5.0, (int(self.camera.get(3)),int(self.camera.get(4))))

        self.imgDictionary={}
        self.lastBodyPosture="none"
        self.lastHeadPosture="none"

        while True:
            start = time.time()
            ret, frame = self.camera.read()
            #frame=cv2.flip(frame,0)
            #bytesImg= cv2.imencode(".jpg",frame)[1].tostring()
            self.image_queue.put(frame)

            if self.recording_stop:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            self.count += 1
            time.sleep(max(1./FPS - (time.time() - start), 0))
        self.grabbing = False


    def generate_videos_iterator(self):
        """
        Iterator. lee frames de cola image_queue
        """
        logger.debug("generate video iterator")
        self.sent_count = 1
        while not self.recording_stop or not self.image_queue.empty() or self.grabbing:
            try:
                frame= self.image_queue.get(block=True, timeout=1)
                self.videoFile.write(frame)
                cv2.imwrite(self.filename+"/tempFrames/frame"+str(self.sent_count)+".png",frame)
                self.image_queue.task_done()
                print ("sent",self.sent_count, "of", self.count, "captured")
                self.sent_count += 1
            except Empty as ex:
                logger.error("No data in image queue")

        logger.debug("Done generating images")
        self.videoFile.release()


    def select_biggest_skeleton(self,keypoints):
        max_id = 0;
        max_size = 0;
        for i in range(0,keypoints.shape[0]):
            rhip_y = keypoints[i, 8, 1]
            lhip_y = keypoints[i, 11, 1]
            neck_y = keypoints[i, 1, 1]
            size = 0
            if (neck_y != 0 and (rhip_y != 0 or lhip_y != 0)):
                 size = (rhip_y + lhip_y) / 2 - neck_y
                 if (size > max_size):
                      max_size = size
                      max_id = i
        return max_id


    def headPosture(self,fk,bodyId):
        landmarks_2D = np.zeros((len(TRACKED_POINTS),2), dtype=np.float32)
        counter = 0
        for point in TRACKED_POINTS:
            landmarks_2D[counter] = [fk[bodyId][point][0], fk[bodyId][point][1]]
            counter += 1
        retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                          landmarks_2D,
                                          camera_matrix, camera_distortion)
        rmat = cv2.Rodrigues(rvec)[0]
        ypr = -180*self.yawpitchrolldecomposition(rmat)/math.pi
        ypr[1,0] = ypr[1,0]+90
        if ypr[0,0]>75 and ypr[0,0]<105:
            if ypr[1,0]>-10 and ypr[1,0]<10:
                return "center"
            else:
                if ypr[1,0]>=10:
                    return "up"
                else:
                    return "down"
        else:
           if ypr[0,0]>=105:
              return "right"
           else:
              return "left"

    def headPostureSkeleton(self,keypoints,bodyId):
        rshoulder_x = keypoints[bodyId][2][0]
        lshoulder_x = keypoints[bodyId][5][0]
        mhip_y = keypoints[bodyId][8][1]
        neck_x = keypoints[bodyId][1][0]
        neck_y = keypoints[bodyId][1][1]
        nose_x = keypoints[bodyId][0][0]
        nose_y = keypoints[bodyId][0][1]
        reye_x = keypoints[bodyId][15][0]
        reye_y = keypoints[bodyId][15][1]
        leye_x = keypoints[bodyId][16][0]
        leye_y = keypoints[bodyId][16][1]
        rear_x = keypoints[bodyId][17][0]
        rear_y = keypoints[bodyId][17][1]
        lear_x = keypoints[bodyId][18][0]
        lear_y = keypoints[bodyId][18][1]
        rdist=reye_x-rear_x
        ldist=lear_x-leye_x
        difference=ldist-rdist
        normalizer= (mhip_y-neck_y)/13


        average_ear=(rear_y+lear_y)/2
        average_eye=(reye_y+leye_y)/2
        distance_eyes=(leye_x-reye_x)
        distance_leye_nose=(leye_x-nose_x)
        distance_reye_nose=(nose_x-reye_x)
        atitude=average_ear-nose_y
        print("Entrando a los ifs")
        print(rdist,ldist,difference,normalizer,average_ear,nose_y,atitude,average_eye,distance_eyes,distance_leye_nose,distance_reye_nose)
        if rshoulder_x != 0 and lshoulder_x != 0 and lshoulder_x < rshoulder_x: #Persona de espaldas
            return "tv"
        if rear_x==0 and abs(difference)>normalizer:
            return "left"
        if lear_x==0 and abs(difference)>normalizer:
            return  "right"
        if difference>normalizer:
            return "right"
        else:
            if difference<-(normalizer):
                return "left"
        if atitude>((normalizer/3)):
            return "up"
        else:
            if atitude<-(normalizer/1.2):
                return "down"
        return "center"

    def bodyPosture(self,keypoints, person_index, face_orientation, head_height):
        rwrist_y = keypoints[person_index][4][1]
        rwrist_x = keypoints[person_index][4][0]
        lwrist_y = keypoints[person_index][7][1]
        lwrist_x = keypoints[person_index][7][0]
        mhip_y = keypoints[person_index][8][1]
        lhip_y = keypoints[person_index][11][1]
        neck_y = keypoints[person_index][1][1]
        nose_y = keypoints[person_index][0][1]
        rshoulder_x = keypoints[person_index][2][0]
        lshoulder_x = keypoints[person_index][5][0]

        if rshoulder_x != 0 and lshoulder_x != 0 and lshoulder_x < rshoulder_x: #Persona de espaldas
            return "bad"
        if mhip_y == 0:
            return "NOT_DETECTED"
        if lwrist_y == 0:
            lwrist_y = rwrist_y
        if rwrist_y == 0:
            rwrist_y = lwrist_y
        if rwrist_y == 0:
            return "bad"

        hand_distance_threshold = neck_y - nose_y
        spinebase = mhip_y
        spinemid = ((3*spinebase) + neck_y)/4
        #spinemid = spinebase-3*hand_distance_threshold
        normalizer = 0
        if head_height > 0:
           normalizer= head_height
        else:
           normalizer=HAND_MID_SPINE_THRESHOLD
        #if lwrist_y < (spinemid - (HAND_MID_SPINE_THRESHOLD/head_height)) or rwrist_y < (spinemid - (HAND_MID_SPINE_THRESHOLD/head_height)):
        if lwrist_y < spinemid or rwrist_y < spinemid:
            if rwrist_x != 0 and lwrist_x != 0 and abs(rwrist_x - lwrist_x) < hand_distance_threshold:
                return "bad"
            return "good"
        return "bad"

    def writeToRapCsv(self, frame, posture, face):
        if posture != "good":
            postureValue = 0
        else:
            postureValue = 1
        self.resultfile.writerow([frame, face, postureValue])

    def captureFacePoseImages(self, img, actualOrientation, lastOrientation, mode, x, y, width, height):
        #Mode 0 Face, Mode 1 Pose
        if (mode != 0 and mode != 1):
            return
        if actualOrientation=="NOT_DETECTED" :
            return
        if actualOrientation!=lastOrientation:
            return
        if actualOrientation in self.imgDictionary.keys():
            countImage=self.imgDictionary[actualOrientation]
        else:
            countImage=0
        countImage=countImage+1

        self.imgDictionary[actualOrientation]=countImage
        if (mode == 0 and height>1 and width>1):
             # Condiciones que debe cumplir el ROI
             # box within the image plane
             img=img[int(y):int(y)+int(height),int(x):int(x)+int(width)]
             imgheight, imgwidth, channels = img.shape
             if imgheight>0 and imgwidth>0:
                 img=cv2.resize(img, (200, 200))
        cv2.imwrite(self.filename+"/"+ actualOrientation+"/img" + str(countImage)+".jpg", img)

    def processVideo(videos_iterator):
        while True:
           for frame in videos_iterator:
               pass

    def start_recording(self, filename):
        """
        Empieza grabacion. Crea hilo para captura y canal grpc para envio
        """
        logger.info("Start recording")
        try:
            #self.record_channel = grpc.insecure_channel(SERVER_URL)
            #if not self.ping():
            #    raise
            #self.grpc_stub = FeatureExtractionApi_pb2_grpc.FeatureExtractionStub(self.record_channel)
            threading.Thread(target=self.capture_continuous, args=(filename, )).start()
            videos_iterator = self.generate_videos_iterator()
            worker = threading.Thread(target=self.processVideo, args=(videos_iterator))
            worker.setDaemon(True)
            worker.start()

            #logger.debug(response)

            #self.record_channel.close()
            self.record_channel = None
        except Exception as e:
            logger.exception("start_recording")
            logger.error("Murio grpc")
            self.on_error()


    def record(self):
        """
        Crea hilo para que inicie grabacion
        """
        filename=CONFIGS["session"]
        self.recording_stop = False
        self.image_queue = Queue()
        threading.Thread(target=self.start_recording, args=(filename, )).start()

    def stop_record(self, callback=None):
        """
        Detiene grabacion de video
        callback: se ejecuta una vez ha finalizado el envio de todos los frames a servidor grpc
        """

        self.recording_stop = True
        time.sleep(5)
        self.image_queue.join()
        self.channel = None

        CONFIGS["session"] = '0'
        if callback:
            callback()
        #subprocess.call(["/home/nvidia/openpose/build/examples/openpose/openpose.bin", "--write_json", self.filename+"/output/", "--display", "0", "--render_pose", "0","--face","--image_dir", self.filename+"/tempFrames/", "-net_resolution","128x96" ])
        subprocess.call(["/home/nvidia/openpose/build/examples/openpose/openpose.bin", "--write_json", self.filename+"/output/", "--display", "0", "--image_dir", self.filename+"/tempFrames/", "--write_video", self.filename+"/result.avi","--write_video_fps", "5" ])
        #subprocess.call(["/home/nvidia/openpose/build/examples/openpose/openpose.bin", "--write_json", self.filename+"/output/", "--display", "0", "--render_pose", "0","--image_dir", self.filename+"/tempFrames/" ])

        self.featureExtraction()
        self.sendData()

    def getHeadRectangle(self,keypoints,bodyId):
        nose_x = keypoints[bodyId][0][0]
        nose_y = keypoints[bodyId][0][1]
        mhip_y = keypoints[bodyId][8][1]
        normalizer=(mhip_y-nose_y)/4
        if nose_y==0:
            return 0,0,0,0
        else:
            x=nose_x-normalizer
            y=nose_y-normalizer
            width=normalizer*2
            height=normalizer*2
            return x,y,width,height

    def featureExtraction(self):
        for i in range(1,self.count):
            f=open(self.filename+"/output/frame"+str(i)+"_keypoints.json")
            frame=cv2.imread(self.filename+"/tempFrames/frame"+str(i)+".png")
            data = json.load(f)
            f.close()
            people=data["people"]
            posture=[]
            #face=[]
            for person in people:
                posture.append(person["pose_keypoints_2d"])
                #face.append(person["face_keypoints_2d"])

            keypoints=self.convertToArray(posture)
            #fk=self.convertToArray(face)
            if not(posture is None) and keypoints.shape[0]>0:
                bodyId=self.select_biggest_skeleton(keypoints)
                #headx,heady,headw,headh=cv2.boundingRect(np.array(fk[bodyId], dtype=np.int32))
                headx,heady,headw,headh=self.getHeadRectangle(keypoints,bodyId)
                #(x,y),radius = cv2.minEnclosingCircle(np.array(fk[bodyId], dtype=np.int32))
                #print(bodyId,radius)
                head_height=headh
                #if (len(rectangles) > 0 and rectangles[bodyId].y>0):
                if(True):
                    #hp=self.headPosture(fk,bodyId)
                    hp=self.headPostureSkeleton(keypoints,bodyId)
                    bp=self.bodyPosture(keypoints,bodyId,hp,head_height)
                    self.writeToRapCsv(i, bp, hp)
                    self.captureFacePoseImages(frame, hp, self.lastHeadPosture, 0, headx, heady, headw, headh)
                    self.captureFacePoseImages(frame, bp, self.lastBodyPosture, 1, headx, heady, headw, headh)
                    self.lastHeadPosture=hp
                    self.lastBodyPosture=bp
                    print(i,hp,bp)


    def get_progress(self):
        try:
            value=int(self.sent_count * 100.0 / self.count)
            if value>100:
                value=100
            return "{} %".format(value)
        except:
            return "0 %"
        # return "{}/{}".format(self.sent_count, self.count)

    def clean(self):
        self.camera.release()
        logger.debug("Camera released")
        # self.camera.close()

    def convert_to_mp4(self):
        filename_mp4 = self.filename.split(".")[0]+".mp4"
        logger.info("file .h264 saved.. Transforming to mp4...")
        os.system("MP4Box -fps 30 -add "+ self.filename + " " + filename_mp4)
        logger.info("File converted to mp4")

    def createFolders(self,path):
         try:
             shutil.rmtree(path)
         except:
             print("Cannot delete")

         try:
              os.mkdir(path)
              os.mkdir(path + "/center")
              os.mkdir(path + "/up")
              os.mkdir(path + "/down")
              os.mkdir(path + "/right")
              os.mkdir(path + "/left")
              os.mkdir(path + "/tv")
              os.mkdir(path + "/good")
              os.mkdir(path + "/bad")
              os.mkdir(path + "/tempFrames")
         except:
              print("Directories already created")



    def sendData(self):
        self.csv_file.close()
        self.selectRandomImages(3)
        self.send_results()
        self.send_images()

    def selectRandomImages(self, maxImages):
        path = self.filename
        for pose in self.imgDictionary.keys():
            value = self.imgDictionary[pose]
            if value>maxImages:
                randomValues=random.sample(range(1, value+1), maxImages)
            else:
                randomValues=range(1,value+1)
            for number in randomValues:
                img = cv2.imread(path+"/"+pose+"/img"+ str(number) + ".jpg")
                cv2.imwrite(path+ "/img" +str(number) +"_"+pose+ ".jpg", img)


    def send_results(self):
        try:
            id = int(self.filename)
        except:
            print("student ID error")
            return
        csv_path = os.path.join(self.filename+"/result.csv")
        csv_file = open(csv_path,"rb");
        #print csv_path

        values = {"resultado_id":id}
        files = {"csvfile":csv_file}

        try:
            response = requests.post(SAVE_CSV_URL, data=values, files=files)
            #print "Send results:",response.status_code
            return response.status_code
        except ConnectionError as e:
            print(e)
            #print "---------Error al conectarse con el servidor--------------- "
            #print "Sent results:",400
            return 400

    def send_images(self):
        try:
            id = int(self.filename)
        except:
            print("Student Id error")
            return
        imagesPath = self.filename
        values = {"resultado_id":id}
        files = {}

        image_type = "img_type_"
        classifier = "classifier_"
        filename = "filename_"
        fileString = "img_"
        count = 0

        for actualfile in glob(imagesPath+"/*.jpg"):

            files["{}{:d}".format(fileString,count)] = open(actualfile, "rb")

            actualfile = actualfile.split('/')
            actualFileName = actualfile[-1]

            actual_classifier = actualFileName.split('_')[-1].split('.')[0]
            if(actual_classifier=="good" or actual_classifier=="bad" ):
                values["{}{:d}".format(image_type,count)] = "p"
            else:
                values["{}{:d}".format(image_type,count)] = "m"

            values["{}{:d}".format(classifier,count)] = actual_classifier
            values["{}{:d}".format(filename,count)] = actualFileName
            print(actualFileName)

            count += 1

        values["num_images"] = count
        try:
            response = requests.post(SAVE_IMAGES_URL, data=values, files=files)
            print("Sent images:",response.status_code)
        except ConnectionError as e:
            #print "---------Error al conectarse con el servidor--------------- "
            #print "Sent images:",400
            print(e)
            return 400

        for key,value in  files.items(): value.close()
        #for actualfile in glob(imagesPath+"/*.jpg"):os.remove(actualfile)
        #print os.path.join(imagesPath,"video.avi")
        # Transforma y envia el video en mp4
        command = [FFMPEG_BIN,'-i',os.path.join(imagesPath,"result.avi"),os.path.join(imagesPath,"video.mp4")]
        print(command)
        FNULL = open(os.devnull, 'w')
        print ("Converting to mp4..",)
        join_process = subprocess.Popen(command, stdout=FNULL, stderr=subprocess.STDOUT, bufsize=10**8)
        join_process.communicate()
        FNULL.close()
        print("Done")
        subprocess.call(["./sendVideo", os.path.join(imagesPath,"video.mp4"),"root@"+WEBSERVER+":/home/rap/RAP/rap_v2/static/resultados/"+str(id)+"/video/video_complete.mp4"])
        response2 = requests.get("http://"+WEBSERVER+"/resultados/process_video/?resultado_id="+str(id))
        print ("Process media: ", response2.status_code)
            #print video_status
        #os.remove(os.path.join(imagesPath,"video.mp4"))
        return response.status_code

    def ping(self):
        """
        Verifica que exista conexion con servidor grpc
        """
        return True
        if self.channel is None:
            self.channel = grpc.insecure_channel(SERVER_URL)
        try:
            grpc.channel_ready_future(self.channel).result(timeout=1)
            logger.info("Ping")
            return True
        except grpc.FutureTimeoutError as e:
            logger.error("Couldnt connect to GRPC SERVER")
            self.channel.close()
            self.channel = None
            return False



    def yawpitchrolldecomposition(self,R):
         sin_x = math.sqrt(R[2,0] * R[2,0] +  R[2,1] * R[2,1])
         validity  = sin_x < 1e-6
         if not validity:
             z1 = math.atan2(R[2,0], R[2,1])     # around z1-axis
             x = math.atan2(sin_x,  R[2,2])     # around x-axis
             z2 = math.atan2(R[0,2], -R[1,2])    # around z2-axis
         else: # gimbal lock
             z1 = 0                                         # around z1-axis
             x = math.atan2(sin_x,  R[2,2])     # around x-axis
             z2 = 0                                         # around z2-axis
         return np.array([[z1], [x], [z2]])

    def convertToArray(self,klist):
        personList=[]
        for person in klist:
            pointList=[]
            for i in xrange(0,len(person)-2,3):
                pointList.append([person[i],person[i+1]])
            personList.append(pointList)
        return np.asarray(personList)

if __name__ == "__main__":
    vid_recorder = VideoRecorder()
    print ("Set vid recorder")
    # vid_recorder.camera.wait_recording(5)
    time.sleep(2)
    start = datetime.now()
    print("Start" , start)
    print(start)
    vid_recorder.record()
    # vid_recorder.camera.wait_recording(2)
    # vid_recorder.camera.capture("foo.jpg", use_video_port=True)
    # print ("Pic taken")
    # vid_recorder.camera.wait_recording(30)
    time.sleep(30)
    vid_recorder.stop_record()

    vid_recorder.clean()
