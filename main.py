import cv2
import torch
import numpy
import serial
import time
from tkinter import *



# define a video capture object
arduino = serial.Serial('COM5', 9600)
vid = cv2.VideoCapture(0)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt')
model = torch.hub.load('yolov5-master', 'custom', path='yolov5-master/best1.pt', source='local')
model.classes = [0,1,3,4,5]  # 0: human, 2: car, 14: bird
classes = model.names  # name of objects
device = 'cuda' if torch.cuda.is_available() else 'cpu'






#arduino.write('T'.encode())
def score_frame(frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    param frame: input frame in numpy/list/tuple format.
    return: Labels and Coordinates of objects detected by model in the frame.
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    param results: contains labels and coordinates predicted by model on the given frame.
    param frame: Frame which has been scored.
    return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    print("labels", labels)
    #print("cord", cord[:, :-1])
    if len(labels) != 0:
        print("list is not empty")
        #for label in labels :
        if(0 in labels) and (3 in labels ):
            if (1 in labels )  and (5 in labels):
               print("1") #gui adruino
               name3 = Label(win, text='     đạt tiêu chuẩn     ',font= (20), fg= 'green')
               name3.place(x=70, y=80)
               arduino.write('X'.encode())
            else:
               print("2")
               name2 =Label(win,text = 'không đạt tiêu chuẩn', font= (20),fg= 'red')
               name2.place(x=70,y=80)
               arduino.write('Y'.encode())
    else:
        print("3")
        name1 = Label(win, text = '      không có vật        ', font= (20))
        name1.place(x= 70,y=80)
        #arduino.write('m'.encode())

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # print("predict", round(cord[i][4], 2))
        if row[4] >= 0.4:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]) + " " + str(round(row[4], 2)), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


"""
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()"""
def click():
    arduino.write('T'.encode())
    while TRUE:

        win.update()



        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        results = score_frame(frame)
        print(results)

        frame = plot_boxes(results, frame)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choiceq
        if cv2.waitKey(1) & 0xFF == ord('q'):
            arduino.write('d'.encode())
            break

def cl1():
  arduino.write('X'.encode())

def cl2():
    arduino.write('d'.encode())

def tg():
    arduino.write('Y'.encode())




win = Tk()
win.title('Do an phan loai san phan')
win.geometry('320x200')
start = Button(win, text='start ', bg='green', command=click)
start.place(x=270, y=140)
stop = Button(win, text = 'stop', bg='red', command= cl2 )
stop.place(x=270, y= 170)
reset = Button(win, text = 'taygat2', bg='blue', command= cl1 )
reset.place(x=50, y=170)
taygat = Button(win, text = 'taygat', bg = 'white', command= tg)
taygat.place(x= 50, y= 140)

win.mainloop()