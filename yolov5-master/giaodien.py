
import cv2
import torch

vid = cv2.VideoCapture("/python/DATN/2.mp4")
model = torch.hub.load('/python/DATN/yolov5-master', 'custom',path='yolov5s.pt', source='local')
model.classes = [2]
classes = model.names

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def score_frame(frame):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, -1].numpy()
    return labels, cord

def class_to_label(x):
    return classes[int(x)]


def plot_boxes(results, frame):
    labels, cord = results

    #print("labels", labels)
    #print("cord", cord[:, :-1])
    #clas = 14
    #if len(labels) !=0:
     #   print("list is not empty")
    #    for label in labels:
    #        if label == clas:
    #            print("send objects")
    #        else:
    #            print("wrong objects")
    #else:
    #    print("list is empty")
    #   print("no objects")

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(row[3] * y_shape)
            bgr = (0,255,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i])+ " " + str(round(row[4],2)), (x1,y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    return frame

while True:
    ret, frame = vid.read()
    results = score_frame(frame)
    #print(results)
    frame = plot_boxes(results, frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()





