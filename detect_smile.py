# from keras.utils import img_to_array
# from keras.models import load_model
# import numpy as np
# import argparse
# import imutils
# import cv2

# ap = argparse.ArgumentParser()

# ap.add_argument("-c", "--cascade", required=True, help="path of face cascade")
# ap.add_argument("-m", "--model", required=True, help="path of trained model")
# ap.add_argument("-v", "--video", help="path of video file")
# args = vars(ap.parse_args())

# detector = cv2.CascadeClassifier(args["cascade"])
# model = load_model(args["model"])

# if not args.get('video', False):
#     camera = cv2.VideoCapture(0)
# else :
#     camera = cv2.VideoCapture(args["video"])

# while True:
#     (grapped, frame) = camera.read()

#     if args.get("video") and not grapped:
#         break

#     frame = imutils.resize(frame, width=300)
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     frameClone = frame.copy()

#     rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
#     for (fX, fY, fW, fH) in rects:
#         roi = gray[fY:fY+fH, fX:fX+fW]
#         roi = cv2.resize(roi, (28, 28))
#         roi = roi.astype("float")/255.0
#         roi = img_to_array(roi)
#         roi = np.expand_dims(roi, axis=0)

#         (notSmiling, smiling) = model.predict(roi)[0]
#         label = "Smiling" if smiling > notSmiling else "Not Smiling"

#         cv2.putText(frameClone, label , (fX-20, fY-20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
#         cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY+ fH), (0, 255, 0), 2)
#     cv2.imshow("Face", frameClone)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# camera.release()
# cv2.destroyAllWindows()



#Nhập hàm img_to_array từ keras.utils để chuyển đổi hình ảnh thành mảng numpy
from keras.utils import img_to_array
#Nhập hàm load_model từ keras.models để tải mô hình học máy đã được huấn luyện.
from keras.models import load_model
#Nhập thư viện NumPy để làm việc với mảng số
import numpy as np
#Nhập thư viện argparse để xử lý các đối số được truyền vào từ dòng lệnh.
#Nhập thư viện imutils để thực hiện một số thao tác xử lý hình ảnh.
#Nhập thư viện OpenCV để thao tác với hình ảnh và video
import argparse
import imutils
import cv2

#Khởi tạo một đối tượng ArgumentParser để xử lý các đối số đầu vào.
ap = argparse.ArgumentParser()

#Thêm đối số bắt buộc -c hoặc --cascade để chỉ định đường dẫn đến tệp chứa khuôn mặt cascade
ap.add_argument("-c", "--cascade", required=True, help="path of face cascade")
#Thêm đối số bắt buộc -m hoặc --model để chỉ định đường dẫn đến mô hình học máy đã được huấn luyện
ap.add_argument("-m", "--model", required=True, help="path of trained model")
#Thêm đối số tùy chọn -v hoặc --video để chỉ định đường dẫn đến tệp video
ap.add_argument("-v", "--video", help="path of video file")
#Lưu các đối số đầu vào vào biến args
args = vars(ap.parse_args())

#Tạo một đối tượng CascadeClassifier bằng cách sử dụng đường dẫn đến tệp chứa khuôn mặt cascade.
detector = cv2.CascadeClassifier(args["cascade"])
#Tải mô hình học máy đã được huấn luyện bằng cách sử dụng đường dẫn đến mô hình
model = load_model(args["model"])

# Tạo nguồn video: Sử dụng webcam hoặc lấy file có sẵn
# Nếu không có đường dẫn video được cung cấp, sử dụng camera web.
if not args.get('video', False):
    camera = cv2.VideoCapture(0)
#Nếu có đường dẫn video được cung cấp, sử dụng video đó.
else :
    camera = cv2.VideoCapture(args["video"])

#Bắt đầu vòng lặp vô hạn để xử lý các khung hình video.
while True:
    # Đọc một khung hình từ camera hoặc video
    (grabbed, frame) = camera.read()

    # Nếu đang xử lý video và không thể đọc thêm khung hình nữa, thoát vòng lặp.
    if args.get('video') and not grabbed:
        break

    # Thay đổi kích thước khung hình để hiển thị rõ hơn.
    frame = imutils.resize(frame, width=700)
    #Chuyển đổi khung hình sang ảnh xám.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Tạo bản sao của khung hình để vẽ lên đó
    frameClone = frame.copy()

    # Phát hiện các khuôn mặt trong khung hình.
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    #Lặp qua từng khuôn mặt được phát hiện.
    for (fX, fY, fW, fH) in rects:
    #Trích xuất vùng quan tâm (ROI) là khuôn mặt từ ảnh xám.
        roi = gray[fY:fY + fH, fX:fX + fW]
    #Thay đổi kích thước ROI thành 28x28 pixel.
        roi = cv2.resize(roi, (28, 28))
    #Chuẩn hóa giá trị điểm ảnh trong ROI.
        roi = roi.astype('float') / 255.0
    #Chuyển đổi ROI thành mảng numpy.
        roi = img_to_array(roi)
    #Thêm chiều batch vào mảng.
        roi = np.expand_dims(roi, axis=0)

        # Sử dụng mô hình đã tải để dự đoán xem người trong ROI có đang cười hay không.
        (notSmiling, Smiling) = model.predict(roi)[0]
        #Xác định nhãn dựa trên kết quả dự đoán.
        label = 'Smiling' if Smiling > notSmiling else "Not Smiling"

        # Vẽ hình chữ nhật và nhãn lên khung hình clone dựa trên kết quả dự đoán.
        if label == 'Smiling':
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 255, 0), 2)
        else:
            cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
    #  # Sử dụng mô hình đã tải để dự đoán xem người trong ROI đang cười, không cười hay khóc.
    #     (notSmiling, Smiling, Crying) = model.predict(roi)[0]
    #     #Xác định nhãn dựa trên kết quả dự đoán.
    #     if Smiling > notSmiling and Smiling > Crying:
    #         label = 'Smiling'
    #         color = (0, 255, 0)  # Xanh lá cây
    #     elif Crying > notSmiling and Crying > Smiling:
    #         label = 'Crying'
    #         color = (255, 0, 0)  # Xanh dương
    #     else:
    #         label = 'Not Smiling'
    #         color = (0, 0, 255)  # Đỏ

    #     # Vẽ hình chữ nhật và nhãn lên khung hình clone dựa trên kết quả dự đoán.
    #     cv2.putText(frameClone, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    #     cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), color, 2)
    # Hiển thị khung hình clone với các nhãn đã vẽ.
    cv2.imshow('Face', frameClone)
    
    #Nếu người dùng nhấn phím 'q', thoát vòng lặp.
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
#Giải phóng tài nguyên camera.
camera.release()
#Đóng tất cả các cửa sổ OpenCV.
cv2.destroyAllWindows()