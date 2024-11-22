from tkinter import *
import os
from tkinter import filedialog
import cv2

from tkinter import messagebox


def file_sucess():
    global file_success_screen
    file_success_screen = Toplevel(training_screen)
    file_success_screen.title("File Upload Success")
    file_success_screen.geometry("150x100")

    Label(file_success_screen, text="File Upload Success").pack()
    Button(file_success_screen, text='''ok''', font=(
        'Palatino Linotype', 15), height="2", width="30").pack()


global ttype



def imgtraining():
    import_file_path = filedialog.askopenfilename()

    image = cv2.imread(import_file_path)

    print(import_file_path)
    fnm = os.path.basename(import_file_path)
    print(os.path.basename(import_file_path))

    # file_sucess()

    print("\n*********************\nImage : " + fnm + "\n*********************")
    img = cv2.imread(import_file_path)
    if img is None:
        print('no data')

    img1 = cv2.imread(import_file_path)
    print(img.shape)
    img = cv2.resize(img, ((int)(img.shape[1] / 5), (int)(img.shape[0] / 5)))
    original = img.copy()
    neworiginal = img.copy()
    img1 = cv2.resize(img1, (960, 540))
    cv2.imshow('original', img1)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img1S = cv2.resize(img1, (960, 540))

    cv2.imshow('Original image', img1S)
    grayS = cv2.resize(gray, (960, 540))
    cv2.imshow('Gray image', grayS)

    dst = cv2.fastNlMeansDenoisingColored(img1, None, 10, 10, 7, 21)
    dst = cv2.resize(dst, (960, 540))
    cv2.imshow("Noise Removal", dst)


def fulltraining():
    import model as mm


def fulltraining1():
    import CnnTest as mm


def prediction():
    import_file_path = filedialog.askopenfilename()
    image = cv2.imread(import_file_path)
    print(import_file_path)
    filename = 'Test.png'
    cv2.imwrite(filename, image)
    print("After saving image:")
    img = cv2.imread(filename)
    cv2.imshow('original', img)
    import joblib
    import numpy as np
    from tensorflow.keras.preprocessing.image import img_to_array, load_img
    from sklearn.preprocessing import LabelEncoder
    from tensorflow.keras.models import load_model

    # Load the saved model and label encoder
    model = load_model('model.h5')
    label_encoder = joblib.load('label_encoder.pkl')

    # Prepare a new input image
    def prepare_image(img_path):
        img = load_img(img_path, target_size=(64, 64))
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    # Example usage
    img_path = 'Test.png'
    input_image = prepare_image(img_path)

    # Make a prediction
    predictions = model.predict(input_image)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_class_label = label_encoder.inverse_transform(predicted_class_index)

    print(f'Predicted class label: {predicted_class_label[0]}')
    messagebox.showinfo("Result", "Prediction Result : " + str(predicted_class_label[0]))





def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 500
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title("Medical Prescription Recognition ")

    Label(text="Medical Prescription Recognition ", width="300", height="5", font=("Palatino Linotype", 16)).pack()

    Button(text="UploadImage", font=(
        'Palatino Linotype', 15), height="2", width="20", command=imgtraining, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()
    Button(text="Model", font=(
        'Palatino Linotype', 15), height="2", width="20", command=fulltraining, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()
   
    Button(text="Prediction", font=(
        'Palatino Linotype', 15), height="2", width="20", command=prediction, highlightcolor="black").pack(side=TOP)


    main_screen.mainloop()


main_account_screen()
