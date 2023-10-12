import pickle
import cv2
from keras.preprocessing import image
from keras.utils import img_to_array
import numpy
from keras.models import load_model
from tkinter import *
import tkinter
from tkinter import filedialog
from PIL import ImageTk, Image
import pandas as pd
disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
root  = Tk()
root.title("Plant Disease Detect")
root.geometry("500x500")
root.minsize(300,300)
root.maxsize(500,500)
root.iconbitmap("C:\\Users\\Arpeet\\Desktop\\Arpeet\\aabb\\icon.ico")
root.configure(bg="lightgreen")
image = Image.open("C:\\Users\\Arpeet\\Desktop\\icon.jpg")
photo = ImageTk.PhotoImage(image)
label = Label(image=photo,text="Hello world")
label.pack()
head_label = tkinter.Label(text="*Snap picture of Plant Leaf\n*Upload the Image",bg="#166f47",font=('Arial', 13))
head_label.place(relx=0.5, rely=0.1, anchor='center',height=70,width=200)

# print(description[0])

filename2 = 'plant_disease_label_transform.pkl'
image_labels = pickle.load(open(filename2, 'rb'))
# print(image_labels.classes_[:-1])

model = load_model('model.h5')

def predictwindow(result,confidence):
    Window2 = Toplevel(root)
    Window2.configure(bg="#166f47")
    Window2.title("Prediction")
    Window2.geometry("500x500")
    Window2.minsize(300,300)
    Window2.maxsize(500,500)
    Window2.iconbitmap("C:\\Users\\Arpeet\\Desktop\\Arpeet\\aabb\\icon.ico")
    prediction = tkinter.Label(Window2,text="",bg="#166f47",font=('Arial', 12))
    prediction.place(relx=0.0, rely=0.0)
    prediction.config(text="")
    prediction.config(text="Disease Name: "+image_labels.classes_[result]+"\nConfidence : "+str(100*confidence)+"%",padx=20)
    info = Message(Window2,text="",bg="#166f47",font=('Arial', 12))
    info.place(relx=0, rely=0.1,width=400)
    info.config(text="Disease Info: \n"+disease_info["description"][result])
    precaution = Message(Window2,text="",bg="#166f47",font=('Arial', 12))
    precaution.place(relx=0.01, rely=0.6)
    precaution.config(text="Disease Precaution(s): \n"+disease_info["Possible Steps"][result])


def showimg(path):
    newWindow = Toplevel(root)
    newWindow.configure(bg="#166f47")
    newWindow.title("Uploaded Image")
    newWindow.geometry("256x256")
    newWindow.iconbitmap("C:\\Users\\Arpeet\\Desktop\\Arpeet\\aabb\\icon.ico")
    image3 = Image.open(path)
    resize = image3.resize((256,256))
    photo3 = ImageTk.PhotoImage(resize)
    label3 = Label(newWindow,image=photo3)
    label3.image = photo3
    label3.pack()


def getImage():
    path = filedialog.askopenfilename()
    return path

DEFAULT_IMAGE_SIZE = tuple((256, 256))

def convert_image_to_array(image_dir):
    try:
        image = cv2.imread(image_dir)
        if image is not None:
            image = cv2.resize(image, DEFAULT_IMAGE_SIZE)   
            return img_to_array(image)
        else:
            return numpy.array([])
    except Exception as e:
        print(f"Error : {e}")
        return None


def predict_disease():
    path = getImage()
    image_array = convert_image_to_array(path)
    np_image = numpy.array(image_array, dtype=numpy.float32) / 225.0
    np_image = numpy.expand_dims(np_image,0)
    confidence = numpy.amax(model.predict(np_image))
    result = numpy.argmax(model.predict(np_image))
    # print(disease_info['description'][result])
    showimg(path)
    predictwindow(result,confidence)
    
    

upload_button = tkinter.Button(text="Upload", command=predict_disease)
upload_button.place(relx=0.5, rely=0.75, anchor='center',height=50,width=100)

root.mainloop()