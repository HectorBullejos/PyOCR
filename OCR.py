import cv2
import pandas as pd
import pytesseract
from pytesseract import Output
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


img = cv2.imread('place_your_.jpg_file')

# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# print(d.keys(),d.values())

dd = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
dd = dd[dd['text'].notna()]
print(dd.columns, dd.iloc[:, -1])

dd.iloc[:, -1] # primera columna
dd.iloc[:, -1].to_csv("text.csv")
#
# df['Unnamed: 0']  =  pd.to_datetime(df['Unnamed: 0'])
# df = df.rename(columns={'Unnamed: 0': "date"}, errors="raise")

h, w, _ = img.shape # assumes color image

# run tesseract, returning the bounding boxes
boxes = pytesseract.image_to_boxes(img)
plt.figure(dpi=1200)
for b in boxes.splitlines():
    b = b.split()
    cv2.rectangle(img, ((int(b[1]), h - int(b[2]))), ((int(b[3]), h - int(b[4]))), (0, 255, 0), 2)

plt.imshow(img)
plt.savefig("imag.jpg")
plt.show()
