import warnings
warnings.filterwarnings("ignore")
# from app import start
import config
import torch
import time
from model import BERTBaseUncased
import torch.nn as nn
from tkinter import Tk, Label, Text, Toplevel, Canvas , WORD , Button 
from PIL import Image, ImageDraw, ImageTk

MODEL = None
DEVICE = "cpu"
PREDICTION_DICT = dict()

def sentence_prediction(sentence):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LEN
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None,
        add_special_tokens=True,
        max_length=max_len
    )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_length = max_len - len(ids)
    ids = ids + ([0] * padding_length)
    mask = mask + ([0] * padding_length)
    token_type_ids = token_type_ids + ([0] * padding_length)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = MODEL(
        ids=ids,
        mask=mask,
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

MODEL = BERTBaseUncased()
MODEL = nn.DataParallel(MODEL)
MODEL.load_state_dict(torch.load(config.MODEL_PATH))
MODEL.to(DEVICE)
MODEL.eval()

root = Tk()
root.title('bert sentiment analysis')

image = Image.open('image.png')
# print(image.size)
width, height = (1300, 1000)

# I disallow window to resizing and I make the size of the window the same than the size of the background image
root.resizable(width=True, height=True)
root.geometry("%sx%s"%(width, height))

text_x = width/8 -10
text_y = height/5.0

draw = ImageDraw.Draw(image)

root.resizable(width=False, height=False)

photoimage2 = ImageTk.PhotoImage(image)

canvas = Canvas(root,width=width, height=height)
canvas.create_image((0,0), image=photoimage2, anchor="nw")
canvas.pack()

textArea = Text(root, height=6, width=70, wrap=WORD)
canvas.create_window((text_x, text_y ), window=textArea, anchor="nw")

id= []

# Output text box
output_text = Text(root, height=2, width=30, wrap=WORD, font=('Arial', 16), fg='green')
canvas.create_window((text_x + 80, text_y + 350), window=output_text, anchor="nw")

def getText():
    global id 
    sentence = textArea.get('1.0','end')
    positive = sentence_prediction(sentence)
    negative = 1 - positive
    if(positive > negative):
        text1 = 'Positive.'
        draw.text((text_x, text_y), text1, fill="green")
    else:
        text1 = 'Negative.'
        draw.text((text_x, text_y), text1, fill="red")
    # text1 = 'Your text is ' + str(format(positive,'.3f')) + ' positive and ' + str(format(negative,'.3f')) + ' negative'
    
    # x = canvas.create_text((text_x+80, text_y+250), text=text1, fill="black", anchor="nw" , font=('serif',15))
    # id.append(x)
    
    # Display result in output text box
    output_text.delete('1.0', 'end')
    output_text.insert('1.0', text1)

button1 = Button(text='Predict Text', bg='green' , fg= 'white', width = 12 , activeforeground='red' ,relief= 'groove' ,command=getText)
canvas.create_window(text_x + 200 , text_y + 107 , window=button1 , anchor="nw")

def clearText():
    print(textArea.delete('1.0','end'))
    for i in id:
        canvas.delete(i)

button2 = Button(text='Clear Input', bg='red', fg='white', width = 12  ,relief= 'groove' , command=clearText)
canvas.create_window(text_y + 280 , text_y + 107, window=button2 , anchor="nw")


root.mainloop()
