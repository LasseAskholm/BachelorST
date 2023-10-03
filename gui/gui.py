import tkinter as tk
from results_table import Table
import json
from inferenceAPI import get_sample_data

# window
root = tk.Tk()
root.geometry("700x800")
root.title("BachelorST - NER")

# header
header1 = tk.Label(root, text="Input", font=('Arial', 18))
header1.pack(padx=20, pady=20)

# input 
input_textbox = tk.Text(root, height=8, font=('Arial', 14))
input_textbox.pack(padx=20, pady=20)

# get text from first char to end of text
def send_text_from_input():
   print("Button Clicked: ", input_textbox.get("1.0", tk.END))

input_button = tk.Button(root, text="Send", font=('Arial', 14), command=send_text_from_input)
input_button.pack(padx=20, pady=10)

header2 = tk.Label(root, text="Results", font=('Arial', 18))
header2.pack(padx=20, pady=20)

# get data
sample_results = get_sample_data()


# results frame
resultframe = tk.Frame(root)
table = Table(resultframe, sample_results)
resultframe.pack(padx=20, pady=20)



# Define an event to close the window
def close_win(e):
   root.destroy()

# Bind the ESC key with the callback function
root.bind('<Escape>', lambda e: close_win(e))


root.mainloop()