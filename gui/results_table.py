from tkinter import *

# https://www.geeksforgeeks.org/create-table-using-tkinter/
 
class Results_Table:
     
    def __init__(self, root, json):
         
        results = self.json_to_list(json)

        total_rows, total_columns = len(results), len(results[0]) - 2
        # code for creating table
        # Column "Headers"
        for i in range(total_columns):
            self.e = Entry(root, width=15, fg='black',
                               font=('Arial',16,'bold'))
            self.e.grid(row=0, column=i)
            self.e.insert(END, results[0][i][0])

        # Results
        for i in range(total_rows):
            for j in range(total_columns):
                 
                self.e = Entry(root, width=15, fg='blue',
                               font=('Arial',16,'bold'))
                 
                self.e.grid(row=i+1, column=j)
                self.e.insert(END, results[i][j][1])

    def json_to_list(self, json):
        list = []
        for element in json:
            list.append(tuple(element.items()))
        return list