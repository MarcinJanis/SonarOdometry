import csv
import numpy as np
import os

class DataLogger:
    def __init__(self, root_pth, filename, header, buff_size = 1):

        self.file_pth = os.path.join(root_pth, filename)

        # open file, save handlers
        self.file = open(self.file_pth, mode='a', newline='', encoding='utf-8')
        self.writer = csv.writer(self.file)

        # create headers 
        file_exists = os.path.isfile(self.file_pth)
        is_empty = not file_exists or os.path.getsize(self.file_pth) == 0
        if is_empty:
            self.writer.writerow(header)
            self.file.flush()
   
        

        # buffers
        self.buff = []  
        self.buff_size = buff_size

    def __del__(self):
        self.close()    

    def log(self, data):
        self.buff.append(data)
        if len(self.buff) >= self.buff_size:
            self.flush()

    def flush(self):
        if self.buff:
            self.writer.writerows(self.buff)
            self.file.flush() 
            self.buff = []    
    
    def close(self):
        self.flush()
        self.file.close()



