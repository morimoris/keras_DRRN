import cv2
import random
import glob

class datacreate:
    def __init__(self):
        self.num = 0
        self.mag = 3

#Function to generate an arbitrary number of datasets
    def datacreate(self,
                img_path,     #Path where training data is stored
                data_number,  #Number of train datasets
                cut_frame,    #Number of data to be generated from a single image
                HR_height,    #Save HR size
                HR_width):

        LR_height = HR_height  #LR size = HR size
        LR_width = HR_width 

        low_data_list = []
        high_data_list = []

        path = img_path + "/*"
        files = glob.glob(path)

        while self.num < data_number:
            photo_num = random.randint(0, len(files) - 1)
            img = cv2.imread(files[photo_num])
            height, width = img.shape[:2]

            if HR_height > height or HR_width > width:
                break
                
            color_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            gray = color_img[:, :, 0]      
            bicubic_img = cv2.resize(gray , (int(width // self.mag), int(height // self.mag)), interpolation=cv2.INTER_CUBIC)
            bicubic_img = cv2.resize(bicubic_img , (int(width), int(height)), interpolation=cv2.INTER_CUBIC)

            for i in range(cut_frame):
                ram_h = random.randint(0, height - LR_height)
                ram_w = random.randint(0, width - LR_width)

                LR_img = bicubic_img[ram_h : ram_h + LR_height, ram_w: ram_w + LR_width]
                high_img = gray[ram_h : ram_h + HR_height, ram_w: ram_w + HR_width]

                low_data_list.append(LR_img)
                high_data_list.append(high_img)

                self.num += 1

                if self.num == data_number:
                    break

        return low_data_list, high_data_list
