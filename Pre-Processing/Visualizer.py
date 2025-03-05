import numpy as np
import pydicom
import matplotlib.pyplot as plt
import matplotlib
import pathlib
import json

import tkinter
import customtkinter
from PIL import Image, ImageTk
from matplotlib.backends.backend_agg import FigureCanvasAgg
import threading
matplotlib.use('agg')

DATA_PATH = pathlib.Path('../Data')
BLACKLIST_PATH = pathlib.Path('../Data') / 'BLACKLIST.json'

class Slice_navigator:
    global DATA_PATH
    global BLACKLIST_PATH
    def __init__(self):
        # get all mice ID by looking in data directory:
        path_mouse_list = DATA_PATH / 'DICOM' / 'HIGH RES' / 'HEAD-THORAX'
        self.mouse_list = [f.name for f in path_mouse_list.iterdir() if f.is_dir()]
        
        self.current_mouse_ID = 0
        self.current_loc = 'HEAD-THORAX'
        self.current_plane = 'Coronal'
        self.current_slice = 0
        
        # initialize first slice
        self.current_slice_list = None
        self.hr_metadata = None
        self.lr_metadata = None
        self.hr_3D = None
        self.lr_3D = None
        self.slice_is_blacklisted = None
        
        self.get_3D_img()
    
    def get_3D_img(self):
        # create path object for directory to look in
        path_hr = DATA_PATH / 'DICOM' / 'HIGH RES' / self.current_loc / self.mouse_list[self.current_mouse_ID]  / self.current_plane
        path_lr = DATA_PATH / 'DICOM' / 'LOW RES'  / self.current_loc / self.mouse_list[self.current_mouse_ID] / self.current_plane

        # get a list of all slices in directory
        hr_slices = [f.name for f in path_hr.iterdir() if f.is_file()]
        lr_slices = [f.name for f in path_lr.iterdir() if f.is_file()]
        
        if not (np.array(hr_slices)==np.array(lr_slices)).all():
            print('Error: LR and HR directory do not correspond')
            return False
        self.current_slice_list = hr_slices
        self.current_slice = 0

        if len(hr_slices) == 0 or len(lr_slices) == 0:
            print('Error: No slices found in directory')
            return False

        # get some metadata
        slice_hr_path = path_hr / hr_slices[0]
        slice_hr = pydicom.dcmread(slice_hr_path, force=True)
        shape_hr = slice_hr.pixel_array.shape
        shape_3D_hr = shape_hr + (len(hr_slices),)
        self.hr_metadata = { "ps": slice_hr.PixelSpacing, "st": slice_hr.SliceThickness, "shape": shape_3D_hr, "mouse": self.mouse_list[self.current_mouse_ID], "loc": self.current_loc, "plane": self.current_plane}

        slice_lr_path = path_lr / lr_slices[0]
        slice_lr = pydicom.dcmread(slice_lr_path, force=True)
        shape_lr = slice_lr.pixel_array.shape
        shape_3D_lr = shape_lr + (len(lr_slices),)
        self.lr_metadata = { "ps": slice_lr.PixelSpacing, "st": slice_lr.SliceThickness, "shape": shape_3D_lr, "mouse": self.mouse_list[self.current_mouse_ID], "loc": self.current_loc, "plane": self.current_plane}

        # create 3D image arrays
        self.hr_3D = np.zeros(shape_3D_hr)
        for i, slice_name in enumerate(hr_slices):
            slice_path = path_hr / slice_name
            self.hr_3D[:,:,i] = pydicom.dcmread(slice_path, force=True).pixel_array

        self.lr_3D = np.zeros(shape_3D_lr)
        for i, slice_name in enumerate(lr_slices):
            slice_path = path_lr / slice_name
            self.lr_3D[:,:,i] = pydicom.dcmread(slice_path, force=True).pixel_array

        return True
    
    def get_slice(self):
        return self.lr_3D[:,:,self.current_slice], self.hr_3D[:,:,self.current_slice]
    
    def next_slice(self):
        if self.current_slice == len(self.current_slice_list) - 1:
            #rollover to next plane
            if self.current_plane == 'Coronal': 
                self.current_plane = 'Sagittal'
            elif self.current_plane == 'Sagittal': self.current_plane = 'Transax'
            else:
                self.current_plane = 'Coronal'
                # rollover to next loc
                if self.current_loc == 'HEAD-THORAX': self.current_loc = 'THORAX-ABDOMEN'
                else:
                    self.current_loc = 'HEAD-THORAX'
                    # rollover to next mouse
                    if self.current_mouse_ID == len(self.mouse_list) - 1:
                        self.current_mouse_ID = 0
                    else: 
                        self.current_mouse_ID = self.current_mouse_ID + 1
                    
            # update image
            self.get_3D_img()
            self.current_slice = 0
        else:
            self.current_slice = self.current_slice + 1
            
        return
        
    def previous_slice(self):
        if self.current_slice == 0:
            # rollover to previous plane
            if self.current_plane == 'Transax': 
                self.current_plane = 'Sagittal'
            elif self.current_plane == 'Sagittal': self.current_plane = 'Coronal'
            else:
                self.current_plane = 'Transax'
                # rollover to previous loc
                if self.current_loc == 'THORAX-ABDOMEN': self.current_loc = 'HEAD-THORAX'
                else:
                    self.current_loc = 'THORAX-ABDOMEN'
                    # rollover to previous mouse
                    if self.current_mouse_ID == 0:
                        self.current_mouse_ID = len(self.mouse_list) - 1
                    else: 
                        self.current_mouse_ID = self.current_mouse_ID - 1
                    
            # update image
            self.get_3D_img()
            self.current_slice = len(self.current_slice_list) - 1
        else:
            self.current_slice = self.current_slice - 1
            
        return
        
    def set_loc(self, loc):
        # loc is either 'HEAD-THORAX' or 'THORAX-ABDOMEN'
        if not (loc == 'HEAD-THORAX' or loc == 'THORAX-ABDOMEN'):
            print('Error: invalid location given')
            return False
        
        self.current_loc = loc
        
        # update image
        self.get_3D_img()
        
        return True
    
    def set_plane(self, plane):
        # plane is either 'Coronal', 'Sagittal' or 'Transax'
        if not (plane == 'Coronal' or plane == 'Sagittal' or plane == 'Transax'):
            print('Error: invalid plane given')
            return False
        
        self.current_plane = plane
        
        # update image
        self.get_3D_img()
        
        return True
    
    def set_slice(self, slice_i):
        if slice_i >= len(self.current_slice_list) - 1:
            print('Error: invalid slice id given')
            return False
        
        self.current_slice = slice_i
        
        # update image
        self.get_3D_img()
        
        return True
    
    def set_mouse_id(self, mouse_i):
        if mouse_i >= len(self.mouse_list) - 1:
            print('Error: invalid slice id given')
            return False
        
        self.current_mouse_ID = mouse_i
        
        # update image
        self.get_3D_img()
        
        return True
    
    def next_mouse(self):
        if self.current_mouse_ID == len(self.mouse_list) - 1:
            self.current_mouse_ID = 0
        else: 
            self.current_mouse_ID = self.current_mouse_ID + 1

        # update image
        self.get_3D_img()
        
        return True
    
    def previous_mouse(self):
        if self.current_mouse_ID == 0:
            self.current_mouse_ID = len(self.mouse_list) - 1
        else: 
            self.current_mouse_ID = self.current_mouse_ID - 1
    
        # update image
        self.get_3D_img()
        
        return True
    
    def get_slice_ID(self):
        mouse_id = f"{self.current_mouse_ID + 1:02}"

        L = ""
        if self.current_loc == "HEAD-THORAX": L = "H"
        else: L = "T"

        P = ""
        if self.current_plane == "Coronal": P = "C"
        elif self.current_plane == "Sagittal": P = "S"
        else: P = "T"

        slice_id = f"{self.current_slice + 1:02}"

        id = "".join([mouse_id, L, P, slice_id])

        return id
    
    def check_blacklisted(self):
        # read json file
        with open(BLACKLIST_PATH, 'r') as f:
            json_object = json.load(f)
        
        blacklist = json_object.get("blacklist")

        if self.get_slice_ID() in blacklist:
            self.slice_is_blacklisted = True
        else:
            self.slice_is_blacklisted = False
        return

    
## visualizer init
slicenavigator = Slice_navigator()
    
## interface
root = customtkinter.CTk()

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("dark-blue")

root.geometry("750x600")
w = 750
h = 600
ws = root.winfo_screenwidth()
hs = root.winfo_screenheight()
x = (ws/2) - (w/2)
y = (hs/2) - (h/2)
root.geometry('%dx%d+%d+%d' % (w, h, x, y))
root.title("MRI slice Visualizer")
root.resizable(False, False)

frame = customtkinter.CTkFrame(master=root)
frame.pack(side="top", fill="both", expand=True)
## 

def fig_to_pil(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = canvas.get_width_height()
    
    # Get ARGB data and convert to RGB
    argb = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
    argb = argb.reshape((height, width, 4))  # ARGB format (A, R, G, B)
    
    # Swap ARGB → RGB (drop alpha)
    rgb = argb[:, :, 1:]  # Remove alpha channel
    
    image = Image.fromarray(rgb, mode="RGB")
    return image

def refresh_screen():
    global lr_img_ctk
    global lr_img_label
    global hr_img_ctk
    global hr_img_label

    mouse_id_val_label.configure(text=slicenavigator.mouse_list[slicenavigator.current_mouse_ID])
    slice_val_label.configure(text=f'{slicenavigator.current_slice + 1}/{len(slicenavigator.current_slice_list)}')
    combobox_plane.set(slicenavigator.current_plane)
    combobox_loc.set(slicenavigator.current_loc)

    # get slices to show image
    plt.close() # close previous plot instances
    lr_img, hr_img = slicenavigator.get_slice()

    fig_lr, ax_lr = plt.subplots(figsize=(5, 5))
    ax_lr.imshow(lr_img, cmap="viridis")
    ax_lr.axis("off")
    fig_lr.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig_hr, ax_hr = plt.subplots(figsize=(5, 5))
    ax_hr.imshow(hr_img, cmap="viridis")
    ax_hr.axis("off")
    fig_hr.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Convert Matplotlib figure to a PIL image
    pil_image_hr = fig_to_pil(fig_hr)
    pil_image_lr = fig_to_pil(fig_lr)

    # update ctk image
    lr_img_ctk = customtkinter.CTkImage(light_image=pil_image_lr, size=(300,300))
    lr_img_label.configure(image=lr_img_ctk)

    hr_img_ctk = customtkinter.CTkImage(light_image=pil_image_hr, size=(300,300))
    hr_img_label.configure(image=hr_img_ctk)

    # Close the figures to free memory
    plt.close(fig_lr)
    plt.close(fig_hr)

    id_label_lr.configure(text="".join([slicenavigator.get_slice_ID(), "L"]))
    id_label_hr.configure(text="".join([slicenavigator.get_slice_ID(), "H"]))

    ps_hr = slicenavigator.hr_metadata['ps']
    ps_label_val_hr.configure(text=f'{ps_hr} mm')
    ps_lr = slicenavigator.lr_metadata['ps']
    ps_label_val_lr.configure(text=f'{ps_lr} mm')

    st_hr = slicenavigator.hr_metadata['st']
    st_label_val_hr.configure(text=f'{st_hr} mm')
    st_lr = slicenavigator.lr_metadata['st']
    st_label_val_lr.configure(text=f'{st_lr} mm')

    shape_hr = slicenavigator.hr_metadata['shape'][:-1]
    dim_label_val_hr.configure(text=f'{shape_hr[0]}x{shape_hr[1]} pixels')
    shape_lr = slicenavigator.lr_metadata['shape'][:-1]
    dim_label_val_lr.configure(text=f'{shape_lr[0]}x{shape_lr[1]} pixels')

    slicenavigator.check_blacklisted()
    if slicenavigator.slice_is_blacklisted:
        blacklist_box.select()
    else:
        blacklist_box.deselect()


# Mouse ID
mouse_id_label = customtkinter.CTkLabel(text="Mouse ID: ", master=root, text_color="#366abf", font=(None,18))
mouse_id_label.place(relx = 0.01, rely=0.01)
mouse_id_val_label = customtkinter.CTkLabel(text="Mouse01", master=root, text_color='#c92477', font=(None,18))
mouse_id_val_label.place(relx = 0.15, rely=0.01)

def next_mouse():
    slicenavigator.next_mouse()
    refresh_screen()

def previous_mouse():
    slicenavigator.previous_mouse()
    refresh_screen()

next_mouse_button = customtkinter.CTkButton(text=">", master=root, corner_radius=5, command=lambda: next_mouse())
next_mouse_button.place(relx=0.5, rely=0.035, relwidth=0.05, anchor=tkinter.CENTER)
previous_mouse_button = customtkinter.CTkButton(text="<", master=root, corner_radius=5, command=lambda: previous_mouse())
previous_mouse_button.place(relx=0.4, rely=0.035, relwidth=0.05, anchor=tkinter.CENTER)

# Location
location_label = customtkinter.CTkLabel(text="Location: ", master=root, text_color="#366abf", font=(None,18))
location_label.place(relx = 0.01, rely=0.066)

def combobox_loc_callback(choice):
    slicenavigator.set_loc(choice)
    refresh_screen()

combobox_loc = customtkinter.CTkComboBox(root, values=["HEAD-THORAX", "THORAX-ABDOMEN"],command=combobox_loc_callback)
combobox_loc.place(relx = 0.15, rely=0.07, relwidth=0.25)

# plane
plane_label = customtkinter.CTkLabel(text="Plane: ", master=root, text_color="#366abf", font=(None,18))
plane_label.place(relx = 0.01, rely=0.122)

def combobox_plane_callback(choice):
    slicenavigator.set_plane(choice)
    refresh_screen()

combobox_plane = customtkinter.CTkComboBox(root, values=["Coronal", "Sagittal", "Transax"],command=combobox_plane_callback)
combobox_plane.place(relx = 0.15, rely=0.126, relwidth=0.25)

# slice
slice_label = customtkinter.CTkLabel(text="Slice: ", master=root, text_color="#366abf", font=(None,18))
slice_label.place(relx = 0.01, rely=0.178)
slice_val_label = customtkinter.CTkLabel(text="1", master=root, text_color='#c92477', font=(None,18))
slice_val_label.place(relx = 0.15, rely=0.178)

def next_slice():
    slicenavigator.next_slice()
    refresh_screen()

def previous_slice():
    slicenavigator.previous_slice()
    refresh_screen()

next_mouse_button = customtkinter.CTkButton(text=">", master=root, corner_radius=5, command=lambda: next_slice())
next_mouse_button.place(relx=0.60, rely=0.95, relwidth=0.15, anchor=tkinter.CENTER)
previous_mouse_button = customtkinter.CTkButton(text="<", master=root, corner_radius=5, command=lambda: previous_slice())
previous_mouse_button.place(relx=0.40, rely=0.95, relwidth=0.15, anchor=tkinter.CENTER)

# images
lowres_label = customtkinter.CTkLabel(text="LOW RESOLUTION", master=root, text_color="#5ab098", font=(None,18))
lowres_label.place(relx = 0.14, rely=0.23, relwidth=0.25)
highres_label = customtkinter.CTkLabel(text="HIGH RESOLUTION", master=root, text_color="#5ab098", font=(None,18))
highres_label.place(relx = 0.62, rely=0.23, relwidth=0.25)

lr_img_ctk = customtkinter.CTkImage(light_image=Image.new("RGB", (500, 500)), size=(500,500)) # placeholder image
lr_img_label = customtkinter.CTkLabel(root, image=lr_img_ctk, text="")
lr_img_label.place(relx=0.06, rely=0.28)
hr_img_ctk = customtkinter.CTkImage(light_image=Image.new("RGB", (500, 500)), size=(500,500)) # placeholder image
hr_img_label = customtkinter.CTkLabel(root, image=lr_img_ctk, text="")
hr_img_label.place(relx=0.54, rely=0.28)

# play button
play_enabled = False
timer = None
cancelled = False

def periodic_update():
    global timer
    global next_slice
    global cancelled
    global speed_slider

    if cancelled:
        cancelled = False
        return

    next_slice()

    interval = round(10**(speed_slider.get()) * 100)/100
    
    timer = threading.Timer(interval, lambda: periodic_update())
    timer.start()

def stop_timer():
    global cancelled
    cancelled = True

def play():
    global play_enabled
    global timer
    global next_slice
    global stop_timer
    global periodic_update
    if play_enabled:
        # stop threas
        stop_timer()
        play_enabled = False
        play_button.configure(text='▷')
    else:
        # start thread
        periodic_update()
        play_enabled = True
        play_button.configure(text='‖')

play_button = customtkinter.CTkButton(text="▷", master=root, corner_radius=5, command=lambda: play())
play_button.place(relx=0.80, rely=0.09, relwidth=0.05, anchor=tkinter.CENTER)

speed_label = customtkinter.CTkLabel(text="Automatic", master=root, text_color="#366abf", font=(None,18))
speed_label.place(relx = 0.74, rely=0.01)
def slider_event_speed(value):
    speed_label_value.configure(text=f'speed (interval): {round(10**(value) * 100)/100}s')
speed_slider = customtkinter.CTkSlider(master=root, from_=0.3, to=-1, command=slider_event_speed, number_of_steps=10)
speed_slider.place(relx=0.70, rely=0.12, relwidth=0.20)
speed_slider.set(0.3)
speed_label_value = customtkinter.CTkLabel(text=f'speed (interval): {round(10**(0.3) * 100)/100}s', master=root, fg_color="transparent", text_color="#5ab098")
speed_label_value.place(relx=0.71, rely=0.15)

# Blacklist
blacklist_label = customtkinter.CTkLabel(text="Blacklist slice: ", master=root, text_color='#c92477', font=(None,18))
blacklist_label.place(relx=0.78, rely=0.925)

def blacklist_event():
    ## add/remove from blacklist.json
    # read json file
    with open(BLACKLIST_PATH, 'r') as f:
        json_object = json.load(f)
    
    blacklist = json_object.get("blacklist")

    id = slicenavigator.get_slice_ID()

    if id in blacklist and not blacklist_box.get():
        # remove
        blacklist.remove(id)
        json_object = json.dumps({"blacklist": blacklist})
        with open(BLACKLIST_PATH, 'w') as f:
            f.write(json_object)
        
    elif id not in blacklist and blacklist_box.get():
        # add
        blacklist.append(id)
        json_object = json.dumps({"blacklist": blacklist})
        with open(BLACKLIST_PATH, 'w') as f:
            f.write(json_object)

blacklist_box = customtkinter.CTkCheckBox(text="", width=10, master=root, command=blacklist_event, fg_color='#e00b0f')
blacklist_box.place(relx=0.95, rely=0.93)

# Metadata
ps_label_lr = customtkinter.CTkLabel(text="Pixel size: ", master=root, text_color='#366abf', font=(None,15))
ps_label_lr.place(relx=0.06, rely=0.78)
ps_label_val_lr = customtkinter.CTkLabel(text="", master=root, text_color="#d4661e", font=(None,15))
ps_label_val_lr.place(relx=0.24, rely=0.78)
ps_label_hr = customtkinter.CTkLabel(text="Pixel size: ", master=root, text_color='#366abf', font=(None,15))
ps_label_hr.place(relx=0.54, rely=0.78)
ps_label_val_hr = customtkinter.CTkLabel(text="", master=root, text_color="#d4661e", font=(None,15))
ps_label_val_hr.place(relx=0.72, rely=0.78)
st_label_lr = customtkinter.CTkLabel(text="Slice Thickness: ", master=root, text_color='#366abf', font=(None,15))
st_label_lr.place(relx=0.06, rely=0.82)
st_label_val_lr = customtkinter.CTkLabel(text="", master=root, text_color="#d4661e", font=(None,15))
st_label_val_lr.place(relx=0.24, rely=0.82)
st_label_hr = customtkinter.CTkLabel(text="Slice Thickness: ", master=root, text_color='#366abf', font=(None,15))
st_label_hr.place(relx=0.54, rely=0.82)
st_label_val_hr = customtkinter.CTkLabel(text="", master=root, text_color="#d4661e", font=(None,15))
st_label_val_hr.place(relx=0.72, rely=0.82)
dim_label_lr = customtkinter.CTkLabel(text="Image Dimension: ", master=root, text_color='#366abf', font=(None,15))
dim_label_lr.place(relx=0.06, rely=0.86)
dim_label_val_lr = customtkinter.CTkLabel(text="", master=root, text_color="#d4661e", font=(None,15))
dim_label_val_lr.place(relx=0.24, rely=0.86)
dim_label_hr = customtkinter.CTkLabel(text="Image Dimension: ", master=root, text_color='#366abf', font=(None,15))
dim_label_hr.place(relx=0.54, rely=0.86)
dim_label_val_hr = customtkinter.CTkLabel(text="", master=root, text_color="#d4661e", font=(None,15))
dim_label_val_hr.place(relx=0.72, rely=0.86)

show_id_label = customtkinter.CTkLabel(text="Show ID: ", master=root, text_color='#366abf', font=(None,18))
show_id_label.place(relx=0.01, rely=0.925)

def show_id_event():
    if show_id.get():
        id_label_lr.lift(frame)
        id_label_hr.lift(frame)
    else: 
        id_label_lr.lower(frame)
        id_label_hr.lower(frame)

show_id = customtkinter.CTkCheckBox(text="", width=10, master=root, command=show_id_event)
show_id.place(relx=0.12, rely=0.93)

id_label_lr = customtkinter.CTkLabel(text="01HC01L", master=root, text_color="#ffffff", font=(None,12), bg_color='transparent')
id_label_lr.place(relx=0.39, rely=0.78)
id_label_hr = customtkinter.CTkLabel(text="01HC01H", master=root, text_color="#ffffff", font=(None,12), bg_color='transparent')
id_label_hr.place(relx=0.87, rely=0.78)
show_id.select()

refresh_screen()
root.mainloop()