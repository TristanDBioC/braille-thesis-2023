# img_viewer.py

import PySimpleGUI as sg
import os.path
from PIL import Image
from PIL.ExifTags import TAGS

# First the window layout in 2 columns


file_list_column = [
    [
        sg.Text("Image Folder"),
        sg.In(size=(50, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Listbox(
            values=[], enable_events=True, size=(70, 30), key="-FILE LIST-"
        )
    ],
    [
        layout = [[sg.Button("Open New Window", key="-OPEN-")]]

        window = sg.Window("Main Window", layout)

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED:
                break
            elif event == "-OPEN-":
                new_layout = [[sg.Text("This is a new window!")], [sg.Button("Close", key="-CLOSE-")]]
                new_window = sg.Window("New Window", new_layout)
                while True:
                    new_event, new_values = new_window.read()
                    if new_event == sg.WIN_CLOSED or new_event == "-CLOSE-":
                        break
                new_window.close()

        window.close()

    ],
]

# For now will only show the name of the file that was chosen
image_viewer_column = [
    [sg.Text("Choose an image from list on left:")],
    [sg.Text(size=(20, 1), key="-TOUT-")],
    [sg.Image(key="-IMAGE-")],
]

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column),
    ]
]

window = sg.Window("Image Viewer", layout)

# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            image = Image.open(fnames)

            info_dict = {
            "Filename": image.filename,
            "Image Size": image.size,
            "Image Height": image.height,
            "Image Width": image.width,
            "Image Format": image.format,
            "Image Mode": image.mode,
            "Image is Animated": getattr(image, "is_animated", False),
            "Frames in Image": getattr(image, "n_frames", 1)
            }

            for label,value in info_dict.items():
                sg.Text(f"{label:25}: {value}")

            exifdata = image.getexif()

            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id)
                # decode bytes 
                if isinstance(data, bytes):
                    data = data.decode()
                print(f"{tag:25}: {data}")

        except:
            pass

window.close()