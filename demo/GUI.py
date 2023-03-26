import os
import PySimpleGUI as sg
from PIL import Image
import os.path
from PIL.ExifTags import TAGS

# function to get image metadata
def get_image_metadata(filename):
    with Image.open(filename) as img:
        metadata = img._getexif()
    return metadata

# define GUI layout
layout = [
    [sg.Text("Select a folder:"),
    sg.FolderBrowse(key="-FOLDER-")],
    [sg.Text("Files in folder:")],
    [sg.Column([
        [sg.Listbox(values=[], size=(70, 30), key="-FILE LIST-", enable_events=True)]
    ]),
    sg.VSeperator(),
    sg.Column([
        [sg.Text("Selected image metadata:")],
        [sg.Text(size=(20, 1), key="-METADATA-")],
    ])],
    [sg.Button("Open new window")]
]


# create the GUI window
window = sg.Window("Image Metadata Viewer", layout)

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

    # display metadata when image is selected
    if event == "-FILE LIST-":
        filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
        metadata = get_image_metadata(filename)
        window["-METADATA-"].update(str(metadata))

    # open a new window with the same features
    if event == "Open new window":
        new_window = sg.Window("Image Metadata Viewer", layout)
        while True:
            new_event, new_values = new_window.read()
            if new_event == sg.WIN_CLOSED or new_event == "Exit":
                break
            if new_event == "-FOLDER-":
                folder = new_values["-FOLDER-"]
                fnames = [f for f in os.listdir(folder)]
                new_window["-FILE LIST-"].update(fnames)
            if new_event == "-FILE LIST-":
                filename = os.path.join(new_values["-FOLDER-"], new_values["-FILE LIST-"][0])
                metadata = get_image_metadata(filename)
                new_window["-METADATA-"].update(str(metadata))
        new_window.close()

window.close()
