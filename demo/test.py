import PySimpleGUI as sg
from PIL import Image

layout = [
        [sg.Button("Open New Window", key="-OPEN-")],
        [sg.Text("Select an image file:")],
        [sg.Input(key="-FILE-"), sg.FileBrowse()],
        [sg.Button("Show Metadata"), sg.Button("Exit")]
        ]

# Create the main window
window = sg.Window("Image Metadata Viewer", layout)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    elif event == "Show Metadata":
        try:
            filename = values["-FILE-"]
            image = Image.open(filename)
            metadata = image._getexif()
            if metadata:
                # Convert the metadata dictionary to a string for display
                metadata_str = ""
                for tag, value in metadata.items():
                    tag_str = f"{tag}: {value}\n"
                    metadata_str += tag_str
            else:
                metadata_str = "No metadata found."
            # Create a popup window to display the metadata
            sg.popup("Image Metadata", metadata_str)
        except Exception as e:
            sg.popup_error(f"Error: {e}")
window.close()

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
        

