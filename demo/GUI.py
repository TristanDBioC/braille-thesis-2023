import os
import tkinter as tk
from tkinter import filedialog

# Function to display folder contents and metadata
def display_folder():
    # Clear any previous contents from the listbox and metadata label
    listbox.delete(0, tk.END)
    metadata_label.config(text='')

    # Get the selected folder from the user
    folder_path = filedialog.askdirectory()

    # Get the contents of the folder and add them to the listbox
    folder_contents = os.listdir(folder_path)
    for item in folder_contents:
        listbox.insert(tk.END, item)

    # Get the metadata for the folder and display it
    folder_metadata = os.stat(folder_path)
    metadata_str = f'Size: {folder_metadata.st_size} bytes\nModified: {folder_metadata.st_mtime}'
    metadata_label.config(text=metadata_str)

# Function to open a new window with the same functionalities
def open_new_window():
    # Create a new window
    new_window = tk.Toplevel(root)
    new_window.geometry("500x500")

    # Create a listbox and metadata label for the new window
    listbox_new = tk.Listbox(new_window)
    metadata_label_new = tk.Label(new_window)

    # Add the listbox and metadata label to the new window
    listbox_new.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    metadata_label_new.pack()

    # Create a button to select a folder in the new window
    select_folder_button_new = tk.Button(new_window, text='Select Folder', command=lambda: display_folder_new(listbox_new, metadata_label_new))
    select_folder_button_new.pack(side=tk.TOP, padx=10, pady=10)

# Function to display folder contents and metadata in the new window
def display_folder_new(listbox_new, metadata_label_new):
    # Clear any previous contents from the listbox and metadata label
    listbox_new.delete(0, tk.END)
    metadata_label_new.config(text='')

    # Get the selected folder from the user
    folder_path = filedialog.askdirectory()

    # Get the contents of the folder and add them to the listbox
    folder_contents = os.listdir(folder_path)
    for item in folder_contents:
        listbox_new.insert(tk.END, item)

    # Get the metadata for the folder and display it
    folder_metadata = os.stat(folder_path)
    metadata_str = f'Size: {folder_metadata.st_size} bytes\nModified: {folder_metadata.st_mtime}'
    metadata_label_new.config(text=metadata_str)

# Create the main window
root = tk.Tk()
root.geometry("500x500")

# Create a listbox and metadata label for the main window
listbox = tk.Listbox(root)
metadata_label = tk.Label(root)

# Add the listbox and metadata label to the main window
listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
metadata_label.pack()

# Create a button to select a folder in the main window
select_folder_button = tk.Button(root, text='Select Folder', command=display_folder)
select_folder_button.pack(side=tk.TOP, padx=5, pady=5)

# Create a button to open a new window with the same functionalities
new_window_button = tk.Button(root, text='Open New Window', command=open_new_window)
new_window_button.pack(side=tk.BOTTOM, padx=10, pady=10)

# Start the main loop
root.mainloop()
