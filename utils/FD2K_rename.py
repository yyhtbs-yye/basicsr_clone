import os

# Set the directory containing the images
folder_path = "./datasets/DF2K/DF2K_train_LR_bicubic/X3"  # Change this to your actual folder path

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        # Check if filename contains "x2" before the extension
        new_filename = filename.replace("x3.png", ".png")

        # If the filename actually changed, rename the file
        if new_filename != filename:
            old_path = os.path.join(folder_path, filename)
            new_path = os.path.join(folder_path, new_filename)
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} → {new_filename}')

print("Renaming complete!")
