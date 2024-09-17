import os

# Directory containing PNG files
image_dir = "F:\\Code\\VOOD\\data\\frames"
output_file = "F:\\Code\\VOOD\\data\\binary\\Polyphia.data"

with open(output_file, 'wb') as output:
    for filename in sorted(os.listdir(image_dir)):
        if filename.endswith(".png"):
            with open(os.path.join(image_dir, filename), 'rb') as image_file:
                output.write(image_file.read())
                output.write(b'\n')  # Add newline if needed
print(f"Binary file {output_file} created successfully.")
