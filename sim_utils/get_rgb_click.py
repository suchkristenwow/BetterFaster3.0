import cv2
import argparse

# Declare image as a global variable
image = None

# Function to handle mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_color = image[y, x]
        print(f"Clicked at (x={x}, y={y}) - RGB value: {pixel_color}")

def main():
    global image  # Declare image as global
    # Create argument parser
    parser = argparse.ArgumentParser(description="Display an image and get pixel RGB value on click.")
    parser.add_argument("image_path", help="Path to the image file")
    args = parser.parse_args()

    # Load the image from the command-line argument
    image = cv2.imread(args.image_path)

    if image is None:
        print(f"Error: Unable to load the image from '{args.image_path}'")
        return

    # Create a window and set the mouse callback function
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    # Display the image
    cv2.imshow('Image', image)

    while True:
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()