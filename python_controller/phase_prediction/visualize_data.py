import h5py
import numpy as np
import cv2


def view_py():
    N = 100000
    phase_size = 128
    view_size = 512
    file_path = f"./train_data/unet_{N}_{phase_size}.hdf5"
    refresh_rate = 1000  # Frames per second

    with h5py.File(file_path, "r") as f:
        inputs = f["inputs"]  # Access input dataset
        outputs = f["outputs"]  # Access output dataset
        num_samples = inputs.shape[0]
        inputs = np.array(inputs)
        outputs = np.array(outputs)
        print(f"Total samples: {num_samples}")

        # Generate a list of indices and shuffle them
        indices = np.arange(num_samples)
        np.random.shuffle(indices)  # Shuffle the indices#

        for i in indices:  # Iterate through the shuffled indices
            input_img = inputs[i]  # Load input image using shuffled index
            output_img = outputs[i]  # Load output image using shuffled index
            #print(np.max(input_img))
            # Normalize images to 0-255 for OpenCV display
            input_img = ((input_img - input_img.min()) / (input_img.max() - input_img.min()) * 255).astype(np.uint8)
            output_img = ((output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255).astype(np.uint8)

            # Resize images for better visualization
            input_img_resized = cv2.resize(input_img, (view_size, view_size))
            output_img_resized = cv2.resize(output_img, (view_size, view_size))

            # Stack images side by side
            combined = np.hstack([input_img_resized, output_img_resized])

            # Display the images
            cv2.imshow("HDF5 Data Slideshow (Left: Input, Right: Output)", combined)

            # Wait for the refresh rate duration (convert FPS to milliseconds)
            key = cv2.waitKey(int(1000 / refresh_rate))
            if key == 27:  # Press 'ESC' to exit early
                break

        cv2.destroyAllWindows()

def view_mat():
    #File path to your HDF5 dataset
    far_field_path = '../../../Archive/unet/train_data/farfield_full.h5'
    phase_mask_path = '../../../Archive/unet/train_data/phase_masks_full.h5'

    refresh_rate = 1  # Frames per second
    phase_size = 512

    # Open the HDF5 file
    with h5py.File(far_field_path, "r") as f1:
        with h5py.File(phase_mask_path, "r") as f2:
            inputs = f1['/far_fields']  # Access input dataset

            outputs = f2['/phase_masks']  # Access output dataset
            num_samples = inputs.shape[0]

            print(f"Total samples: {num_samples}")

            # Generate a list of indices and shuffle them
            indices = np.arange(num_samples)
            np.random.shuffle(indices)  # Shuffle the indices

            for i in indices:  # Iterate through the shuffled indices
                input_img = inputs[i]  # Load input image using shuffled index
                output_img = outputs[i]  # Load output image using shuffled index

                # Normalize images to 0-255 for OpenCV display
                input_img = ((input_img - input_img.min()) / (input_img.max() - input_img.min()) * 255).astype(np.uint8)
                output_img = ((output_img - output_img.min()) / (output_img.max() - output_img.min()) * 255).astype(np.uint8)

                # Resize images for better visualization
                input_img_resized = cv2.resize(input_img, (phase_size, phase_size))
                output_img_resized = cv2.resize(output_img, (phase_size, phase_size))

                # Stack images side by side
                combined = np.hstack([input_img_resized, output_img_resized])

                # Display the images
                cv2.imshow("HDF5 Data Slideshow (Left: Input, Right: Output)", combined)

                # Wait for the refresh rate duration (convert FPS to milliseconds)
                key = cv2.waitKey(int(1000 / refresh_rate))
                if key == 27:  # Press 'ESC' to exit early
                    break

            cv2.destroyAllWindows()

#view_mat()
view_py()
