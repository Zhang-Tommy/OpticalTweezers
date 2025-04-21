import numpy as np
import h5py
from scipy.fft import fft2, fftshift
from multiprocessing import Pool, cpu_count
import os
import tempfile
from phase_calculator import calculate_phase_mask

max_num_spots = 15
N = 200000
sizes = [128]
num_processes = cpu_count()  # or set to your preferred number

def generate_data(args):
    """Worker function to generate a chunk of data"""
    start_idx, end_idx, mask_sz, process_id = args
    l = int((mask_sz / 512) * 16)
    line_len = (mask_sz / 512) * 6

    # Create a temporary file for this process
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.hdf5')
    temp_path = temp_file.name
    temp_file.close()

    with h5py.File(temp_path, 'w') as f:
        inputs = f.create_dataset("inputs", (end_idx - start_idx, mask_sz, mask_sz))
        outputs = f.create_dataset("outputs", (end_idx - start_idx, mask_sz, mask_sz))

        for idx in range(start_idx, end_idx):
            rand_num_spots = np.random.randint(1, max_num_spots)
            spot_params = np.zeros((rand_num_spots, 4, 4))

            rand_x = np.random.uniform(0, (mask_sz / 512) * 120, rand_num_spots)
            rand_y = np.random.uniform(-(mask_sz / 512) * 120, 0, rand_num_spots)

            spot_params[:, 0, 0] = rand_x
            spot_params[:, 0, 1] = rand_y
            spot_params[:, 1, 0] = 1  # intensity
            spot_params[:, 2, 2] = 1  # na.r

            for trap in spot_params:
                trap_type = np.random.uniform()
                if 0.8 <= trap_type < 0.9:
                    trap[0, 3] = l
                    trap[1, 0] = 1.5
                elif trap_type >= 0.9:
                    trap[3, 0] = line_len
                    trap[1, 0] = 1.5

            phase_masks_reference = calculate_phase_mask(spot_params, rand_num_spots, mask_sz, False)[0]
            phase_masks_corrected = calculate_phase_mask(spot_params, rand_num_spots, mask_sz, True)[0]
            phase_masks_reference = np.rot90(phase_masks_reference, 1)
            phase_masks_reference = np.flip(phase_masks_reference, axis=0)

            def gaussian_beam(mask_size, beam_width):
                x = np.linspace(-1, 1, mask_size)
                y = np.linspace(-1, 1, mask_size)
                xx, yy = np.meshgrid(x, y)
                return np.exp(-(xx ** 2 + yy ** 2) / (2 * beam_width ** 2))

            def compute_beam_width(mask_sz, reference_size=512, reference_bm=0.05):
                return reference_bm * (reference_size / mask_sz)

            bm = compute_beam_width(mask_sz)
            incident_beam = gaussian_beam(mask_sz, beam_width=bm)

            slm_field = incident_beam * np.exp(1j * phase_masks_reference)

            far_field = fftshift(fft2(slm_field)) / ((2 * mask_sz) ** 2)
            intensity = np.abs(far_field) ** 2

            # f, axarr = plt.subplots(2, 1)
            # axarr[0].imshow(intensity, cmap='gray')
            # axarr[1].imshow(phase_masks_reference, cmap='gray')
            # plt.show()

            intensity = intensity / np.max(intensity)

            inputs[idx - start_idx] = intensity
            outputs[idx - start_idx] = phase_masks_corrected / np.pi

            if (idx - start_idx) % 1000 == 0:
                print(f"Process {process_id}: {idx - start_idx}/{end_idx - start_idx}")

    return temp_path


def parallel_generate_data():
    for mask_sz in sizes:
        final_N = N
        chunk_size = final_N // num_processes

        # Prepare arguments for each process
        args_list = []
        for i in range(num_processes):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_processes - 1 else final_N
            args_list.append((start, end, mask_sz, i))

        # Use multiprocessing
        with Pool(processes=num_processes) as pool:
            temp_files = pool.map(generate_data, args_list)

        # Merge temporary files into final output
        final_path = f"./train_data/unet_{final_N}_{mask_sz}.hdf5"
        with h5py.File(final_path, 'w') as final_f:
            final_inputs = final_f.create_dataset("inputs", (final_N, mask_sz, mask_sz))
            final_outputs = final_f.create_dataset("outputs", (final_N, mask_sz, mask_sz))

            offset = 0
            for temp_file in temp_files:
                with h5py.File(temp_file, 'r') as temp_f:
                    chunk_size = temp_f['inputs'].shape[0]
                    final_inputs[offset:offset + chunk_size] = temp_f['inputs'][:]
                    final_outputs[offset:offset + chunk_size] = temp_f['outputs'][:]
                    offset += chunk_size
                os.unlink(temp_file)  # Clean up temporary file

        print(f"Finished generating {final_N} samples at {mask_sz}x{mask_sz} resolution")


if __name__ == "__main__":
    parallel_generate_data()