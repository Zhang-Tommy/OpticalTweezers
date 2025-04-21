import numpy as np
import h5py
import jax
import matplotlib.pyplot as plt
from phase_calculator import batch_calculate_masks, calculate_phase_mask
from scipy.fft import fft2, fftshift
import time

max_num_spots = 15
N = 1000
sizes = [512]


for mask_sz in sizes:
    f = h5py.File(f"./train_data/unet_{N}_{mask_sz}.hdf5", "w")
    inputs = f.create_dataset("inputs", (N, mask_sz, mask_sz))
    outputs = f.create_dataset("outputs", (N, mask_sz, mask_sz))
    l = int((mask_sz / 512) * 16)
    line_len = (mask_sz / 512) * 6

    tot_time = 0
    for idx in range(N):
        rand_num_spots = np.random.randint(1, max_num_spots)
        #rand_num_spots = 1
        spot_params = np.zeros((rand_num_spots, 4, 4))

        rand_x = np.random.uniform(0, (mask_sz / 512) * 120, rand_num_spots)
        rand_y = np.random.uniform(-(mask_sz / 512) * 120, 0, rand_num_spots)

        spot_params[:, 0, 0] = rand_x
        spot_params[:, 0, 1] = rand_y
        spot_params[:, 1, 0] = 1 # intensity
        spot_params[:, 2, 2] = 1 # na.r

        for trap in spot_params:
            trap_type = np.random.uniform()
            if 0.7 <= trap_type < 0.8:
                trap[0, 3] = l
                trap[1, 0] = 1.5
            elif trap_type >= 0.8:
                trap[3, 0] = line_len
                trap[1, 0] = 1.5

        st = time.time()
        phase_masks_reference = calculate_phase_mask(spot_params, rand_num_spots, mask_sz, False)[0]
        et = time.time()
        #print(et - st)
        tot_time += et - st
        phase_masks_corrected = calculate_phase_mask(spot_params, rand_num_spots, mask_sz, True)[0]

        phase_masks_reference = np.rot90(phase_masks_reference, 1)
        phase_masks_reference = np.flip(phase_masks_reference, axis=0)

        phase_masks_corrected = np.rot90(phase_masks_corrected, 1)
        phase_masks_corrected = np.flip(phase_masks_corrected, axis=0)

        def gaussian_beam(mask_size, beam_width):
            x = np.linspace(-1, 1, mask_size)
            y = np.linspace(-1, 1, mask_size)
            xx, yy = np.meshgrid(x, y)
            return np.exp(-(xx ** 2 + yy ** 2) / (2*beam_width ** 2))

        def compute_beam_width(mask_sz, reference_size=512, reference_bm=0.05):
            return reference_bm * (reference_size / mask_sz)

        bm = compute_beam_width(mask_sz)
        incident_beam = gaussian_beam(mask_sz, beam_width=bm)

        slm_field = incident_beam * np.exp(1j * phase_masks_reference)
        far_field = fftshift(fft2(slm_field)) / ((2*mask_sz)**2)
        intensity = np.abs(far_field)

        slm_field_c = incident_beam * np.exp(1j * phase_masks_corrected)
        far_field_c = fftshift(fft2(slm_field_c)) / ((2*mask_sz)**2)
        intensity_c = np.abs(far_field_c)

        # f, axarr = plt.subplots(4, 1)
        # axarr[0].imshow(intensity, cmap='gray')
        # axarr[1].imshow(phase_masks_reference, cmap='gray')
        # axarr[2].imshow(phase_masks_corrected, cmap='gray')
        # axarr[3].imshow(intensity_c, cmap='gray')
        # plt.show()

        intensity = intensity / np.max(intensity)

        #np.save("./ref_intensity/point_trap_128.npy", intensity)
        inputs[idx:(idx + 1), :, :] = intensity
        outputs[idx:(idx + 1), :, :] = phase_masks_reference / np.pi
        if idx % 1000 == 0:
            print(f"{idx + 1000} / {N}")

    print(f"Avg Time: {tot_time / N}")