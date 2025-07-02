
import torch
import torch.nn.functional as F
from motion import align_image
import numpy as np
from new_utils import to_cv2_image, numpy_to_tensor



'''def frame_denoise(model, noise_frame, sigma_map, context):
    _, _, h, w = noise_frame.shape
    pad_h = (4 - h % 4)
    pad_w = (4 - w % 4)
    if pad_h or pad_w:
        noise_frame = F.pad(noise_frame, (0, pad_w, 0, pad_h), mode="reflect")
        sigma_map = F.pad(sigma_map, (0, pad_w, 0, pad_h), mode="reflect")
    with context:
        denoise_frame = model(noise_frame, sigma_map)
        denoise_frame = torch.clamp(denoise_frame, 0.0, 1.0)
    if pad_h:
        denoise_frame = denoise_frame[:, :, :-pad_h, :]
    if pad_w:
        denoise_frame = denoise_frame[:, :, :, :-pad_w]

    return denoise_frame
    '''


def spatial_denoise(model, image, noise_map):
    size = image.size()
    expand_h = size[-2] % 2
    expand_w = size[-1] % 2
    pad = (0, expand_w, 0, expand_h)
    image = F.pad(input=image, pad=pad, mode='reflect')
    noise_map = F.pad(input=noise_map, pad=pad, mode='reflect')
    denoised_image = torch.clamp(model(image, noise_map), 0., 1.)
    if expand_h != 0:
        denoised_image = denoised_image[:, :, :-1, :]
    if expand_w != 0:
        denoised_image = denoised_image[:, :, :, :-1]
    return denoised_image
def temporal_denoise(model, images, noise_map):
    size = images.size()
    expand_h = size[-2] % 2
    expand_w = size[-1] % 2
    pad = (0, expand_w, 0, expand_h)
    images = F.pad(input=images, pad=pad, mode='reflect')
    noise_map = F.pad(input=noise_map, pad=pad, mode='reflect')
    denoised_images = torch.clamp(model(images, noise_map), 0., 1.)
    if expand_h != 0:
        denoised_images =denoised_images[:, :, :-1, :]
    if expand_w != 0:
        denoised_images =denoised_images[:, :, :, :-1]
    return denoised_images


def denoise_seq_dvdnet(seq, noise_std, temporal_patch, spatial_model, temporal_model):
    num_frames, C, H, W = seq.size()
    noise_map = torch.full((1, C, H, W), noise_std)
    inframes_wrpd = np.empty((temporal_patch, H, W, C))
    denoise_window = list()
    denframes = torch.empty((num_frames, C, H, W)).to(seq.device)
    for central_frame in range(num_frames):
        if central_frame == 0:
            for i in range(-(temporal_patch//2), (temporal_patch//2)+1):
                index = min(max(central_frame + i, 0), num_frames-1)
                denoise_window.append(spatial_denoise(spatial_model,seq[index],noise_map))
        else:
            del denoise_window[0]
            index = min(max(central_frame+ (temporal_patch//2),0), num_frames-1)
            denoise_window.append(spatial_denoise(spatial_model,seq[index],noise_map))
            # (B,C,H,W)
        for i in [x for x in range(0, temporal_patch) if x != temporal_patch//2]:
            inframes_wrpd[i] = align_image(denoise_window[i],denoise_window[temporal_patch//2])
        inframes_wrpd[temporal_patch//2] = to_cv2_image(denoise_window[temporal_patch//2])
        temporal_seq = numpy_to_tensor(inframes_wrpd, seq.device)
        denframes[central_frame] = temporal_denoise(temporal_model, temporal_seq, noise_map)
    del denoise_window
    del inframes_wrpd
    del temporal_seq
    torch.cuda.empty_cache()
    return denframes






'''
def denoise_seq_fastdvdnet(seq, noise_std, model, temporal_window=5, is_training=False):
    frame_num, c, h, w = seq.shape
    center = (temporal_window - 1) // 2
    denoise_frames = torch.empty_like(seq).to(seq.device)
    noise_map = noise_std.view(1, 1, 1, 1).expand(1, 1, h, w).to(seq.device)
    model.to(seq.device)
    context = torch.enable_grad() if is_training else torch.no_grad()
    frames = []
    with context:
        for denoise_index in range(frame_num):
            # load input frames
            if not frames:

                for index in range(temporal_window):
                    rel_index = abs(index - center)  # handle border conditions, reflect
                    frames.append(seq[rel_index])
            else:
                del frames[0]
                rel_index = min(denoise_index + center,
                                -denoise_index + 2 * (frame_num - 1) - center)  # handle border conditions
                frames.append(seq[rel_index])

            input_tensor = torch.stack(frames, dim=0).view(1, temporal_window * c, h, w).to(seq.device)
            denoise_frames[denoise_index] = frame_denoise(model, input_tensor, noise_map, context)
        del frames
        del input_tensor
        torch.cuda.empty_cache()
        return denoise_frames
'''