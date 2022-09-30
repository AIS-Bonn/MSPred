"""
Methods for data visualization
"""
import os
import numpy as np
from math import ceil
import torch
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from webcolors import name_to_rgb
from scipy.ndimage import maximum_filter
import torch.nn.functional as F
import cv2
from data.heatmaps import HeatmapGenerator


def visualize_metric(vals, x_axis=None, title=None, xlabel=None, savepath=None, **kwargs):
    """ Function for visualizing the average metric per frame """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(1, 1)
    x_axis = x_axis if x_axis is not None else np.arange(len(vals))
    ax.plot(x_axis, vals, linewidth=3)
    ax.scatter(x_axis, vals)
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    return


def make_gif_hierarch(gif_frames, context, savepath, gif_names, periods, pad=2, interval=55):
    """
    Creating a plot of several GIFs in a grid.
    """
    fig, ax = plt.subplots(len(gif_frames), 2)
    fig.set_size_inches(4, 5)

    n_preds = len(gif_frames[1][0])
    n_frames = n_preds * (max(periods) if periods else 1) + context
    update_ = lambda i: update_hierarch(i, ax, gif_frames, gif_names, context, periods, pad)
    anim = FuncAnimation(fig, update_, frames=np.arange(n_frames), interval=interval)

    # print(f"Saving GIF...{savepath}")
    plt.tight_layout()
    anim.save(savepath, dpi=120, writer='imagemagick')
    fig.clear()
    return


def make_gif_combined(gif_frames, context, savepath, gif_names, pad=2, interval=55):
    """
    Creating a plot of several GIFs side by side.
    """
    n_gifs, n_frames = len(gif_frames), len(gif_frames[0])
    fig, ax = plt.subplots(1, n_gifs)
    fig.set_tight_layout(True)

    colors = {i: "green" if i < context else "red" for i in range(n_frames)}
    update_ = lambda i: update_combined(i, ax, gif_frames, f"Frame {i}", gif_names, colors[i], pad)

    anim = FuncAnimation(fig, update_, frames=np.arange(n_frames), interval=interval)
    # print(f"Saving GIF...{savepath}")
    anim.save(savepath, dpi=120, writer='imagemagick')
    fig.clear()
    return


def make_gif(sequence, context, savepath, pad=2, interval=55):
    """
    Creating a GIF displaying the sequence

    Args:
    ------
    sequence: torch Tensor
        Tensor containing the sequence of images (T,C,H,W)
    context: integer
        number of frames used for context
    savepath: string
        path where GIF is stored
    pad: integer
        number of pixels to pad with color
    interval: integer
        number of milliseconds in between frames (55=18fps)
    """
    fig, ax = plt.subplots()
    fig.set_tight_layout(True)

    gif_frames = sequence.permute(0, 2, 3, 1).detach().clamp(0, 1).numpy()
    n_frames = len(gif_frames)
    colors = {i: "green" if i < context else "red" for i in range(n_frames)}
    upd = lambda i: update(frame=gif_frames[i], pad=pad, color=colors[i], ax=ax, header_title=f"Frame {i}")

    anim = FuncAnimation(fig, upd, frames=np.arange(n_frames), interval=interval)
    anim.save(savepath, dpi=50, writer='imagemagick')
    ax.axis("off")
    fig.clear()
    return


def update_hierarch(i, axes, gif_frames, gif_names, context, periods, pad):
    """
    Auxiliary function to plot gif frames
    """
    for h, (pred_seq, gt_seq) in gif_frames.items():
        if i < context:
            pred_frame, gt_frame = gif_frames[0][0][i], gif_frames[0][0][i]
        else:
            # repeating intermediate frames if 'periods' is given
            idx = i if h == 0 else ((i-context)//periods[h] if periods else (i-context))
            idx = min(idx, len(pred_seq)-1)
            pred_frame, gt_frame = pred_seq[idx], gt_seq[idx]
        ax = axes[len(axes)-h-1]
        for s, frame in enumerate([pred_frame, gt_frame]):
            color = "green" if (gif_names[s] == "Ground-truth" or i < context) else "red"
            footer_title = gif_names[s] if h == 0 else ""
            update(frame=frame, pad=pad, color=color, ax=ax[s],
                   header_title=f"Frame_{i}", footer_title=footer_title)
    return


def update_combined(i, axes, frames, common_title, gif_names, color="green", pad=2):
    """
    Auxiliary function to plot gif frames
    """
    for s, ax in enumerate(axes):
        update(frame=frames[s][i], pad=pad, color=color, ax=ax,
               header_title=common_title, footer_title=gif_names[s])
    return


def update(frame, color="green", pad=2, header_title="", footer_title="", ax=None):
    """
    Auxiliary function to plot gif frames
    """
    footer_title = footer_title.replace("_", " ")
    header_title = header_title.replace("_", " ")

    disp_frame = add_border(frame, color=color, pad=pad)
    ax.imshow(disp_frame)
    ax.set_title(header_title, fontsize=18)

    ax.set_xticks([], [])
    ax.set_yticks([], [])
    if footer_title != "":
        ax.set_xlabel(footer_title, fontsize=18)
    else:
        ax.axis("off")
    return ax


def add_border(x, color, pad=2, extend=True):
    """
    Adds colored border to frames.

    Args:
    -----
    x: numpy array
        image to add the border to
    color: string
        Color of the border
    pad: integer
    extend: boolean
        Extend the image by padding or not.
    number of pixels to pad each side
    """
    H, W = x.shape[:2]
    x = np.squeeze(x)
    C = 3 if len(x.shape) == 3 else 1
    px_h, px_w = (H+2*pad, W+2*pad) if extend else (H, W)
    px = np.zeros((px_h, px_w, 3))
    color_rgb = name_to_rgb(color)
    for c in range(3):
        px[:, :, c] = color_rgb[c] / 255.

    p = 0 if extend else pad
    x_ = x[p:H-p, p:W-p]
    for c in range(3):
        px[pad:px_h-pad, pad:px_w-pad, c] = (x_ if C == 1 else x_[:, :, c])
    return px


def visualize_sequence(sequence, savepath, seq_id=None, gt=False,
                       add_title=True, add_axis=False, n_cols=10):
    """
    Visualizing a grid with all frames from a sequence
    """
    if seq_id is not None:
        suffix = "_gt" if gt is True else ""
        seq_path = os.path.dirname(savepath) + f"/seq_{seq_id}{suffix}/"
        os.makedirs(seq_path, exist_ok=True)
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3*n_cols, 3*n_rows)

    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_frames > 10 else ax[col]
        img = sequence[i].permute(1, 2, 0).cpu().detach()
        img = torch.squeeze(img, dim=2)
        img = img.clamp(0, 1).numpy()
        cmap = "gray" if img.ndim == 2 else None
        a.imshow(img, cmap=cmap)
        if (add_title):
            a.set_title(f"Frame {i}")
        if (not add_axis):
            a.axis("off")
        if seq_id is not None:
            plt.imsave(seq_path + f"img_{i}.png", img)
    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        fig.clear()
    return


def visualize_preds(sequence, savepath=None, n_seed=4, n_cols=10, suptitle=None, add_title=True,
                    add_axis=False, cmap=None, titles=None, border=True, add_colorbar=False):
    """
    Nice method for visualizing the elements in a sequence.
    Almost the same as 'visualize_sequence', but this works better.
    """
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))

    # displaying all elements in the sequence
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3*n_cols, 3*n_rows)
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        img = sequence[i].permute(1, 2, 0).cpu().detach()
        img = torch.squeeze(img, dim=2)
        img = img.clamp(0, 1).numpy()
        if border:
            color = "green" if i < n_seed else "red"
            img = add_border(img, color=color)
        cmap = "gray" if cmap is None else cmap
        im = a.imshow(img, cmap=cmap)
        if add_colorbar:
            fig.colorbar(im, ax=a)
        # handling titles for each frame
        if (add_title and titles is not None):
            for title in titles:
                a.set_title(title)

    if (not add_axis):
        for ax in fig.axes:
            ax.axis("off")

    # handling global title
    if(suptitle is not None):
        plt.suptitle(suptitle)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    else:
        plt.tight_layout()

    # saving or displaying
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    fig.clear()
    return


def visualize_hierarch_preds(gt_seq, preds, fnums, n_seed, struct_types, seed_stride=4,
                             seq_dir=None, border=False, cmap=None,
                             savepath=None, gt_bgnd=False, save_frames=False):
    """
    Visualize hierarchical predictions - each level on a single row.
    """
    if save_frames is True:
        seed_stride = 1
    n_seed_plt = ceil(n_seed / seed_stride)

    n_rows = 3
    n_cols = n_seed_plt + len(preds[0])
    fig, ax = plt.subplots(n_rows, n_cols)
    fig.set_size_inches(3*n_cols, 3*n_rows)

    for row in range(n_rows):
        if save_frames:
            assert seq_dir is not None
            os.makedirs(seq_dir, exist_ok=True)
            for h in range(n_rows):
                os.makedirs(f"{seq_dir}/hier_{h}", exist_ok=True)
        preds_ = preds[n_rows-row-1]
        fnums_ = fnums[n_rows-row-1]
        for f in range(n_seed_plt+len(preds_)):
            a = ax[row, f]
            if f < n_seed_plt:
                if row != n_rows-1:
                    continue
                fnum = f*seed_stride
                border_color = "green"
                img = gt_seq[fnum]
            else:
                fnum = fnums_[f-n_seed_plt]
                border_color = "red"
                if row == n_rows-1:
                    img = preds_[f-n_seed_plt]
                else:
                    img = gt_seq[fnum]
            img = img.permute(1, 2, 0).cpu().detach()
            img = torch.squeeze(img, dim=2)
            img = img.clamp(0, 1).numpy()
            if border and (row == n_rows-1):
                img = add_border(img, color=border_color, pad=1)
            cmap = "gray" if cmap is None else cmap
            f_str = str(f).zfill(2)
            if row == n_rows-1 or gt_bgnd is True:
                a.imshow(img, cmap=cmap)
                if save_frames and img is not None:
                    plt.imsave(f"{seq_dir}/hier_{n_rows-1-row}/img_{f_str}.png", img)
            a.set_title(f"Frame {fnum}", fontsize=25)
            if row != n_rows-1:
                # plot higher level structured predictions on top of GT frames
                heatmaps = preds_[f-n_seed_plt].cpu().detach().numpy()
                struct_type = struct_types[n_rows-row-2] if isinstance(struct_types, list) else struct_types
                img = plot_structured_preds(a, heatmaps, struct_type, image=img)
                if save_frames and img is not None:
                    plt.imsave(f"{seq_dir}/hier_{n_rows-1-row}/img_{f_str}.png", img)
    plt.tight_layout()
    for ax in fig.axes:
        ax.axis("off")
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    fig.clear()
    return


def visualize_mixed_seq(inputs, preds, pred_fnums, struct_types,
                        seed_stride=4, pred_stride=2, savepath=None):
    assert len(struct_types) == 2
    n_seed_plt = ceil(len(inputs) / seed_stride)
    n_pred_plt = ceil(len(preds) / pred_stride)
    n_cols = n_seed_plt + n_pred_plt
    fig, ax = plt.subplots(1, n_cols)
    fig.set_size_inches(3*n_cols, 3)

    for f in range(n_cols):
        a = ax[f]
        struct_type = struct_types[0] if f < n_seed_plt else struct_types[1]
        if struct_type == "IMAGE":
            fnum = f*seed_stride
            img = inputs[fnum]
            img = img.permute(1, 2, 0).cpu().detach()
            img = torch.squeeze(img, dim=2)
            img = img.clamp(0, 1).numpy()
            a.set_title(f"Frame {fnum}", fontsize=25)
            a.imshow(img)
        else:
            fnum = pred_fnums[(f-n_seed_plt)*pred_stride]
            heatmaps = preds[(f-n_seed_plt)*pred_stride].cpu().detach().numpy()
            a.set_title(f"Frame {fnum}", fontsize=25)
            plot_structured_preds(a, heatmaps, struct_type)
    plt.tight_layout()
    for ax in fig.axes:
        ax.axis("off")
    if savepath is not None:
        plt.savefig(savepath)
    else:
        plt.show()
    fig.clear()
    return


def label2rgb(mask, image, num_classes, colors, bg_label=0, bg_color=None, img_alpha=0.5):
    img = image
    for cls in range(num_classes):
        if cls == bg_label and bg_color is None:
            continue
        color = colors[cls] if cls != bg_label else bg_color
        color_rgb = name_to_rgb(color)
        for c in range(3):
            img[mask == cls, c] = (img[mask == cls, c] * img_alpha + (color_rgb[c] / 255.) * (1. - img_alpha))
    return img


def plot_structured_preds(a, heatmaps, struct_type, image=None):
    COLORS = ["blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
              "brown", "pink", "darkorange", "goldenrod", "forestgreen", "springgreen",
              "aqua", "royalblue", "navy", "darkviolet", "plum", "magenta", "slategray",
              "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
              "darkcyan", "sandybrown"]
    MIN_CONF = 0.03
    MIN_PEAK = 0.03
    num_kpoints, h, w = heatmaps.shape
    img = None
    if struct_type == "BLOB_HEATMAP":
        img = np.sum(heatmaps, axis=0)
        img = np.clip(img, 0.0, 1.0)
        a.imshow(img, alpha=0.4)

    elif struct_type == "KEYPOINTS":
        for i, hmap in enumerate(heatmaps):
            kpos = np.unravel_index(np.argmax(hmap, axis=None), hmap.shape)
            if hmap[kpos] >= MIN_CONF:
                kpoint = plt.Circle(kpos[::-1], 1, color=COLORS[i % len(COLORS)])
                a.add_patch(kpoint)

    elif struct_type == "SEGMENTATION_MAPS":
        hmap = np.argmax(heatmaps, axis=0)
        img = label2rgb(mask=hmap, image=image, num_classes=heatmaps.shape[0],
                        colors=COLORS, bg_color="seashell", img_alpha=0.)
        a.imshow(img)

    elif struct_type in ["KEYPOINT_BLOBS", "MULTIPEAK_HEATMAPS"]:
        if struct_type == "KEYPOINT_BLOBS":
            kpoints = [(0., 0.)] * num_kpoints
            for i, hmap in enumerate(heatmaps):
                kpos = np.unravel_index(np.argmax(hmap, axis=None), hmap.shape)
                if hmap[kpos] >= MIN_CONF:
                    x, y = kpos[::-1]
                    kpoints[i] = (x/w, y/h)
            blob_size = 1.0 if len(heatmaps) > 1 else 4.0
            hmaps = HeatmapGenerator(shape=(h, w), num_kpoints=num_kpoints, sigma=blob_size)(kpoints)
        else:
            hmaps = heatmaps
            for i, hmap in enumerate(hmaps):
                hmaps[i] = maximum_filter(hmap, size=2, mode='constant', cval=0.0)
                hmaps[i][hmaps[i] < MIN_PEAK] = 0.0  # filter-out non-significant peaks
                max_peak_val = np.max(hmaps[i])  # amplify remaining peaks
                scale = 1./max_peak_val if max_peak_val != 0. else 0.
                hmaps[i] *= scale
                hmaps[i] = np.clip(hmaps[i], 0.0, 1.0)

        hmaps = np.repeat(np.expand_dims(hmaps, axis=1), 3, axis=1)
        # color the keypoint-blobs
        for i in range(num_kpoints):
            for c in range(3):
                color_rgb = name_to_rgb(COLORS[i % len(COLORS)])
                hmaps[i][c] *= color_rgb[c] / 255.
        # create white background
        img = np.sum(hmaps, axis=0)
        img = np.clip(img, 0.0, 1.0)
        img = invert_colors(img).transpose(1, 2, 0)
        if struct_type == "MULTIPEAK_HEATMAPS":
            img = add_border(img, color="white", pad=4, extend=False)  # hack!
        img = add_border(img, color="black", pad=1)
        a.imshow(img)
    else:
        assert False, f"Invalid structured prediction type: {struct_type}"
    return img


def invert_colors(img):
    assert len(img.shape) == 3 and img.shape[0] == 3
    color_obj_0 = [0.961, 0.29, 0.05]  # sun
    color_obj_1 = [0.05, 0.09, 0.961]  # earth
    color_obj_2 = [0.10, 0.895, 0.219]  # moon
    image = np.ones(img.shape)
    for c in range(3):
        image[c] -= (1.0 - color_obj_0[c]) * img[0]
        image[c] -= (1.0 - color_obj_1[c]) * img[1]
        image[c] -= (1.0 - color_obj_2[c]) * img[2]
    image = np.clip(image, 0.0, 1.0)
    return image


def save_gif_new(frames, savepath, n_seed=4):
    """ """
    with imageio.get_writer(savepath, mode='I') as writer:
        for i, frame in enumerate(frames):
            up_frame = F.upsample(frame.unsqueeze(0), scale_factor=2)[0]  # to make it larger
            up_frame = up_frame.permute(1, 2, 0).cpu().detach().clamp(0, 1)
            disp_frame = add_border(up_frame, color="green") if i < n_seed else add_border(up_frame, color="red")
            writer.append_data(disp_frame)


def visualize_seed_preds(gt_seq, preds, savepath=None):
    """ """
    seed_ids = torch.Tensor([0, 5, 11, 16]).long()
    pred_ids = torch.Tensor([17, 18, 19, 20, 21]).long()
    gt = torch.cat([gt_seq[seed_ids], gt_seq[pred_ids]], dim=0).cpu().detach()
    preds = preds.cpu().detach()

    border = 2
    # breakpoint()
    _, c, img_H, img_W = gt.shape
    hb, wb = img_H + border, img_W + border  # img sizes with borders
    W_gt = (wb * 9)  # width of ground truth
    W_seed = (wb * 4)  # width of white space in bottom row

    large_img_context = np.ones((c, hb, W_gt))
    for n_frame, frame in enumerate(gt):
        w_start = n_frame * wb
        if n_frame >= 4:
            w_start = w_start + 1
        large_img_context[:, :img_H, w_start:w_start+img_W] = gt[n_frame]

    large_img_pred = np.ones((c, img_H, W_gt))
    for n_frame, frame in enumerate(preds):
        w_start = n_frame * wb + W_seed + 1
        large_img_pred[:, :, w_start:w_start+img_W] = preds[n_frame]

    large_img = np.concatenate([large_img_context, large_img_pred], axis=-2)
    large_img = np.transpose(large_img * 255, (1, 2, 0)).astype(np.uint8)
    cv2.imwrite(savepath, cv2.cvtColor(large_img, cv2.COLOR_RGB2BGR))
    return


#
