from IPython import embed
import time

import jax
from jax import lax, numpy as jnp

import numpy as np

from skimage import io
from skimage.filters import gaussian
from skimage.restoration import denoise_nl_means, estimate_sigma


def _ixs(y_ixs, x_ixs):
    return jnp.meshgrid(x_ixs, y_ixs)


def _vmap_2d(f, y_ixs, x_ixs):
    _x, _y = _ixs(y_ixs, x_ixs)
    return jax.vmap(jax.vmap(f))(_y, _x)


def _load_image(x):
    return jnp.float32(io.imread(x)) / 255


def _psnr(img1, img2):
    return 20 * jnp.log10(img1.max() /
                          jnp.sqrt(jnp.mean((img1 - img2) ** 2)))


def _rms(img1, img2):
    return jnp.sqrt(jnp.mean((img1 - img2) ** 2))


@jax.partial(jax.jit, static_argnums=(1, 2))
def nlm(img, search_window_radius, filter_radius, h, sigma):
    _h, _w = img.shape
    pad = search_window_radius
    img_pad = jnp.pad(img, pad)

    filter_length = 2*filter_radius + 1
    search_window_length = 2*search_window_radius + 1
    win_y_ixs = win_x_ixs = jnp.arange(
        search_window_length - filter_length + 1)

    filter_size = (filter_length, filter_length)

    def compute(y, x):
        # (y + pad, x + pad) are the center of the current neighborhood
        win_center_y = y + pad
        win_center_x = x + pad

        center_patch = jax.lax.dynamic_slice(
            img_pad, (win_center_y-filter_radius, win_center_x-filter_radius), filter_size)

        # Iterate over all patches in this neighborhood
        def _compare(center):
            center_y, center_x = center
            patch = lax.dynamic_slice(
                img_pad, (center_y - filter_radius, center_x - filter_radius), filter_size)
            d2 = jnp.sum((patch - center_patch) ** 2) / (filter_length ** 2)
            weight = jnp.exp(-(jnp.maximum(d2 - 2 * (sigma**2), 0) / (h**2)))
            intensity = img_pad[center_y, center_x]
            return (weight, intensity)

        def compare(patch_y, patch_x):
            patch_center_y = patch_y + filter_radius
            patch_center_x = patch_x + filter_radius
            # Skip if patch is out of image boundaries or this is the center patch
            skip = lax.lt(patch_center_y, pad) | lax.ge(patch_center_y, _h +
                                                        pad) | lax.lt(patch_center_x, pad) | lax.ge(patch_center_x, _w+pad) | (lax.eq(patch_center_y, win_center_y) & lax.eq(patch_center_x, win_center_x))
            return lax.cond(skip, lambda _: (0., 0.), _compare, (patch_center_y, patch_center_x))

        weights, intensities = _vmap_2d(compare, y + win_y_ixs, x + win_x_ixs)

        # Use max weight for the center patch
        max_weight = jnp.max(weights)
        total_weight = jnp.sum(weights) + max_weight
        pixel = (jnp.sum((weights * intensities)) +
                 max_weight * img_pad[win_center_y, win_center_x]) / total_weight
        # embed()
        return pixel

    # embed()
    h_ixs = jnp.arange(_h)
    w_ixs = jnp.arange(_w)
    out = _vmap_2d(compute, h_ixs, w_ixs)
    return out


if __name__ == "__main__":
    clean = _load_image("images/night.png")
    noisy = _load_image("images/night_downsampled_noisy_sigma_0.0781.png")

    search_window_radius = 7
    filter_radius = 1

    sigma_est = jnp.mean(jnp.array(estimate_sigma(noisy, multichannel=True)))
    print(f"Estimated noise standard deviation: {sigma_est}")

    k = 0.75
    h = k * sigma_est

    # nlm(noisy[..., 0], search_window_radius, filter_radius, h, sigma_est)
    def _nlm(img):
        return nlm(img, search_window_radius,
                   filter_radius, h, sigma_est)

    def _denoise(img):
        return jax.vmap(_nlm, in_axes=-1, out_axes=-
                        1)(img)

    start = time.time()
    denoised = _denoise(noisy)
    print(f"Denoising took {time.time() - start} seconds")
    psnr = _psnr(denoised, clean)
    print(f"PSNR = {psnr}, saving to images/denoised.png")
    io.imsave("images/denoised.png", denoised)

    # Feel free to differentiate wrt any of the inputs.
    print(jax.grad(lambda x1, x2: _rms(_denoise(x1), x2))(noisy, clean))
    embed()
