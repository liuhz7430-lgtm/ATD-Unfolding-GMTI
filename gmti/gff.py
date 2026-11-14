"""MiniSAR GFF 1.8 complex-image reader; range x azimuth ordering."""
from pathlib import Path
import struct
import numpy as np

def read_gff(path):
    path = Path(path)
    with path.open('rb') as stream:
        header = stream.read(2048)
    if len(header) < 2048 or header[:7] != b'GSATIMG':
        raise ValueError(f'{path}: invalid/truncated GFF header')
    endian = '<' if header[54:56] == b'\0\0' else '>'

    def field(offset, fmt):
        return struct.unpack_from(endian + fmt, header, offset)[0]

    minor, major = field(8, 'H'), field(10, 'H')
    if (major, minor) != (1, 8):
        raise ValueError(f'{path}: this inspector supports GFF 1.8 only')
    length = field(12, 'I')
    bpp_value = field(56, 'f')
    if bpp_value not in (1., 4., 8.):
        raise ValueError(f'{path}: unsupported bytes-per-pixel {bpp_value}')
    bpp = int(bpp_value)
    frames, kind, row_major, nr, na = [field(o, 'I') for o in (60, 64, 68, 72, 76)]
    if frames != 1 or row_major not in (0, 1) or min(nr, na) < 1:
        raise ValueError(f'{path}: unsupported image layout')
    expected = length + nr * na * bpp
    if length < 2048 or path.stat().st_size != expected:
        raise ValueError(f'{path}: file size does not match header ({expected})')
    first, second = (nr, na) if row_major else (na, nr)
    if kind == 1 and bpp in (4, 8):
        # Phase first, then unsigned magnitude; these are NOT I/Q integers.
        bits = bpp * 4
        pairs = np.memmap(path, dtype=endian + ('u2' if bpp == 4 else 'u4'),
                          mode='r', offset=length, shape=(second, first, 2))
        phase = pairs[..., 0].astype(np.float64) * (2 * np.pi / 2**bits)
        image = (pairs[..., 1] * np.exp(1j * phase)).astype(np.complex64)
        encoding = f'uint{bits} phase + uint{bits} magnitude'
    elif kind == 2 and bpp == 8:
        pairs = np.memmap(path, dtype=endian + 'f4', mode='r', offset=length,
                          shape=(second, first, 2))
        image = (pairs[..., 0] + 1j * pairs[..., 1]).astype(np.complex64)
        encoding = 'float32 real + float32 imaginary'
    else:
        raise ValueError(f'{path}: unsupported image type {kind}, bpp {bpp}')
    if row_major:
        image = image.T
    if not np.isfinite(image).all():
        raise ValueError(f'{path}: nonfinite image samples')
    metadata = {
        'file': path.name, 'version': f'{major}.{minor}', 'frames': frames,
        'range_pixels': nr, 'azimuth_pixels': na, 'bytes_per_pixel': bpp,
        'encoding': encoding, 'row_major': row_major, 'header_bytes': length,
        'image_plane_raw': field(296, 'I'),
        'range_pixel_m': field(300, 'I') / 2**16,
        'azimuth_pixel_m': field(304, 'I') / 2**16,
        'azimuth_resolution_m': field(692, 'I') / 2**16,
        'range_resolution_m': field(696, 'I') / 2**16,
        'grazing_angle_deg': field(636, 'I') / 2**23,
        'platform_speed_m_s': field(668, 'I') / 2**16,
        'nominal_center_frequency_raw': field(856, 'f'),
        'wavelength_m': field(872, 'f'),
        'comment': header[130:130 + field(128, 'H')].decode('ascii', errors='replace').strip('\0'),
        'phase_history_name': header[366:366 + field(364, 'H')].decode('ascii', errors='replace').strip('\0\n'),
        'nonzero_fraction': float(np.mean(np.abs(image) > 0)),
    }
    return image, metadata

def display_db(image):
    amplitude = np.abs(image)
    reference = max(float(np.quantile(amplitude, .995)), 1e-12)
    return 20 * np.log10(np.maximum(amplitude, 1e-12) / reference)
