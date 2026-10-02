"""Convert legacy Type-3 figure fonts to vector outlines without rerunning analyses.

Original PDFs are preserved. Rendering comparisons reject material changes;
images are not downsampled or given lossy recompression. Requires gs/pdftoppm.
"""
from pathlib import Path
import hashlib
from collections import Counter
import json
import shutil
import subprocess
import tempfile
import numpy as np
from PIL import Image, ImageFilter

PAPER = Path(__file__).resolve().parents[1]
REVIEW = PAPER/'review/formal_requirements_2026-10-02'


def render(path, dest):
    subprocess.run(['pdftoppm', '-singlefile', '-scale-to', '1600', '-png', str(path), str(dest)],
                   check=True, capture_output=True)
    return np.asarray(Image.open(dest.with_suffix('.png')).convert('RGB'), dtype=np.int16)


def run():
    records = []
    for source in sorted((PAPER/'figures').glob('*.pdf')):
        backup = REVIEW/'before'/source.relative_to(PAPER)
        original = backup if backup.exists() else source
        listing = subprocess.run(['pdffonts', str(original)], check=True, capture_output=True, text=True).stdout
        if 'Type 3' not in listing:
            continue
        backup.parent.mkdir(parents=True, exist_ok=True)
        if not backup.exists():
            shutil.copy2(source, backup)
        with tempfile.TemporaryDirectory(prefix='sij_font_outline_') as temp:
            tmp = Path(temp)
            converted = tmp/'outlined.pdf'
            command = ['gs', '-q', '-dSAFER', '-dBATCH', '-dNOPAUSE', '-sDEVICE=pdfwrite',
                       '-dCompatibilityLevel=1.7', '-dNoOutputFonts', '-dAutoRotatePages=/None',
                       '-dDownsampleColorImages=false', '-dDownsampleGrayImages=false',
                       '-dDownsampleMonoImages=false', '-dAutoFilterColorImages=false',
                       '-dColorImageFilter=/FlateEncode', '-dAutoFilterGrayImages=false',
                       '-dGrayImageFilter=/FlateEncode', '-dMonoImageFilter=/FlateEncode',
                       '-sOutputFile='+str(converted), str(original)]
            subprocess.run(command, check=True, capture_output=True)
            fonts = subprocess.run(['pdffonts', str(converted)], check=True, capture_output=True, text=True).stdout.splitlines()[2:]
            assert not fonts, (source, fonts)
            a, b = render(original, tmp/'old'), render(converted, tmp/'new')
            assert a.shape == b.shape, (source, a.shape, b.shape)
            difference = np.abs(a-b)
            mean = float(difference.mean())
            changed = float((difference.max(axis=2)>5).mean())
            blurred = [np.asarray(Image.fromarray(x.astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)), dtype=np.int16) for x in (a,b)]
            blurred_mean = float(np.abs(blurred[0]-blurred[1]).mean())
            # Outline conversion changes low-resolution glyph antialiasing;
            # separately require exact preservation of decoded bitmap data.
            bitmaps = []
            for prefix, pdf in [('original_image', original), ('outlined_image', converted)]:
                subprocess.run(['pdfimages', '-png', str(pdf), str(tmp/prefix)], check=True, capture_output=True)
                counts = Counter()
                for bitmap in tmp.glob(prefix+'-*.png'):
                    im = Image.open(bitmap).convert('RGB')
                    counts[(im.size, hashlib.sha256(im.tobytes()).hexdigest())] += 1
                bitmaps.append(counts)
            assert bitmaps[0] == bitmaps[1], (source, 'bitmap data changed')
            # The two-pixel blur discounts glyph-edge antialiasing; the raw
            # comparison and exact decoded-image check remain additional gates.
            assert mean < 5.0 and changed < .05 and blurred_mean < 1.5, (source, mean, changed, blurred_mean)
            record = dict(file=str(source.relative_to(PAPER)),
                          original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                          outlined_sha256=hashlib.sha256(converted.read_bytes()).hexdigest(),
                          decoded_bitmap_data_identical=True,
                          decoded_bitmap_count=sum(bitmaps[0].values()),
                          render_mean_absolute_rgb_difference=mean,
                          render_blurred_mean_absolute_rgb_difference=blurred_mean,
                          fraction_pixels_with_channel_difference_over_5=changed,
                          conversion='Ghostscript pdfwrite NoOutputFonts; vector outlines; no image downsampling; lossless image filters')
            shutil.copy2(converted, source)
            records.append(record)
            (REVIEW/'figure_font_normalization.json').write_text(json.dumps(records, indent=2))
            print(source.name, 'render difference', mean, flush=True)
    if records:
        REVIEW.mkdir(exist_ok=True)
        (REVIEW/'figure_font_normalization.json').write_text(json.dumps(records, indent=2))
    print('Converted', len(records), 'legacy figure PDFs.', flush=True)


if __name__ == '__main__':
    run()
