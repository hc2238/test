#!/usr/bin/env python3
"""comparekit - forensic comparison of comic page images extracted from a PDF vs a CBZ/CBR.

Answers: "Which file holds the better version of this page's art, and was one made from the other?"

Subcommands (run `comparekit.py <cmd> -h` for options):
  inspect      format, size, mode, JPEG quant-table sums, chroma subsampling, grey levels
  register     content scale/offset between a PDF image and a CBZ page (SIFT + tile refinement)
  fingerprint  is the PDF JPEG's quantization lattice / block grid present in the CBZ?
  reencode     re-save the PDF crop with the CBZ's exact JPEG tables and compare
  chroma       does an image's full-resolution colour carry real detail (force 4:2:0 test)?
  dupes        detect nearest-neighbour upscaling (duplicated rows/columns)
  blocks       JPEG block-edge strength of an image on a given grid
  coherence    signal-vs-noise detail comparison per frequency band (the main detail test)
  bw           is a greyscale/antialiased CBZ just a downscale of a 1-bit PDF image?

Conventions: A = PDF-side image, B = CBZ-side image. Offsets: B(x, y) == A(x + dx, y + dy).
Requires: numpy, pillow, opencv-python-headless (scipy not needed).
"""
import argparse, io, json, sys
import numpy as np
import cv2
from PIL import Image

# ----------------------------------------------------------------------------- helpers

def load(path, mode='RGB'):
    im = Image.open(path)
    if im.mode == 'CMYK' and mode != 'CMYK':
        im = im.convert('RGB')   # naive CMYK->RGB: NOT colour-managed (see PLAYBOOK pitfalls)
    return np.array(im.convert(mode)).astype(np.float32)

def gray8(a):
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)

def phase_shift(a, b, win):
    """Return (dx, dy, response) such that b(x) ~= a(x - dx), i.e. b is a shifted RIGHT by dx.
    NOTE: cv2.phaseCorrelate can modify its inputs in place - always pass copies."""
    (dx, dy), r = cv2.phaseCorrelate(np.ascontiguousarray(a, np.float32).copy(),
                                     np.ascontiguousarray(b, np.float32).copy(), win)
    return dx, dy, r

def sift_affine(A, B, partial=True, max_side=2400):
    """Feature-match A and B (any scales). Returns 2x3 matrix mapping B coords -> A coords."""
    ga, gb = gray8(A), gray8(B)
    fa = min(1.0, max_side / max(ga.shape)); fb = min(1.0, max_side / max(gb.shape))
    sa = cv2.resize(ga, None, fx=fa, fy=fa, interpolation=cv2.INTER_AREA) if fa < 1 else ga
    sb = cv2.resize(gb, None, fx=fb, fy=fb, interpolation=cv2.INTER_AREA) if fb < 1 else gb
    sift = cv2.SIFT_create(10000)
    ka, da = sift.detectAndCompute(sa, None); kb, db = sift.detectAndCompute(sb, None)
    m = cv2.BFMatcher().knnMatch(db, da, k=2)
    good = [x for x, y in (p for p in m if len(p) == 2) if x.distance < 0.75 * y.distance]
    if len(good) < 20:
        raise SystemExit('too few feature matches - are these the same page?')
    src = np.float32([kb[g.queryIdx].pt for g in good]) / fb
    dst = np.float32([ka[g.trainIdx].pt for g in good]) / fa
    est = cv2.estimateAffinePartial2D if partial else cv2.estimateAffine2D
    M, inl = est(src, dst, ransacReprojThreshold=3, maxIters=20000, confidence=0.999)
    if M is None or inl.sum() < 20:
        raise SystemExit('registration failed - are these the same page?')
    return M, int(inl.sum()), len(good)

def refine_same_scale(A, B, M0, tile=128, step=64):
    """Refine a near-1:1 B->A mapping with per-tile phase correlation; fits a full affine."""
    ga, gb = gray8(A).astype(np.float32), gray8(B).astype(np.float32)
    win = cv2.createHanningWindow((tile, tile), cv2.CV_32F); h = tile // 2
    pb, pa = [], []
    for y in range(h + 8, gb.shape[0] - h - 8, step):
        for x in range(h + 8, gb.shape[1] - h - 8, step):
            bb = gb[y - h:y + h, x - h:x + h]
            if bb.std() < 8: continue
            cx, cy = M0 @ np.array([x, y, 1.0]); ix, iy = int(round(cx)), int(round(cy))
            if iy - h < 0 or ix - h < 0 or iy + h > ga.shape[0] or ix + h > ga.shape[1]: continue
            aa = ga[iy - h:iy + h, ix - h:ix + h]
            dx, dy, r = phase_shift(aa, bb, win)          # bb(x) ~= aa(x - dx)
            if r > 0.5 and abs(dx) < 2 and abs(dy) < 2:
                pb.append((x, y)); pa.append((ix - dx, iy - dy))   # B point x sits at A point ix - dx
    if len(pb) < 20: return M0, 0, 0
    M, inl = cv2.estimateAffine2D(np.float32(pb), np.float32(pa), ransacReprojThreshold=0.3)
    return M, int(inl.sum()), len(pb)

def describe(M):
    s = np.hypot(M[0, 0], M[1, 0])
    return dict(a_px_per_b_px=round(float(s), 5), sx=round(float(M[0, 0]), 5), sy=round(float(M[1, 1]), 5),
                shear=(round(float(M[0, 1]), 6), round(float(M[1, 0]), 6)),
                dx=round(float(M[0, 2]), 3), dy=round(float(M[1, 2]), 3))

def overlap(A, B, dx, dy):
    """Crops of A and B covering the same content, for integer offset B(x,y)=A(x+dx,y+dy)."""
    x0, y0 = max(0, -dx), max(0, -dy)
    W = min(B.shape[1], A.shape[1] - dx); H = min(B.shape[0], A.shape[0] - dy)
    return A[y0 + dy:H + dy, x0 + dx:W + dx], B[y0:H, x0:W]

def block_profile(Y, per):
    gx = np.abs(np.diff(Y, axis=1)).mean(0); gy = np.abs(np.diff(Y, axis=0)).mean(1)
    r = np.array([(gx[o::per].mean() + gy[o::per].mean()) / 2 for o in range(per)])
    return r / np.median(r)   # index i = edge between pixel i and i+1 (mod per)

# ----------------------------------------------------------------------------- commands

def cmd_inspect(a):
    for p in a.files:
        try: im = Image.open(p)
        except Exception as e: print(f'{p}: not readable by Pillow ({e})'); continue
        rgb = np.array(im.convert('RGB')).astype(int)
        chroma = np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean()
        info = f'{p}: {im.format} {im.mode} {im.size[0]}x{im.size[1]} grey-levels={len(np.unique(np.array(im.convert("L"))))} chroma={chroma:.2f}'
        if im.format == 'JPEG':
            q = {k: sum(v) for k, v in im.quantization.items()}
            lay = getattr(im, 'layer', None)
            sub = '4:4:4' if lay and all(h == 1 and v == 1 for _, h, v, _ in lay) else ('4:2:0' if lay and lay[0][1] == 2 and lay[0][2] == 2 else str(lay))
            info += f' quant-sums={list(q.values())} subsampling={sub} adobe={im.info.get("adobe_transform")} icc={bool(im.info.get("icc_profile"))}'
        print(info)
    if any(p.lower().endswith('.jxl') for p in a.files):
        for p in a.files:
            if p.lower().endswith('.jxl'):
                d = open(p, 'rb').read(); print(f'{p}: JPEG-XL, jbrd box (lossless JPEG reconstruction data) present: {b"jbrd" in d}')

def cmd_register(a):
    A, B = load(a.pdf), load(a.cbz)
    if a.scale_pdf != 1.0:
        A = cv2.resize(A, None, fx=a.scale_pdf, fy=a.scale_pdf, interpolation=cv2.INTER_AREA)
    M, ni, ng = sift_affine(A, B)
    out = dict(sift=describe(M), sift_inliers=f'{ni}/{ng}')
    if abs(np.hypot(M[0, 0], M[1, 0]) - 1) < 0.01:
        M2, ni2, n2 = refine_same_scale(A, B, M)
        out['refined'] = describe(M2); out['refined_inliers'] = f'{ni2}/{n2}'
        M = M2
    out['note'] = 'mapping is CBZ(x,y) -> PDF(x*sx + dx, y*sy + dy); a_px_per_b_px is the content-resolution ratio'
    print(json.dumps(out, indent=1))
    if a.save: np.save(a.save, M)

def lattice_distance(Y, ox, oy, Q, n=15000, seed=0):
    rng = np.random.default_rng(seed); H, W = Y.shape
    ys = np.arange(oy, H - 8, 8); xs = np.arange(ox, W - 8, 8); out = []
    for _ in range(n):
        y = rng.choice(ys); x = rng.choice(xs); b = Y[y:y + 8, x:x + 8]
        if b.std() < 2: continue
        c = cv2.dct(b) / Q; out.append(np.abs(c - np.round(c)).ravel()[1:])
    return float(np.concatenate(out).mean())

def cmd_fingerprint(a):
    """PDF must be JPEG. Integer offsets only (1:1 scale). 0 = coefficients sit on the PDF's lattice, ~0.15-0.25 = no trace."""
    imA = Image.open(a.pdf)
    if imA.format != 'JPEG': raise SystemExit('PDF image must be a JPEG for the lattice test')
    QA = np.array(imA.quantization[0], float).reshape(8, 8)   # Pillow returns natural order (verified)
    YA = load(a.pdf, 'YCbCr')[..., 0] - 128; YB = load(a.cbz, 'YCbCr')[..., 0] - 128
    Ac, Bc = overlap(YA, YB, a.dx, a.dy)
    # PDF block grid: PDF col 8k <-> overlap col 8k - (dx + x0) ... recompute in overlap coordinates
    x0, y0 = max(0, -a.dx), max(0, -a.dy)
    ox, oy = (-(a.dx + x0)) % 8, (-(a.dy + y0)) % 8   # first full PDF block inside the overlap crop
    on = lattice_distance(Bc, ox, oy, QA)
    offs = [lattice_distance(Bc, (ox + 3) % 8, (oy + 5) % 8, QA), lattice_distance(Bc, (ox + 5) % 8, (oy + 2) % 8, QA)]
    self_ = lattice_distance(YA, 0, 0, QA)
    print(f'PDF JPEG lattice distance: PDF itself {self_:.3f} | CBZ on PDF grid {on:.3f} | CBZ off-grid controls {offs[0]:.3f}, {offs[1]:.3f}')
    print('verdict:', 'CBZ carries the PDF JPEG fingerprint -> CBZ was made from this PDF image' if on < 0.75 * min(offs)
          else 'no clear fingerprint (CBZ from another source, OR resampled/smoothed, OR recompressed more coarsely than the PDF)')
    pa, pb = block_profile(YA, 8), block_profile(Bc, 8)
    print(f'block-edge strength on PDF grid: PDF {pa[7]:.2f}  CBZ {pb[(7 - (a.dx + x0)) % 8]:.2f}  (1.00 = no visible block edges)')

def cmd_reencode(a):
    A = Image.open(a.pdf).convert('RGB'); Bim = Image.open(a.cbz)
    if Bim.format != 'JPEG': raise SystemExit('CBZ image must be JPEG')
    B = np.array(Bim.convert('RGB')).astype(np.float32)
    crop = A.crop((a.dx, a.dy, a.dx + B.shape[1], a.dy + B.shape[0]))
    lay = getattr(Bim, 'layer', None); sub = 0 if lay and all(h == 1 and v == 1 for _, h, v, _ in lay) else 2
    buf = io.BytesIO(); crop.save(buf, 'JPEG', qtables=Bim.quantization, subsampling=sub)
    R = np.array(Image.open(buf).convert('RGB')).astype(np.float32); C = np.array(crop).astype(np.float32)
    g = lambda x: gray8(x).astype(np.float32)
    bad = cv2.dilate((np.abs(cv2.GaussianBlur(g(C), (0, 0), 2) - cv2.GaussianBlur(g(B), (0, 0), 2)) > 12).astype(np.uint8), np.ones((25, 25))) > 0
    m = ~bad
    print(f'shared art {100 * m.mean():.0f}% | CBZ vs PDF crop mean {np.abs(C - B)[m].mean():.2f} | one re-save at CBZ settings adds {np.abs(R - C)[m].mean():.2f} | '
          f'CBZ vs my re-save {np.abs(R - B)[m].mean():.2f} (within 1 level: {100 * (np.abs(R - B).max(2) <= 1)[m].mean():.1f}%)')
    print('interpretation: CBZ~=my re-save (>90% within 1) proves derivation with the same encoder; otherwise CBZ-vs-PDF ~= one-save error is merely consistent with it')

def cmd_chroma(a):
    for p in a.files:
        im = Image.open(p); im = im.convert('RGB') if im.mode == 'CMYK' else im
        ycc = np.array(im.convert('YCbCr')).astype(np.float32)
        toRGB = lambda y: np.array(Image.fromarray(np.clip(y, 0, 255).round().astype(np.uint8), 'YCbCr').convert('RGB')).astype(np.float32)
        base = toRGB(ycc); res = []
        for off in (0, 1):
            y2 = ycc.copy()
            for ch in (1, 2):
                c = y2[off:, off:, ch]; h, w = c.shape[0] // 2 * 2, c.shape[1] // 2 * 2
                sm = c[:h, :w].reshape(h // 2, 2, w // 2, 2).mean((1, 3))
                y2[off:off + h, off:off + w, ch] = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
            d = np.abs(toRGB(y2) - base)[4:-4, 4:-4]
            res.append(f'grid@{off}: mean {d.mean():.2f}, px>20 levels {100 * (d.max(2) > 20).mean():.2f}%')
        print(f'{p}: forcing half-res colour changes -> ' + ' | '.join(res))
    print('interpretation: mean < ~1 level and px>20 < ~1% => full-res colour holds ~no real colour detail (often only colour/black line edges)')

def cmd_dupes(a):
    B = load(a.image).astype(int)
    if a.box: x0, y0, x1, y1 = a.box; B = B[y0:y1, x0:x1]
    cd = (np.abs(np.diff(B, axis=1)).sum((0, 2)) == 0); rd = (np.abs(np.diff(B, axis=0)).sum((1, 2)) == 0)
    print(f'columns identical to left neighbour: {cd.sum()}/{len(cd)} | rows identical to row above: {rd.sum()}/{len(rd)}')
    print(f'unique grid after collapsing duplicates: {(~cd).sum() + 1} x {(~rd).sum() + 1}  (large duplicate fractions => nearest-neighbour upscale of that size)')

def cmd_blocks(a):
    Y = load(a.image, 'YCbCr')
    for ch, nm, per in ((0, 'luma', 8), (1, 'Cb', 16), (2, 'Cr', 16)):
        p = block_profile(Y[..., ch], per); i = (per - 1 + a.offset) % per
        print(f'{nm}: block-edge strength on grid offset {a.offset}: {p[i]:.2f}   (profile max {p.max():.2f} at {int(p.argmax())}; 1.00 = invisible)')

# --- coherence (signal-vs-noise) test ---------------------------------------
BANDS = [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]

def coherence(A, B, M, T=64, min_corr=0.95):
    f = np.fft.fftfreq(T); FX, FY = np.meshgrid(f, f); R = np.sqrt(FX ** 2 + FY ** 2) * 2
    W2 = np.outer(np.hanning(T), np.hanning(T)).astype(np.float32); WIN = cv2.createHanningWindow((T, T), cv2.CV_32F)
    acc = {k: dict(xx=np.zeros((T, T)), yy=np.zeros((T, T)), xy=np.zeros((T, T), complex)) for k in ('Y', 'C')}
    n = 0
    for by in range(8, B.shape[0] - T - 8, T):
        for bx in range(8, B.shape[1] - T - 8, T):
            cx, cy = M @ np.array([bx + T / 2, by + T / 2, 1.0]); ax, ay = int(round(cx - T / 2)), int(round(cy - T / 2))
            if ax < 0 or ay < 0 or ax + T > A.shape[1] or ay + T > A.shape[0]: continue
            ta, tb = A[ay:ay + T, ax:ax + T], B[by:by + T, bx:bx + T]
            if tb[..., 0].std() < 4: continue
            dx, dy, r = phase_shift(ta[..., 0], tb[..., 0], WIN)
            if r < 0.3 or abs(dx) > 1.5 or abs(dy) > 1.5: continue
            la = cv2.GaussianBlur(ta[..., 0], (0, 0), 2).ravel()
            lb = cv2.GaussianBlur(np.roll(tb[..., 0], (-int(round(dy)), -int(round(dx))), (0, 1)), (0, 0), 2).ravel()
            if np.corrcoef(la, lb)[0, 1] < min_corr: continue          # lettering / content differences
            ramp = np.exp(2j * np.pi * (FX * dx + FY * dy))              # undo sub-pixel shift (b(x)=a(x-d))
            for key, chs in (('Y', [0]), ('C', [1, 2])):
                for ch in chs:
                    Fa = np.fft.fft2((ta[..., ch] - ta[..., ch].mean()) * W2)
                    Fb = np.fft.fft2((tb[..., ch] - tb[..., ch].mean()) * W2) * ramp
                    acc[key]['xx'] += np.abs(Fa) ** 2; acc[key]['yy'] += np.abs(Fb) ** 2; acc[key]['xy'] += Fa * np.conj(Fb)
            n += 1
    res = {}
    for key, N in (('Y', n), ('C', 2 * n)):
        xx, yy, xy = acc[key]['xx'], acc[key]['yy'], acc[key]['xy']
        if n == 0 or xx.sum() < 1e-6 or yy.sum() < 1e-6: res[key] = None; continue
        rows = []
        for lo, hi in BANDS:
            b = (R > lo) & (R <= hi)
            c2 = np.maximum(np.abs(xy[b]) ** 2 - xx[b] * yy[b] / N, 0)   # finite-sample bias correction
            rows.append(dict(raw=yy[b].sum() / xx[b].sum(), coh=float((c2 / (xx[b] * yy[b])).mean()),
                             pdf_lb=(c2 / yy[b]).sum() / yy[b].sum(),    # lower bound PDF_signal / CBZ_signal
                             cbz_lb=(c2 / xx[b]).sum() / xx[b].sum()))   # lower bound CBZ_signal / PDF_signal
        res[key] = rows
    return n, res

def cmd_coherence(a):
    A, B = load(a.pdf, 'YCbCr'), load(a.cbz, 'YCbCr')
    if a.scale_pdf != 1.0:   # bring a higher-res PDF image to CBZ scale (INTER_AREA: biased AGAINST the PDF = conservative)
        A = cv2.resize(A, None, fx=a.scale_pdf, fy=a.scale_pdf, interpolation=cv2.INTER_AREA)
    M, _, _ = sift_affine(A[..., 0], B[..., 0], partial=False)
    n, res = coherence(A, B, M, min_corr=a.min_corr)
    print(f'tiles used: {n}  (mapping CBZ->PDF: {describe(M)})')
    for key, name in (('Y', 'luma'), ('C', 'colour')):
        rows = res[key]
        if rows is None: print(f'{name}: no data'); continue
        print(f'{name}:  band       raw CBZ/PDF  coherence  PDF>=x  CBZ>=x   normalized-to-coarsest-band     verdict (needs >{a.margin:.2f})')
        for (lo, hi), r in zip(BANDS, rows):
            npdf = r['pdf_lb'] * rows[0]['cbz_lb']; ncbz = r['cbz_lb'] * rows[0]['pdf_lb']
            p, c = (npdf, ncbz) if a.normalize else (r['pdf_lb'], r['cbz_lb'])
            v = f'PDF has >= {p:.2f}x real detail' if p > a.margin else (f'CBZ has >= {c:.2f}x real detail' if c > a.margin else 'undetermined')
            print(f'  {lo:.2f}-{hi:.2f}     {r["raw"]:7.2f}    {r["coh"]:6.2f}    {r["pdf_lb"]:5.2f}  {r["cbz_lb"]:5.2f}    PDF>={npdf:4.2f} CBZ>={ncbz:4.2f}          {v}')
    print('bands are fractions of the CBZ Nyquist frequency. "raw" counts noise as detail - do not judge by it alone.')

def cmd_bw(a):
    """CBZ greyscale page vs 1-bit (or line-art) PDF image at higher resolution: is the CBZ just a downscale?"""
    C = load(a.cbz, 'L'); P = load(a.pdf, 'L')
    M, _, _ = sift_affine(P, C)            # CBZ -> PDF
    Minv = cv2.invertAffineTransform(M); s = np.hypot(Minv[0, 0], Minv[1, 0])   # PDF -> CBZ scale
    small = cv2.resize(P, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    best = None
    for tx in np.arange(-1, 1.01, 0.25):
        for ty in np.arange(-1, 1.01, 0.25):
            T = np.float32([[1, 0, Minv[0, 2] + 0.5 * s - 0.5 + tx], [0, 1, Minv[1, 2] + 0.5 * s - 0.5 + ty]])
            w = cv2.warpAffine(small, T, (C.shape[1], C.shape[0]), flags=cv2.INTER_LINEAR, borderValue=-1); m = w >= 0
            r = np.corrcoef(w[m][::7], C[m][::7])[0, 1]
            if best is None or r > best[0]: best = (r, w, m)
    r, w, m = best
    grey = (C > 40) & (C < 215)
    pw, pb = (w >= 254.5) & m, (w <= 0.5) & m
    print(f'PDF px per CBZ px {1 / s:.3f} | corr(CBZ, downscaled PDF) {r:.4f}')
    print(f'CBZ grey px where PDF footprint is pure white: {100 * (grey & pw).sum() / max(1, pw.sum()):.3f}% | pure black: {100 * (grey & pb).sum() / max(1, pb.sum()):.3f}%')
    print('interpretation: corr ~0.99 and ~0% grey where PDF is pure => CBZ has no tone/detail beyond the PDF (PDF >= CBZ)')

# ----------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('inspect'); p.add_argument('files', nargs='+'); p.set_defaults(f=cmd_inspect)
    p = sp.add_parser('register'); p.add_argument('pdf'); p.add_argument('cbz'); p.add_argument('--scale-pdf', type=float, default=1.0)
    p.add_argument('--save', help='save CBZ->PDF matrix as .npy'); p.set_defaults(f=cmd_register)
    for name, fn in (('fingerprint', cmd_fingerprint), ('reencode', cmd_reencode)):
        p = sp.add_parser(name); p.add_argument('pdf'); p.add_argument('cbz')
        p.add_argument('--dx', type=int, required=True); p.add_argument('--dy', type=int, required=True); p.set_defaults(f=fn)
    p = sp.add_parser('chroma'); p.add_argument('files', nargs='+'); p.set_defaults(f=cmd_chroma)
    p = sp.add_parser('dupes'); p.add_argument('image'); p.add_argument('--box', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1')); p.set_defaults(f=cmd_dupes)
    p = sp.add_parser('blocks'); p.add_argument('image'); p.add_argument('--offset', type=int, default=0,
                     help='grid offset: for a CBZ with B(x)=A(x+dx), the PDF grid sits at offset -dx'); p.set_defaults(f=cmd_blocks)
    p = sp.add_parser('coherence'); p.add_argument('pdf'); p.add_argument('cbz'); p.add_argument('--scale-pdf', type=float, default=1.0)
    p.add_argument('--normalize', action='store_true', help='judge fine detail relative to the coarsest band (use when colour conversions/gains differ)')
    p.add_argument('--margin', type=float, default=1.10); p.add_argument('--min-corr', type=float, default=0.95); p.set_defaults(f=cmd_coherence)
    p = sp.add_parser('bw'); p.add_argument('pdf'); p.add_argument('cbz'); p.set_defaults(f=cmd_bw)
    a = ap.parse_args(); a.f(a)

if __name__ == '__main__':
    main()
