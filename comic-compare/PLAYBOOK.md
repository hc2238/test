# PDF vs CBZ comic image comparison: playbook

This is a handoff kit for continuing the comparison in a new session. The tool is `comparekit.py` (one file, CLI subcommands). This document covers the method, the decision rules, the pitfalls already hit, and every result so far.

## Setup

```bash
pip install -r comic-compare/requirements.txt     # numpy, pillow, opencv-python-headless
python3 comic-compare/comparekit.py -h
```

Treat uploaded zips as untrusted data. Unzip each one into its own empty folder outside the repo, and run Python with `-I`. Don't commit the comic images.

## Inputs the user provides

For each book, the user sends a zip of matched pages. Each page usually has:
- **`PDF_raw`**: the image extracted from the PDF (JPEG, or PNG for lossless/Flate images, and sometimes CMYK JPEG).
- **`PDF_*_for_viewing`**: the user's own conversion. It may be inverted (1-bit masks), CMYK→RGB (a naive conversion), or a render.
- **`CBZ`**: the page from the CBZ/CBR (JPEG, PNG, or JXL plus a `_for_viewing` PNG).

PDF images often lack elements the PDF draws separately: lettering, page numbers, and some vector boxes. The user knows this, so compare the shared art only. Sample pairs can be mismatched pages; one case had CBZ page 59 against a different PDF page. Check visually before trusting numbers.

## Process (in order)

1. **Look first.** Build a downscaled side-by-side strip of each pair and view it. Check that it's the same page, and note missing lettering, borders, colour differences and layout.
2. **`inspect`** every file: format, mode, size, quantization sums (lower = finer), subsampling (4:2:0 vs 4:4:4), grey levels (2 = 1-bit), JXL `jbrd` box (a lossless JPEG repack).
3. **`register`** PDF → CBZ. The key number is `a_px_per_b_px`, the content resolution ratio. It's independent of borders and canvas size, so **never compare raw pixel dimensions**. Exact ratios (2.000, 1.000, 0.400) hint that one file was derived from the other.
4. **Provenance tests.** Was the CBZ made from the PDF image? Use whichever fits:
   - Same scale, whole-pixel offset, PDF is JPEG → **`fingerprint`**. A CBZ lattice distance well below the off-grid controls proves the CBZ was made from that PDF JPEG. It only works when the CBZ was saved at a quality similar to or finer than the PDF's and wasn't resampled or smoothed.
   - PDF lossless (PNG), same grid → the PDF *is* the source at that resolution. The CBZ can only be equal or worse. `reencode` shows how much error one JPEG save adds.
   - CBZ at a larger scale → **`dupes`**. A large fraction of duplicated rows and columns means a nearest-neighbour upscale; collapse the duplicates and compare with the PDF.
   - 1-bit PDF vs greyscale CBZ → **`bw`**. Correlation of about 0.99 and almost no grey where the PDF is pure white or black means the CBZ is a downscale of the PDF.
   - Layered PDFs (line art + colour layer, or artwork + paper texture): composite the layers, then compare. See Hate Revisited and Hobtown below.
5. **Detail test: `coherence`.** This is the main quality metric. Use `--scale-pdf 1/ratio` when the PDF is at a higher resolution; downscaling with INTER_AREA is conservative against the PDF. Use `--normalize` whenever colour conversions or overall gains differ (CMYK conversions, paper multiply).
6. **Colour reality: `chroma`.** Does 4:4:4 hold real full-resolution colour, or was it upsampled from 4:2:0?
7. **Blocking: `blocks` / `fingerprint`.** Block-edge strength on the PDF's grid. A drop to about 1.00 in the CBZ means smoothing, or a cleaner source.
8. **Zoom by eye at 4–5×** (nearest neighbour) on detailed regions to confirm whatever the numbers say.

## How to judge (decision rules)

- **Never judge by raw high-frequency energy.** JPEG noise, ringing, block edges, sharpening halos and resampling aliasing all add fine-scale energy. Only `coherence` bounds count, and they need a margin above 1.10.
  - How the bound works: model PDF = a·S + Nₚ and CBZ = b·S + N_c, with independent noise. Per frequency bin, |Sxy|² = (PDF signal)(CBZ signal), so PDF signal ≥ |Sxy|²/Syy and CBZ signal ≥ |Sxy|²/Sxx. Bias correction: subtract Sxx·Syy/N.
  - It was validated on synthetic cases: it detects a blurred copy, and it doesn't credit added noise.
- **Provenance beats metrics.** If the CBZ was made from the PDF image, the PDF is upstream and can't be worse, apart from display-side processing (see Knights). If the PDF is lossless at the same grid, the PDF wins in the strict sense.
- **Separate "strictly better" from "visibly better".** Many PDF wins are under 1–2 levels and invisible.
- **The CBZ can look better without having more information,** for example through deblocking (Knights). Say so explicitly. The user could get the same result from the PDF's data with jpeg2png or jpeg-quantsmooth.
- **CMYK images:** a naive CMYK→RGB conversion is oversaturated. A colour-managed reader will look closer to the CBZ. Don't count colour-strength differences as detail; use `--normalize`.

## Pitfalls already hit (don't repeat these)

1. **`cv2.phaseCorrelate` modifies its input arrays in place** when given a window, which silently corrupted images. Always pass `.copy()`; `phase_shift()` does this.
2. **Sign of phaseCorrelate:** for `b(x) = a(x − d)` it returns `+d`. An earlier script used the wrong sign, which over-masked pages and misplaced tiles. This is fixed in the kit and was verified synthetically.
3. **Resampling bias:** shrinking or warping one image onto the other's grid blurs it and biases detail comparisons. Compare each image's native tiles; spectra don't depend on shifts. For scale differences, use `--scale-pdf` with INTER_AREA, which is conservative against the PDF.
4. **Tone mismatches break masks:** a raw PDF image can have blacks at 32 when the colour space isn't applied (Lastman), and high-contrast line art trips difference masks. The coherence tool masks per tile by correlation instead.
5. **Pillow quantization tables come back in natural order** (verified empirically). `qtables=im.quantization` round-trips correctly for `reencode`.
6. **The lattice test fails silently** when the CBZ was saved more coarsely than the PDF (Rain, Astro City), resampled (Mercy), or composited (Hobtown). "No fingerprint" doesn't prove "independent source".
7. **Finite-sample bias** in coherence produced a false "PDF ≥ 1.06×" with few tiles. It's now corrected, and a 1.10 margin is required.
8. **Gain differences** (saturation, paper multiply) create fake "detail" advantages at every band. Use `--normalize`, which compares fine detail relative to the coarsest band.

## Results so far (15 books)

| Book | Key finding | Verdict |
|---|---|---|
| How I Make Comics | The PDF is 1-bit at 2.0× resolution; the CBZ is its downscale (corr 0.99, no extra tone). Sample 3 was a mismatched page. | **PDF, clearly** |
| War on Gaza | The PDF is 1-bit at 2.0× (1.15× on sample 1); the CBZ is a 16-grey downscale. | **PDF, clearly** |
| Mercy | Slightly resampled CBZ. Coherence: the PDF has ≥1.2–1.5× real brightness detail and ≥1.3× colour at mid bands. | **PDF, clearly** |
| Seven to Eternity | PDF at 1.19×. Coherence after shrinking the PDF: the PDF has ≥1.4–2.6× real colour detail and ≥1.1–1.2× brightness (pair 1). | **PDF, clearly** |
| Okinawa | Colour: the CBZ is a nearest-neighbour 2.5× upscale of the PDF JPEG (median difference 0). B&W: identical bitmap. | PDF (colour, slight); tie (B&W) |
| Hate Revisited | Layered PDF: 1-bit line art at 2× plus a colour layer at 1×. The CBZ is the composite downscaled. The cover adds logo, price and bars. | PDF (strict) |
| Nocturnos | The CBZ carries the PDF JPEG fingerprint; its 4:4:4 colour is empty. The PDF's lettering is separate. | PDF (strict) |
| Lastman Book 5 | The PDF is lossless PNG (B&W) or CMYK; the CBZ is resized 0.981 plus JPEG. Equal detail. | PDF (strict, <1 level) |
| Astro City Metrobook 1 | Whole-pixel crops or tiny resizes. Coherence undetermined; page 410 is lossless in the PDF. | PDF (strict) |
| PeePee PooPoo #2 | Pair 1: the PDF is lossless. Pair 2: CMYK, and the colour advantage vanishes once normalized. | PDF (strict) |
| 20th Century Men | No fingerprint; coherence undetermined. | Tie |
| Rain (Joe Hill) | Whole-pixel crop, about 1.4 levels difference; undetermined. | Tie |
| Die | The CBZ (a JXL repack of a JPEG) carries the PDF fingerprint; identical detail. | Tie |
| Knights of Heliopolis | The PDF JPEG is very heavily compressed. The CBZ carries its fingerprint, but deblocked (block edges 1.26→1.09 brightness, 2.4–3.3→1.7–1.9 colour); 97–98% of coefficients stay in the PDF's quantization bins. | CBZ looks better (smoothing only) |
| Hobtown Vol 1 & 11 | Layered PDF: artwork JPEG plus paper texture plus masks. The CBZ is the artwork × paper with **zero** PDF block edges, and provably ≥1.1–2.3× more colour detail (normalized). A control using the user's PDF render, which includes the paper, shows no gain and keeps the blocks, so the gain isn't from the paper or from rendering. Brightness detail is equal. | **CBZ better** (better-handled source or very good decoding; unresolved) |

## Handoff prompt for a new session

> Read `comic-compare/PLAYBOOK.md` and use `comic-compare/comparekit.py` to compare the PDF-vs-CBZ page pairs in the zip I'm attaching, following the Process and decision rules. Look at the pages first, report a per-book verdict in the same style as the results table, and add the book to that table.
