# MiniSAR data setup

Download the [Sandia National Laboratories MiniSAR archive](https://www.sandia.gov/files/radar/complex-data/MiniSAR20050519p0009image003.zip). From the ZIP, extract `MiniSAR20050519p0004image005.gff` into this directory. The archive filename differs from the name of the GFF member.

Verify the extracted file with `sha256sum real_scene/MiniSAR20050519p0004image005.gff` from the repository root. The expected SHA-256 is:

```text
1d20aa5a8f223594adb191d7ee8718bf2a9e4fecaa232b4857d2f8e9faf3f232
```

The default source crop uses range pixels `[0,1000)` and azimuth pixels `[0,800)`. The loader searches this directory recursively for a single copy of the named GFF file. Credit: Sandia National Laboratories, Radar ISR.
