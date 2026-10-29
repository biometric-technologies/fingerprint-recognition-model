# Fingerprint embedding model

An Xception-style network that maps a 192x192 grayscale fingerprint image to a
128-dimensional L2-normalized embedding, trained with triplet loss on
CASIA-FingerprintV5. Two impressions of the same finger should be close; different
fingers should be far apart.

## Train
Put CASIA-FingerprintV5 in `fingerprint-v5-master` (`<subject>/<L|R>/<subject>_<hand><finger>_<sample>.bmp`)
and run `python3 train.py`. Subjects are split 80/20 into training and validation.

## Evaluate
Put held-out subjects in `fingerprint-v5-master-test` and run `python3 test.py`. It reports,
over all pairs of test images, FMR and FNMR at distance 1.0, the EER and FNMR at FMR 1%,
0.1% and 0.01% (when there are enough impostor pairs), in the terms of ISO/IEC 19795-1.

## Notes
- CASIA-FingerprintV5 is licensed for non-commercial research; check the license before
  using a model trained on it in a product.
- An embedding model is not interoperable with minutiae templates (ISO/IEC 19794-2,
  ANSI/INCITS 378). For exchange with other systems, use minutiae extracted by
  [biometric-converter](https://github.com/biometric-technologies/biometric-converter).
