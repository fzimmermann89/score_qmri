# Final bounded M4Raw GRE follow-up

No GPU work, scheduler jobs, changes to original files, full-corpus regeneration, or PR push performed. Evidence was saved only in this new audit directory. One new real slice precision probe used about6seconds on one CPU; the remaining work read small source/history/metadata files and hashed two12MB raw inputs at each independent store.

## The actual executed producer was recovered

IPython session267 began2026-01-05 at12:49:20.871986. Executed cell72 defines clean_mask_2d and remove_polynomial_phase exactly AST-identical to the current original m4raw.py. Cell73 points to /home/zimmer08/datacheck/data/m4raw; cell74 groups all recursive H5 inputs by subject and sequence. Executed cell89 defines ift and the current ESPIRiT routine exactly, including calibration48, kernel12, threshold0.001, crop0.1 and50power iterations. Final cell97 writes the full GRE result with fourth-order phase fit threshold0.2..0.7, normalization foreground>0.1 quantile0.98 and mask0.1. The preserved original H5 mtime2026-01-05T14:32:38Z is consistent with the final Jan5 editor changes and this executed producer. This is stronger than inferring behavior from whichever source file was saved last.

See m4raw-executed-original-cells.json for exact selected cells and SHA256. It excludes unrelated history. m4raw-editor-function-history.json records the bounded editor-source change timeline. The phase fitting helper had been stable since Dec18; the only final Jan5 calibration difference from Dec19 source was the documented threshold/crop change.

## Candidate explanations checked

- Rewriting the source: excluded for the tested case. Running isolated original CPU functions and the cleaned CPU pipeline on the complete18sliceGRE subject produced bit-identical arrays, maximumdifference0.
- Curation/import: excluded for the tested case. Archived pre-curation GRE subject and selected current GRE subject are bit-identical.
- Raw copies differing between available stores: excluded for this subject. The two independent Echo public raw files and Ceph files have identical full-file SHA256. GRE01=bd92f53670e1227cbbad572f6569a87323c551ecba3f3abf63e0f869c9b42b91; GRE02=95a917e8e5041cbd80d655709a40b1ff094deb55247b4ac3d98a38bcbc3f1e09. This does not prove historicalJan5 raw bytes were identical because no contemporaneous raw manifest was recorded.
- Native versus sorted repeat/coil order: both complete subject runs retain the discrepancy, MAE about0.0041.
- Source-backed older calibration0.05/crop0.5: worsens error toMAEabout0.118 rather than resolving it. This was an explicitly recovered historical version, not arbitrary parameter tuning.
- A changed mrpro pad/crop helper: the available preJan5 implementation and current installed implementation have identical function AST. The exact old installed wheel is unknown, but there is no evidence supporting a padding-alignment change.
- A single global complex phase/scale: fitted correction leavesMAE0.00408, so insufficient.
- Straightforward complex64 precision loss: single actual slice9 was calibrated in complex64 andcomplex128 on CPU, both then castcomplex64 before the original phase fit. The two CPU phase outputs differ by only0.000326rad on common foreground, while either differs from archived output byabout0.052rad. Higher CPU precision does not restore the historical target. This probe does not reproduce a historical CUDA SVD driver.
- Explicit numerical backend settings in the executed session before finalwrite: none of the searched precision/default-dtype/determinism settings were recorded. Absence does not establish the actual runtime state.

## What is and is not established

The recovered executed producer, input grouping and formulas are known. Current source is reproduced faithfully by the cleaned implementation. The selected archived H5 has an exact reproducible path through the preserved intermediate. The remaining raw-to-archive discrepancy is real and unresolved; attributing it simply to CPU versus GPU would exceed the evidence.

The current preparation environment contains torch2.9.1 and mrpro0.260420+dirty4e90b588. No historical dependency lock/runtime fingerprint for the January kernel was found, so those current versions cannot be claimed as historical. A historical environment snapshot, contemporaneous raw-input hashes and/or an explicitly authorized isolated comparison using the original CUDA reconstruction would distinguish remaining possibilities: historical numerical/runtime behavior, stale kernel state not represented in source, or historical input revision. No such GPU comparison was launched.

The committed-in-progress primary reproduction route remains m4raw-prepared: all four modalities of fullsubject2022061001 were confirmed bit-identical after original slice curation. Raw reconstruction stays an explicit alternative with its limitations; canonical dataset files and catalogue selections were left untouched during this follow-up.
