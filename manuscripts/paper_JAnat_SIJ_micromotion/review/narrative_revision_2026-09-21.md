# Anatomical narrative revision — 2026-09-21

## Scope

Writing and restructuring using existing outputs only. No simulations, cohort extraction, numerical analysis, multiprocessing or web browsing were performed in this revision. The bibliography includes the 19 sources already present after the preceding reference audit; no references were added during this writing task.

## Revised hierarchy

1. Primary anatomical result: adjustment for volume and the selected dimensions substantially attenuates age-adjusted female translation coefficients. All fully adjusted translation intervals span zero. The pooled subpubic-angle correlation is not reproduced within either sex.
2. Model support: individual geometry retains response under standardized bone density, conditional on the implemented density law, fixed cartilage and ligament properties and pretension, and passive static mechanics.
3. Mechanical context: matched load redistribution and application-site contrasts establish why no single scalar mobility describes the anatomy across loads. The complete transverse-load hypothesis remains unsupported.
4. Supporting result: inverse conditional volume associations, interpreted against dimensional expectations.

Results retain the requested load → sex/anatomy → geometry/density → size → sensitivity sequence. Discussion instead develops the anatomical argument first. All numerical estimates, uncertainty intervals and null findings were retained; the rotation null finding is also stated explicitly.

## Files changed in this writing revision

- `sections/abstract.tex`, `introduction.tex`, `discussion.tex`, `conclusion.tex`: substantial rewrites.
- `sections/results.tex`: revised hierarchy, topic sentences and captions; removed the redundant closing hypothesis catalogue while preserving findings in their corresponding subsections.
- `sections/methods.tex`: retrospective H1–H4 definitions retained in subordinate methodological form; technical checks moved to Appendix.
- `sections/appendix.tex`: archived interpolation and correspondence details retained in a labelled subsection; corrected an unmatched quotation mark.
- `main.tex`: float barrier separating Methods from Results. Figure labels, order, filenames and table inputs remain unchanged. The DOI bibliography style change belongs to the preceding reference task.
- `analysis/REPRODUCE.md`: serial editorial build separated from historical numerical reproduction; warning that the old prose generator would overwrite the new manually edited narrative.
- `editor_pitch.md`: unsent draft aligned with the new principal result.
- `CHANGELOG.md`: revision record.
- `main.pdf` and LaTeX build products: rebuilt.

## Quality control

- Serial latexmk build successful: 27 pages, no LaTeX warnings, undefined citations/references or overfull/underfull boxes.
- SHA-256 comparison confirmed all 326 protected figure, table/data and bibliography files unchanged during this writing revision.
- All 19 citation keys resolve; all labels are unique; every main figure has an in-text reference; no duplicate long prose paragraphs detected.
- Supplementary hard-coded references remain correct: main morphology Figure 1, allometry Figure 7 and Table 4. Supplementary sources/data were not modified.
- Visually inspected the compiled first page and the central sex/anatomy and geometry/density figure pages. Captions and plots fit without overlap.
- Read Abstract, final Introduction paragraph, first two Discussion paragraphs and Conclusion consecutively: they consistently foreground explicit anatomy, conditional geometry retention and load context, without asserting sex equivalence or causal mediation.

## Items for author review

- Anatomy-adjusted attenuation is not an out-of-sample model comparison or proof of causal explanation. The revised Discussion explicitly distinguishes those claims.
- Coverage and overlap of male/female anatomical predictor ranges, retrospective triad selection and collinearity warrant scrutiny; no new diagnostic was computed.
- The modulus floor can reduce density sensitivity; individual cartilage, ligament and pretension variation remains untested.
- Cohort overlap/provenance relative to the earlier 281-model study, ethical documentation, public data/code archive links and final author approval remain submission requirements stated in the working manuscript.
- SIJ kinematics still require independent validation, mapping/mesh assessment and broader sensitivity checks. The six-subject extraction check is not a cohort-wide bound.
- The existing geometry figure shows variance retention rather than a paired agreement scatterplot. Its caption now gives existing agreement statistics and points to Supplementary Table S2; no new plot was fabricated.
