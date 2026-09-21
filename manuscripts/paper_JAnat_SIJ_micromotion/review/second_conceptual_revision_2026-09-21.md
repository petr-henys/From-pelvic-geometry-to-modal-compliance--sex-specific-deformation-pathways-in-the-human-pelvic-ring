# Second conceptual revision — 21 September 2026

The manuscript now leads with anatomical scale, individual architecture and load path. Sex adjustment remains an important anatomical comparison, without claims of mediation, equivalence or superior prediction. Scaling remains retrospective/exploratory. The dimensional derivation is a mechanical reference, not biological allometry or a new hypothesis test. All statistical estimates and generated numerical tables were retained.

## Files revised

- `main.tex`: title, data/code availability and visible submission TODOs.
- `sections/abstract.tex`, `introduction.tex`, `results.tex`, `discussion.tex`, `conclusion.tex`: connected scale–architecture–load-path narrative; Discussion reordered as requested.
- `sections/methods.tex`: cohort provenance TODO and alignment of comparison descriptions; model specifications unchanged.
- `bib/refs.bib`: five verified primary references added, no references removed and no existing DOI changed.
- `analysis/generate_publication_figures.py`: separate `build_scaling_benchmarks()` function reading only saved `allometry_models.csv`; added dashed references and moved legends outside data panels.
- `figures/Fig6_allometry_loglog.pdf` and `.png`: Figure 7 in manuscript numbering (historical asset name retained). Rotation reference −2/3; translation −1/3. Vertical references correspond to the horizontal exponent axis; caption explains their dimensional meaning.
- `editor_pitch.md`: matched broader scientific hierarchy.
- `analysis/REPRODUCE.md`, `CHANGELOG.md`, this report and saved reference verification records: revision provenance.

## Reference verification and claim audit

Crossref metadata for all five additions are saved in `review/references/*_crossref.json`; primary abstracts are in `second_revision_primary_abstracts.json`. Author lists, titles, journals, issue years, volumes, issues, pages and DOIs were checked. Nishi uses the 2020 issue year (online 2019). Joukar includes the subtitle present in PubMed but omitted from Crossref's title field; the fourth author is indexed as Vosoughi, Ardalan Seyed. Poilliot names and diacritics follow primary metadata. The optional Ulas review was not needed.

| Passage/claim | Supporting evidence and scope |
|---|---|
| Introduction 1: SIJ load transfer, compression and ligament restraint | Vleeming 2012 provides synthesis; Snijders 1993 supplies self-bracing mechanics; Hammer 2013 supplies primary ligament/load-distribution modeling. These do not validate this cohort's motion magnitudes. |
| Introduction 2: developmental pelvic sex differences and local auricular shape | Huseynov 2016 retained for developmental form; [Nishi 2020](https://pubmed.ncbi.nlm.nih.gov/31792910/) added for Japanese skeletal auricular morphology, without inferring mechanics from shape alone. |
| Introduction 3 and Discussion size/shape/sex: stature–shape covariation; allometric versus non-allometric dimorphism | [Fischer 2015](https://doi.org/10.1073/pnas.1420325112) and [Fischer 2017](https://doi.org/10.1002/ar.23549). Subpubic-angle dimorphism is described as largely non-allometric; neither paper supports the mechanical benchmarks. |
| Introduction 4 and Discussion sex comparison: prior standing FE sex differences | [Joukar 2018](https://pubmed.ncbi.nlm.nih.gov/29509655/) added as substantive prior representative-model evidence; cohort adjustment is a complementary question, not a dismissal. |
| Introduction 4: obstetric/support interpretation | Pavličev 2020 retained for contextual synthesis, not proof of mechanical or evolutionary trade-offs in this cohort. |
| Introduction 5 and Discussion implications: prior population auricular mechanics | Henyš & Hammer 2025 retained and distinguished from present motion endpoints; overlap remains unresolved. |
| Introduction 6: source workflow; childbirth-model scope | Henyš & Kuchař 2025 retained for BoneDat; Chen & Grimm 2021 retained specifically as a review of childbirth modeling requirements. |
| Discussion scale: conditional slopes and dimensional benchmarks | Existing saved estimates, model specifications, and elementary dimensional derivation. Potential contributors are explicitly hypotheses for future investigation, not externally established explanations of these slopes. |
| Discussion geometry: local mineralization versus morphology | [Poilliot 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10439371/) added. Its limited pattern differences across shape categories and higher iliac mineralization in larger surfaces are reported cautiously; it is not cited as proof that geometry dictates density or that density is mechanically irrelevant. |
| Discussion geometry: pretension and represented tissues | Heyland 2025 retained for sensitivity to ligament pretension; fixed properties, floor and non-additive variance ratios are this model's limitations. |
| Discussion load path and functional anatomy | SP2leg/SP1leg and LAB1/LAB2 claims derive from saved paired results. Snijders/Vleeming provide load-transfer context, not evidence of the new contrasts. Fischer 2015 supplies obstetric anatomical integration only. No signed canal expansion, delivery outcomes or optimization inferred. |
| Discussion limitations: pelvic validation and small SIJ motion measurement | Anderson 2005 retained for cortical-strain validation in a subject-specific pelvic model, explicitly not transferable to these kinematics; Goode 2008 retained as measurement synthesis. |

All other factual statements in Introduction and Discussion are either current study findings, documented modeling choices/limitations, or explicitly prospective interpretations. Existing-reference scope was checked against the previous detailed audit and saved primary records (`reference_audit_cs.md`); that audit's 19-reference count describes the earlier revision. Current bibliography: 24 entries, all cited; no duplicate keys or DOI records, missing citation keys, unused entries or malformed DOI strings.

## Submission blockers and provenance evidence

Repository text searches and the existing author checklist do not supply an identifier-level reconciliation of the present 278 subjects with the earlier 281-model study. The exact overlap, exclusions explaining the count difference, reused data and novelty of endpoints must be documented by the authors. Methods now contains a prominent TODO covering all four points and explicitly avoids independent-validation framing. Counts alone were not used to infer overlap.

A real public development remote was found and its GitHub page verified:
https://github.com/petr-henys/From-pelvic-geometry-to-modal-compliance--sex-specific-deformation-pathways-in-the-human-pelvic-ring

It is linked in Code Availability. It does not establish a versioned release of this revised manuscript package. The data release identifier/access terms, original-field access and exact archived code/environment remain visible submission blockers. No URL or release DOI was fabricated. Existing ethics and final-author-approval checklist items remain for author resolution.

## Computational scope

Only the saved-coefficient Figure 7 function was invoked. The full figure generator was not run: its supplementary scatter routine can refit models. No FE solve, displacement extraction, regression fitting, bootstrap, parameter sweep or statistical test was run. New benchmark tests, between-load slope tests, causal explanations of scaling, tissue-variation analyses and independent validation were left uncomputed. Numerical results remain unchanged.

## Final quality control

Read the Abstract, final Introduction paragraph, Results 3.5, opening/scale Discussion, functional-anatomy synthesis and Conclusion consecutively: each distinguishes standardized-force scaling from organismal allometry and connects scale, architecture and load path. Numerical Results outside the scaling interpretation were preserved. The final serial latexmk/BibTeX build completed successfully (29 pages), with no undefined citations/cross-references, LaTeX warnings or overfull/underfull boxes in `main.log`. All 24 bibliography entries are cited. Figure 7's saved-estimate image and compiled page were visually checked; benchmark labels and legends do not cover the estimates. No statistical model was refitted.
