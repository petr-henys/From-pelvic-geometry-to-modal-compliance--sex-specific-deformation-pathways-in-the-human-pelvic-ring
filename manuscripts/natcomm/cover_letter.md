Dear Editors of *PLOS Computational Biology*,

Please consider our manuscript, **“Stable and interchangeable deformation pathways preserve pelvic mechanical function across anatomical variability,”** for publication as a Research Article in *PLOS Computational Biology*.

A central methodological challenge in population biomechanics and computational morphology is comparing mechanical deformation pathways across anatomically variable individuals. Standard computational pipelines sort deformation modes by stiffness or eigenvalue rank, implicitly assuming that the same rank index carries identical biological function in every subject. In this study, we show that this assumption breaks down whenever adjacent eigenvalues approach each other: small anatomical perturbations drive mode veering and label exchanges, even when the mechanically functional subspace remains completely preserved across the population.

To address this problem, we developed an invariant-subspace and covector projection framework and applied it to a population-scale cohort of $N = 278$ CT-derived human pelvic finite-element models assembled on a shared reference domain with subject-specific bony geometry and heterogeneous bone elasticity. Across the cohort, pelvic mechanics separates into two distinct organisational scales:
1. **An order-locked low-rank backbone (modes 1–3):** Spectrally gap-protected, rarely reordered across subjects ($< 0.4\%$ swap rate), and routing over 84% of retained modal strain energy during physiological standing.
2. **Compact recurrent clustered subspaces at higher ranks:** Near-degenerate blocks where individual solver labels become unstable. Most prominently, the rank 9–10 block (present in 54.7% of subjects) preserves an anterior–posterior inlet-opening deformation program. When rank labels 9 and 10 exchange across subjects, label-level coupling drops drastically, whereas coupling within the shared two-dimensional subspace remains consistently high ($p = 0.083$, median 0.523 vs. 0.509 mm/mm). This block is selectively recruited under parturition-motivated proxy loading.

We further show analytically, using a minimal two-degree-of-freedom model, that parameter-driven detuning across a near-degenerate zone fully explains this label–subspace dissociation. First-order analytical sensitivity analysis incorporating shared-parameter covariance confirms that the fragile-versus-robust spectral gap architecture remains identifiable under plausible soft-tissue and bone density uncertainty.

We believe this study is particularly suited for *PLOS Computational Biology* because it bridges computational mechanics, structural biology, and functional morphology. Rather than treating higher-order modal variability as unstructured noise, our approach provides a general, reproducible method for identifying robust biological deformation programs in complex anatomical structures without assuming one-to-one preservation of mode labels.

In accordance with PLOS Computational Biology policies:
- All computational workflows, solver-order arrays, and plotting routines are fully reproducible and available in the study repository.
- The manuscript represents original work, has not been published previously, and is not under consideration for publication elsewhere.
- All authors have reviewed and approved the manuscript and declare no competing interests.

Thank you for your consideration.

Sincerely,

Petr Henyš, Ph.D.  
Institute of New Technologies and Applied Informatics  
Faculty of Mechatronics, Informatics and Interdisciplinary Studies  
Technical University of Liberec, Czech Republic  

Niels Hammer, M.D., Ph.D. (Corresponding Author)  
Division of Macroscopic and Clinical Anatomy  
Gottfried Schatz Research Center  
Medical University of Graz, Austria  
E-mail: niels.hammer@medunigraz.at
