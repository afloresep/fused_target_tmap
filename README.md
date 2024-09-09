This project aims to provide a comprehensive visualization of the most active compounds for a diverse array of targets sourced from the ChemBL database.

- Visualization of Top Compounds: Identify and visualize the top 5 most active compounds for approximately 15,000 distinct biological targets.
- Fingerprint Merging and Minhashing: For each target, take 5 compounds and calculate their fingerprint using MAP*. These fingerprints, which encapsulate the chemical characteristics of each compound, are then merged to create a unified representation (`merged_fp`). Subsequently, the merged fingerprints are subjected to minhashing techniques to reduce dimensionality and facilitate efficient comparison (i.e. calculating Jaccard distances).
- Run TMAP
