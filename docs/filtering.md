# Structure filtering summary

Funnel from raw `xyz-files` through the gather/clean filters used by the clustering pipeline (`gather_structures`): reject `-extended`/`.pc`/`.gzmat` → drop mixed-ligand coordination (a coordinating non-Cys/His residue; water tolerated) → drop off-modal coordinating-atom count → drop off-majority composition. `Kept` is what gets clustered.

Generated 2026-07-07.

| Config | Listed | Mixed-ligand dropped | Off-count dropped | Off-composition dropped | Parse fail | Kept | Kept composition |
|---|--:|--:|--:|--:|--:|--:|---|
| 4cys | 3007 | 0 | 0 | 0 | 8 | 2999 | 4Cys |
| 1cys3his | 30 | 0 | 1 | 0 | 0 | 29 | 1Cys 3His |
| 2cys2his | 521 | 0 | 1 | 0 | 0 | 520 | 2Cys 2His |
| 3cys1his | 2428 | 1 | 1 | 0 | 2 | 2424 | 3Cys 1His |
| 4his | 36 | 19 | 3 | 0 | 0 | 14 | 4His |

