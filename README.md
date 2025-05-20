# GWFish

## About

<p align="center">
  <img src="gwfish-1.png" width="200" title="Logo">
</p>
Simulation of gravitational-wave detector networks with Fisher-matrix PE

Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

Read the documentation [here](https://gwfish.readthedocs.io)!

## Citation

Please cite [GWFish publication](https://doi.org/10.1016/j.ascom.2022.100671) if you make use of the code:
```
@ARTICLE{DupletsaHarms2023,
        author = {{Dupletsa}, U. and {Harms}, J. and {Banerjee}, B. and {Branchesi}, M. and {Goncharov}, B. and {Maselli}, A. and {Oliveira}, A.~C.~S. and {Ronchini}, S. and {Tissino}, J.},
        title = "{GWFISH: A simulation software to evaluate parameter-estimation capabilities of gravitational-wave detector networks}",
        journal = {Astronomy and Computing},
        keywords = {General Relativity and Quantum Cosmology},
        year = 2023,
        month = jan,
        volume = {42},
        eid = {100671},
        pages = {100671},
        doi = {10.1016/j.ascom.2022.100671},
        archivePrefix = {arXiv},
        eprint = {2205.02499},
        primaryClass = {gr-qc},
        adsurl = {https://ui.adsabs.harvard.edu/abs/2023A&C....4200671D},
        adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Open GWFish tutorial in Google Colab

The tutorial notebook can be opend in Google Colab without the need to download locally any package. Here is the link: [notebook GWFish](<https://colab.research.google.com/github/janosch314/GWFish/blob/main/gwfish_tutorial.ipynb>)


## A note about sensitivity curves 

- A# sensitivity: data from [here](https://dcc.ligo.org/LIGO-T2300041/public)
- Aplus: data from [here](https://dcc.ligo.org/LIGO-T2000012/public):
    - LIGO: [AplusDesign.txt](https://dcc.ligo.org/public/0165/T2000012/002/AplusDesign.txt)
    - Virgo: [avirgo_O5high_NEW.txt](https://dcc.ligo.org/public/0165/T2000012/002/avirgo_O5high_NEW.txt)
    - KAGRA: kagra_Design.txt
- CE 40km updated: [cosmic_explorer_strain.txt](https://dcc.cosmicexplorer.org/CE-T2000017/public)
- ET: data from [here](https://apps.et-gw.eu/tds/?r=18213)
    - 10 km cryogenic: [ET10kmcolumns.txt](https://apps.et-gw.eu/tds/?r=18213), third column
    - 15 km cryogenic: [ET15kmcolumns.txt](https://apps.et-gw.eu/tds/?r=18213), third column

This work is licensed under a [Creative Commons Attribution 4.0 International
License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg
