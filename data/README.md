# Data directory

## Tracked in the repository

- **`cascade_beta.csv`**, the inter-sector dependence matrix used by the model,
  derived from the WIOD 2016 tables. This is the only data file the main
  pipeline needs at run time.

## Not tracked (download separately)

- **`WIOTS_in_EXCEL/`**, the World Input-Output Database (WIOD) 2016 release
  tables (`WIOT2000_Nov16_ROW.xlsb` ... `WIOT2014_Nov16_ROW.xlsb`). These files
  are ~62 MB each (~930 MB total), above GitHub's file-size limit, and are the
  property of the WIOD project, so they are not redistributed here.

  They are needed only to **re-derive** `cascade_beta.csv` from scratch with
  `code/experiments.py build-cascade`. The main optimization pipeline does not read
  them.

### How to obtain the WIOD tables

1. Download the 2016 release ("WIOTs in Excel", November 2016 update) from the
   WIOD website (`https://www.rug.nl/ggdc/valuechain/wiod/`).
2. Place the `.xlsb` files in `data/WIOTS_in_EXCEL/`.
3. Regenerate the matrix:

   ```bash
   pip install pyxlsb          # only needed for this step
   python code/experiments.py build-cascade
   ```

This rewrites `data/cascade_beta.csv`. Because that file is already provided, you
only need the WIOD tables if you want to reproduce the cascade calibration itself.

See `../DATA_PROVENANCE.md` for the source and interpretation of every
calibration input.
