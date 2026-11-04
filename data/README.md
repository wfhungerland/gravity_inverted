# Data

The engine reads `src/gravity_inverted/data/wiod2016_flows.csv.gz`: bilateral
flows (exporter, importer, US$ million) for 43 economies plus the rest of the
world, 2000–2014, aggregated from the WIOD 2016 release (Timmer et al. 2015).

Build it from the R-format tables (`WIOTS_in_R.zip`,
https://doi.org/10.34894/PJ2M1C):

```bash
pip install -e ".[data]"
python scripts/build_wiod.py /path/to/WIOTS_in_R
```

Own data: a long CSV with columns `exporter, importer, value`, including
domestic sales (`exporter == importer`) for every economy. See
`custom_template.csv`.

## Licence and attribution

WIOD is licensed under CC BY 4.0. The bundled file is a derived aggregate.
Source: Timmer, M. P., Dietzenbacher, E., Los, B., Stehrer, R. and de Vries,
G. J. (2015), "An Illustrated User Guide to the World Input–Output Database:
the Case of Global Automotive Production", *Review of International
Economics*, 23(3), 575–605. Data: https://doi.org/10.34894/PJ2M1C
