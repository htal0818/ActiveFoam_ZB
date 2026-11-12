"""Download the paper's published Source Data (Nature Physics 17, 859, 2021)
and extract the Fig 2b/2c grids.  The spreadsheets are the publisher's files, so
they are fetched on demand rather than stored in this repository.

Fig 2b (volume fraction) and Fig 2c (mean number of contacts) are tabulated on
  rho = psi^2, psi = 0.90, 0.92, ..., 1.20   (16 densities, rho = 0.81 ... 1.44)
  W/T0 = 0, 0.02, ..., 0.80                  (41 adhesions)
Writes data/paper_fig2bc_source.npz (rhos, Ws, phi[rho, W], z[rho, W]).
"""
import io, os, urllib.request
import numpy as np
import openpyxl

BASE = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41567-021-01215-1/"
        "MediaObjects/41567_2021_1215_MOESM{}_ESM.xlsx")
FIG2 = 11          # MOESM11 = Fig 2, 12 = Fig 3, 13 = Fig 4, 14 = Fig 5


def main():
    raw = urllib.request.urlopen(BASE.format(FIG2), timeout=60).read()
    ws = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True).worksheets[0]
    rows = [r for r in ws.iter_rows(min_row=6, max_row=661, max_col=7, values_only=True)]
    a = np.array(rows, dtype=object)
    b = a[:, 0:3].astype(float); c = a[:, 4:7].astype(float)
    rhos = np.unique(np.round(b[:, 0], 4)); Ws = np.unique(np.round(b[:, 1], 4))
    phi = b[:, 2].reshape(len(rhos), len(Ws)); z = c[:, 2].reshape(len(rhos), len(Ws))
    os.makedirs("data", exist_ok=True)
    np.savez("data/paper_fig2bc_source.npz", rhos=rhos, Ws=Ws, phi=phi, z=z)
    print("rho:", rhos); print("W/T0:", Ws[0], "...", Ws[-1], f"({len(Ws)} values)")
    i = int(np.argmin(abs(rhos - 1.0)))
    print("rho=1: phi(W=0)=%.3f z(W=0)=%.2f" % (phi[i, 0], z[i, 0]))


if __name__ == "__main__":
    main()
