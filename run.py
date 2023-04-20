import numpy as np
import sys
import datetime

output_folder = f"logs_{datetime.datetime.today().strftime('%Y-%m-%d')}"

if "--plan" in sys.argv:
    import pandas as pd
    data = []
    def main(**kwargs):
        print(kwargs)
        data.append(kwargs)
        pd.DataFrame(data).to_csv("jobs.csv")
else:
    from main import main

for noise in [0, 0.05, 0.10, 0.15, 0.20]:
    for k in range(1, 21):
        main(k=k, label_noise=noise,
             output_folder=f"{output_folder}/noise-{noise}")