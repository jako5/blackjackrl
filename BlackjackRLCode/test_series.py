from BlackjackRLCode.Blackjack import *
from tqdm import tqdm
import os

def run_series():
    st = time()

    SAMPLES = 1000

    testconfigs = [(ds, bs, de) for ds in ["Threshold","DecisionTable"]  # ,
                   for bs in ["Constant", "HiLo", "Halves"]  #
                   for de in [1]]

    subfolder = "/strategies"

    print("TESTING", len(testconfigs), "configs")
    for DRAW_STRATEGY, BET_STRATEGY, DECKS in tqdm(testconfigs):
        specs = {
            "MAXROUNDS": 1000,
            "THRESHOLD": 16,
            "DECKS": DECKS,
            "DRAW_STRATEGY": DRAW_STRATEGY,  # Manual, Threshold, DecisionTable
            "BET_STRATEGY": BET_STRATEGY,  # Constant, HiOpt, HiLo, Omega, Halves
            "LOG": False,
            "LOGRESULTS": False,
            "PLOT": False,
        }

        basepath = f"testseries{subfolder}"
        assert os.path.exists(basepath)

        filename = f"{DRAW_STRATEGY}_{BET_STRATEGY}_{DECKS}.csv"
        output_path = f"{basepath}/{filename}"
        if os.path.exists(output_path): continue

        pool = mp.Pool()
        data_for_samples = pool.map(proxylauncher, [{**specs, "sid": sid + 1} for sid in range(SAMPLES)])
        df = pd.DataFrame(data_for_samples)
        df.drop(columns="balanceprogress").to_csv(output_path)

    print(f"DONE AFTER {(time() - st) / 60:.2f} minutes.")


if __name__ == '__main__':
    run_series()
