import time
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "logs/demo.csv"
REFRESH_SEC = 0.2
WINDOW_SEC = 60  # show last 60 seconds

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("PointNemo – Live Reduction (Demo)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Reduction (dB)")
ax.grid(True)

last_len = 0

while True:
    try:
        df = pd.read_csv(CSV_PATH)
        if len(df) == 0:
            time.sleep(REFRESH_SEC)
            continue

        # Only update when file grows (optional)
        if len(df) == last_len:
            time.sleep(REFRESH_SEC)
            continue
        last_len = len(df)

        # Keep a rolling window
        t = df["t_sec"].to_numpy()
        y = df["reduction_db"].to_numpy()
        t_max = t[-1]
        mask = t >= max(0.0, t_max - WINDOW_SEC)

        line.set_xdata(t[mask])
        line.set_ydata(y[mask])

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(REFRESH_SEC)

    except KeyboardInterrupt:
        break
    except Exception:
        # If file is temporarily locked/being written, just retry
        time.sleep(REFRESH_SEC)
        continue
