# === Standard Libraries ===
import os
import glob
from datetime import datetime, timedelta
import warnings

# === Warning Suppression ===
warnings.filterwarnings("ignore", message="no explicit representation of timezones available for np.datetime64")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="matplotlib.backends._backend_tk")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# === Data Handling ===
import polars as pl
import numpy as np

# === ML Tools ===
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras import layers, models, Input
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# === Utilities ===
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

# === Alpaca & Databento ===
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from databento import DBNStore

# === Configuration ===
API_KEY = ""
SECRET = ""

# === Data Extraction Functions ===



def get_historical_data(api_key, secret_key, symbol, startdate, endDate, daily=False):
    client = StockHistoricalDataClient(api_key, secret_key)
    timeframe = TimeFrame.Day if daily else TimeFrame(5, TimeFrameUnit.Minute)

    request = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe,
        start=startdate,
        end=endDate
    )
    bars = client.get_stock_bars(request)
    df = pl.DataFrame([bar.__dict__ for bar in bars[symbol]])
    print("Historical data retrieved")
    df.write_csv("historic/historic_data.csv")
    return df

def filter_rows(df):
    """Filter raw data to only have that are actual trades and have a labelled aggressor"""
    df = df.filter((pl.col('side').is_in(['A', 'B'])) & (pl.col('action') == 'T'))
    return df

def aggregate_rows(df, historic_df, bins=50, lookahead=5):
    """Aggregate rows into candles of a specified size, returning an array of (timestamp, footprint) tuples."""
    # Ensure timestamp is datetime and create 5-min bins
    df = df.with_columns(
    pl.col('ts_event').str.strptime(pl.Datetime)  # assume input string is UTC
    )
    df = df.with_columns(
        pl.col('ts_event').dt.truncate('5m').alias('time_bin')
    )


    results = []  # To store (time_bin, footprint) tuples

    # Group by time_bin and aggregate all columns
    grouped = df.group_by('time_bin').agg(pl.all()).sort('time_bin')

    # Wrap groupby iterator with tqdm for progress bar
    for row in tqdm(grouped.iter_rows(named=True), total=len(grouped), desc="Aggregating footprints"):
        time_bin = row['time_bin']
        group_df = pl.DataFrame(row).select(pl.exclude('time_bin'))

        if group_df.is_empty():
            continue

        # Get low and high prices from historic_df for the time_bin
        low_price = historic_df.filter(pl.col('timestamp') == time_bin)['low'][0]
        high_price = historic_df.filter(pl.col('timestamp') == time_bin)['high'][0]

        after_trend = historic_df.filter(
            (pl.col('timestamp') > time_bin) &
            (pl.col('timestamp') <= time_bin + timedelta(minutes=5 * lookahead))
        )
        label = 0
        prev_trend = 0

        # Define trends
        pos_threshold = 0.005  # e.g., 0.5% move
        neg_threshold = -0.005

        if after_trend.is_empty():
            label = 0  # No data available to judge future trend
        else:
            future_close = after_trend['close'][-1]  # Closing price at end of lookahead
            current_close = historic_df.filter(pl.col('timestamp') == time_bin)['close'][0]

            price_change = (future_close - current_close) / current_close

            if price_change > pos_threshold:
                label = 1
            elif price_change < neg_threshold:
                label = -1
            else:
                label = 0


        previous_trend = historic_df.filter(
            (pl.col('timestamp') < time_bin) &
            (pl.col('timestamp') >= time_bin - timedelta(minutes=5 * lookahead))
        )
        prev_trend = 0
        
        if previous_trend.is_empty():
            prev_trend = 0
        else:
            prev_close = previous_trend['close'][0]
            current_close = historic_df.filter(pl.col('timestamp') == time_bin)['close'][0]

            price_diff = (current_close - prev_close) / prev_close

            if price_diff > pos_threshold:
                prev_trend = 1
            elif price_diff < neg_threshold:
                prev_trend = -1
            else:
                prev_trend = 0

        if low_price == high_price:
            continue  # Skip flat candles

        # Create bin edges for price levels
        bin_edges = np.linspace(low_price, high_price, bins + 1)
        buy_volume = np.zeros(bins)
        sell_volume = np.zeros(bins)

        # Aggregate buy and sell volumes
        for row in group_df.iter_rows(named=True):
            price_level = row['price']
            volume = row['size']
            side = row['side']

            bin_idx = np.searchsorted(bin_edges, price_level, side='right') - 1
            bin_idx = np.clip(bin_idx, 0, bins - 1)

            if side == 'A':
                buy_volume[bin_idx] += volume
            else:
                sell_volume[bin_idx] += volume

        # Calculate delta and create footprint
        delta = buy_volume - sell_volume
        footprint = np.stack((buy_volume, sell_volume, delta), axis=1)
        
        # Append tuple of (time_bin, footprint, label)
        results.append((time_bin, footprint, prev_trend, label))

    # Convert results to a NumPy array with structured dtype
    dtype = [
        ('timestamp', 'datetime64[ms]'),
        ('footprint', float, (bins, 3)),
        ('prev_trend', 'i4'),
        ('label', 'i4')
    ]

    return np.array(results, dtype=dtype)


def train(data):
    X_image = np.stack(data['footprint']).astype(np.float32)

    # in-max per sample normalization:
    footprint_min = X_image.min(axis=(1,2), keepdims=True)
    footprint_max = X_image.max(axis=(1,2), keepdims=True)
    X_image = (X_image - footprint_min) / (footprint_max - footprint_min + 1e-8)

    X_image = X_image[..., np.newaxis]

    X_meta = np.array(data['prev_trend'])

    y = np.array(data['label'])

    y_cat = to_categorical(y + 1, num_classes=3)

    # train/test split

    X_img_train, X_img_test, X_meta_train, X_meta_test, y_train, y_test = train_test_split(
        X_image, X_meta, y_cat, test_size=0.2, random_state=42
    )

    # builing the model

    # Image input: shape (bins, 3, 1)
    image_input = Input(shape=(X_image.shape[1], 3, 1), name='footprint_input')

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Conv2D(64, (3, 1), activation='relu', padding='same')(x)
    x = layers.GlobalMaxPooling2D()(x)

    # Meta input: previous trend
    meta_input = Input(shape=(1,), name='trend_input')
    meta_x = layers.Embedding(3, 3, input_length=1)(meta_input)
    meta_x = layers.Flatten()(meta_x)

    # Combine
    combined = layers.concatenate([x, meta_x])
    dense = layers.Dense(64, activation='relu')(combined)
    output = layers.Dense(3, activation='softmax')(dense)

    model = models.Model(inputs=[image_input, meta_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([-1, 0, 1]), y=y)
    class_weight_dict = {-1: class_weights[0], 0: class_weights[1], 1: class_weights[2]}


    model.fit(
        [X_img_train, X_meta_train], y_train,
        validation_data=([X_img_test, X_meta_test], y_test),
        epochs=20,
        batch_size=32,
        class_weight=class_weight_dict
    )

    
    # Evaluate on test set
    loss, acc = model.evaluate([X_img_test, X_meta_test], y_test)
    print(f"Test Accuracy: {acc:.4f}")

    # Predict class probabilities and convert to labels
    y_pred_probs = model.predict([X_img_test, X_meta_test])
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # === Classification Report ===
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Down (-1)", "Flat (0)", "Up (1)"]))

    # === Confusion Matrix ===
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down (-1)", "Flat (0)", "Up (1)"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # === Export Model ===
    os.makedirs("trained_models", exist_ok=True)
    model.save("trained_models/footprint_model.keras")
    print("Model saved to 'trained_models/footprint_model.keras'")


def visualize_footprint(footprint, filepath, time):
    """
    Save a heatmap showing buy and sell volume side-by-side.
    - Left: buy_volume
    - Right: sell_volume
    Y-axis represents price bins.
    """
    buy = footprint[:, 0]
    sell = footprint[:, 1]

    # Stack horizontally: shape becomes (bins, 2)
    combined = np.stack((buy, sell), axis=1)

    # Log transform to reduce extreme values
    heatmap = np.log1p(combined)

    fig, ax = plt.subplots(figsize=(4, 8))
    im = ax.imshow(heatmap, aspect='auto', cmap='Blues', origin='lower')

    ax.set_title(time)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Buy', 'Sell'])
    ax.set_ylabel("Price Bins")
    ax.set_xlabel("Side")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


def process(symbol, raw_data_path="input", output_path="output", bins=20):
    csv_files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    print("Reading CSV files from:", raw_data_path)
    print(f"Loading {len(csv_files)} CSV files from {raw_data_path}")
    
    df = pl.read_csv(csv_files[0], n_threads=os.cpu_count())
    print(f"Loaded data from {csv_files[0]}, shape: {df.shape}")

    # Download historical OHLCV data
    historic_df = get_historical_data(
        API_KEY, SECRET, symbol,
        startdate=datetime(2025, 5, 8, 7, 55, 0),
        endDate=datetime(2025, 7, 10, 23, 55, 0)
    )

    # Filter + aggregate order flow
    df = filter_rows(df)
    footprints = aggregate_rows(df, historic_df, bins=bins)

    # Save result
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, f"footprints_{symbol}.npy")
    np.save(output_file, footprints)
    print("Saved to:", output_file)

    # Load & summarize
    loaded = np.load(output_file, allow_pickle=True)
    labels = loaded['label']
    prev_trends = loaded['prev_trend']

    label_counts = {label: count for label, count in zip(*np.unique(labels, return_counts=True))}
    trend_counts = {trend: count for trend, count in zip(*np.unique(prev_trends, return_counts=True))}

    print("\nLabel Counts:")
    for k, v in label_counts.items():
        print(f"  Label {k}: {v} samples")

    print("\nPrevious Trend Counts:")
    for k, v in trend_counts.items():
        print(f"  Prev Trend {k}: {v} samples")
    


if __name__ == "__main__":
    process("TQQQ")
    data = np.load("output/footprints_TQQQ.npy", allow_pickle=True)
    #train(data)

    # Output directory
    os.makedirs("exported_visuals", exist_ok=True)

    # Loop through and export selected footprints
    count = 0
    for i, (time_bin, footprint, prev_trend, label) in enumerate(data):
        if prev_trend == -1 and label == 1:
            filepath = f"exported_visuals/footprint_{i}_trend-1_label1.png"
            visualize_footprint(footprint, filepath, time_bin)
            count += 1

    print(f"Exported {count} footprint visualizations with prev_trend = -1 and label = 1.")


    
