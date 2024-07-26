## Stock Data Downloader

This script automates the process, efficiently downloading historical and recent OHLCV (Open, High, Low, Close, Volume)
data for your desired stock symbols from Yahoo Finance.

Effortlessly build comprehensive datasets for analysis and upload them to platforms like Kaggle!

**Key Features:**

- **Efficient Downloading:** Downloads minute-level data using `yfinance`. ⚡
- **Historical & Recent Data:** Combines historical and recent data for a complete picture.
- **Multithreading:** Leverages multithreading for faster downloads on multi-core systems. ️
- **Error Handling:** Logs exceptions for troubleshooting.
- **Customizable:** Set the desired stock symbols in a separate `symbols.txt` file.

**Getting Started:**

1. **Prerequisites:**
    - Python 3.x
    - Required libraries: `pandas`, `yfinance`, `pathvalidate`, `numba` (for optimization, optional)
    - Install dependencies with `pip install -r requirements.txt`

2. **Create a `symbols.txt` file:**
    - List each stock symbol you want to download data for, one per line. (e.g., AAPL, TSLA, GOOG)

3. **Schedule Automatic Download:** (optional)

   This script is designed to be run automatically on a schedule using GitHub Actions. To configure this:

    - Go to your GitHub repository settings -> Actions.
    - Create secrets in your repository settings to store sensitive information like your Kaggle credentials (
      KAGGLE_USERNAME and KAGGLE_KEY).
    - Enable the workflow `python-app.yml` to run in your fork's GitHub actions page

**Technical Details (for the curious):**

- The script utilizes [`yfinance`](https://github.com/ranaroussi/yfinance) for data retrieval.
- `pandas` is used for efficient data manipulation and storage as CSV files.
- `pathvalidate` ensures valid filenames for downloaded data.
- The `numba` library (optional) can be used for performance optimization (requires installation).

**Contributing & Further Development:**

- We welcome contributions and suggestions! Feel free to open pull requests.

**Let's automate your stock data collection!**

## Setting Up GitHub Secrets for Kaggle Upload:

**Here's a table outlining the secrets required for uploading your downloaded data to Kaggle:**

| Secret Name     | Description                                        |
|-----------------|----------------------------------------------------|
| KAGGLE_USERNAME | Your Kaggle username                               |
| KAGGLE_KEY      | Your Kaggle API key (create one from your profile) |
| KAGGLE_DATASET  | The remote dataset name                            |


**Note:** These secrets should not be directly added to your code. Instead, create them securely within your GitHub
repository settings -> Secrets.