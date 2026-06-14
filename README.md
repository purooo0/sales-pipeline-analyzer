# Sales Pipeline Analyzer

Sales Pipeline Analyzer is a Streamlit dashboard for analyzing sales opportunity pipeline data from Excel files. It was built to make recurring sales reports easier to explore without manually editing Python code every time the Excel format changes.

The app detects the uploaded workbook structure, maps columns into a standard sales pipeline schema, and generates summary tables plus professional charts for contract value, win rate, sales cycle, product mix, open pipeline, and account manager performance.

## Highlights

- Adaptive Excel import that scans workbook sheets, detects the header row, and maps columns automatically
- Manual column mapping fallback for low-confidence or unusual Excel formats
- Sales pipeline summary: total opportunities, won opportunities, contract value, and conversion rate
- Product mix donut charts for all-stage and won-only contract value
- Quarterly contract value comparison between all-stage and won-only opportunities
- Opportunity count pivot table by year, quarter, product type, and stage
- Top and bottom closed-won opportunities by close year
- Open pipeline view grouped by future close year
- Average sales cycle analysis by Line of Business (LoB)
- Average stage-to-close duration by LoB, product type, and sales engineer
- Win rate by LoB
- Top account manager views by conversion rate, total contract value, and total deals
- Professional chart styling with a clean dashboard palette and Telkomsel red accent

## Adaptive Excel Import

The original version expected a stable Excel layout. This version adds an adaptive import layer so non-technical users can upload reports with common layout changes.

The importer can handle:

- Data table is not on the first sheet
- Header starts below title, notes, or metadata rows
- Column order changes
- Column names use Indonesian or English variants
- Extra sheets such as summary, notes, or export metadata

Example aliases that can be detected:

- `Nilai Kontrak` -> `Schedule Amount`
- `Tanggal Closing` -> `Close Date`
- `Tanggal Dibuat` -> `Created Date`
- `Segmen Industri` -> `Industry Segment`
- `Nama Peluang` -> `Opportunity Name`
- `Nama AM` -> `AM Name`

If the app is unsure, the user can still review and correct the mapping from the sidebar.

## Tech Stack

- Python
- Streamlit
- pandas
- NumPy
- Matplotlib
- Seaborn
- openpyxl

## Project Structure

```text
.
├── app.py              # Streamlit UI, upload flow, filters, and chart rendering
├── excel_adapter.py    # Adaptive Excel sheet/header/column detection
├── processing.py       # Data preprocessing and analysis logic
├── requirements.txt    # Python dependencies
├── test_data/          # Dummy Excel files for local testing
└── README.md
```

## Setup

Use Python 3.9-3.12. Python 3.12 is recommended because the pinned data-science dependencies install more reliably on it.

```bash
cd /Users/adheliaputri/Documents/Sales-Pipeline-Analyzer
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `python3.12` is not available on macOS, install it with Homebrew:

```bash
brew install python@3.12
$(brew --prefix python@3.12)/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

Then open the local URL printed in the terminal.

## Usage

1. Upload an Excel `.xlsx` file.
2. Review the detected sheet, header row, and automatic column mapping.
3. Adjust column mapping only if needed.
4. Select LoB filters and analysis features from the sidebar.
5. Click `Run` to generate the dashboard.

## Sample Data

Dummy Excel files are available for testing:

- `test_data/sales_pipeline_sample_standard.xlsx`
- `test_data/sales_pipeline_sample_messy_multisheet.xlsx`

The messy multi-sheet sample is useful for testing the adaptive import flow because it contains a non-data summary sheet and a data sheet with report title rows above the actual header.

## Expected Data Fields

The app works best when the uploaded Excel file contains these business fields or recognizable aliases:

- Opportunity and account: `Opportunity Name`, `Account Name`
- Deal status: `Stage`
- Value: `Schedule Amount`
- Dates: `Schedule Date`, `Created Date`, `Close Date`, `Last Stage Change Date`
- Classification: `Industry Segment`, `Pilar`
- People: `AM Name`, `Opportunity Owner`

If a required field is missing, the related chart or table is skipped and the app shows a warning.

## Notes

- Large Excel files may take longer to process.
- Uploaded data is cached during the Streamlit session for faster iteration.
- For best results, date columns should use valid Excel date or ISO date formats.
- This project uses dummy/sample data only; do not commit confidential internship or company data.
