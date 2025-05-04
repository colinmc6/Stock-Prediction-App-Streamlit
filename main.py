import os
import sys
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(page_title="SIAM Vehicle Data Dashboard", layout="wide")

# Get the absolute path of the current script (even when run interactively)
script_path = os.path.abspath(sys.argv[0])
script_dir = os.path.dirname(script_path)
script_dir_forward_slash = script_dir.replace('\\', '/')

print(script_dir_forward_slash)  
# Construct full path to the Excel file
file_name = "CMIE_2024.xlsx"
file_path = script_dir_forward_slash +'/' + file_name

print(file_path)  # Print the full path to the file
df_raw = pd.read_excel(file_path, engine='openpyxl', header=None, sheet_name=None)

print(df_raw)

# Create global variables for each sheet
for idx, sheet_name in enumerate(df_raw.keys(), start=1):
    df = pd.DataFrame(df_raw[sheet_name])
    globals()[f"df_{idx}"] = df

def clean_cmie_sheet_final_flexible_test(df):
    import re

    df = df.iloc[2:].copy()
    df.columns = ['Main'] + list(df.columns[1:])
    df.reset_index(drop=True, inplace=True)

    # Add structure columns
    df['Category'] = None
    df['Segment'] = None
    df['Sub-Segment'] = None

    # Define expected structure

    structure = {
        "Passenger Vehicles": {  # Original key
            "Passenger Cars - Upto 5 Seats": [
                "Micro", "Mini", "Compact", "Mid-Size", "Executive", "Premium", "Luxury", "Total Passenger Cars"
            ],
            "Utility Vehicles/ Sports Utility Vehicles": [
                "UVC", "UV1", "UV2", "UV3", "UV4", "UV5", "Total Utility Vehicles"
            ],
            "Vans": [
                "V1", "V2", "Total Vans"
            ]
        },
        "Passenger Vehicles (PVs)": {  # Alias key, same value
            "Passenger Cars - Upto 5 Seats": [
                "Micro", "Mini", "Compact", "Mid-Size", "Executive", "Premium", "Luxury", "Total Passenger Cars"
            ],
            "Utility Vehicles/ Sports Utility Vehicles": [
                "UVC", "UV1", "UV2", "UV3", "UV4", "UV5", "Total Utility Vehicles"
            ],
            "Vans": [
                "V1", "V2", "Total Vans"
            ]
        },
        "Three Wheelers": {
            "Passenger Carrier": [
                "A1", "A2", "Total Passenger Carriers", "E-Rickshaw"
            ],
            "Goods Carrier": [
                "B1", "Total Goods Carrier", "E-Cart"
            ]
        },
        "Two Wheelers": {
            "Scooters": [
                "A1", "A2", "A3", "A4", "A5", "AE1", "AE2", "Total Scooters"
            ],
            "Motorcycles": [
                "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "Total Motorcycles"
            ],
            "Moped": [
                "C1", "Total Mopeds"
            ]
        },
        "Quadricycle": {
            "Quadricycle": ["Total Quadricycle"]
        }
    }


    # Build category matcher (lowercased)
    category_alias_map = {}
    for k in structure.keys():
        category_alias_map[k.lower()] = k
        category_alias_map[f"{k.lower()} (pvs)"] = k  # Handle variants like "Passenger Vehicles (PVs)"
        category_alias_map[f"{k.lower()} (2w)"] = k   # Just in case of Two Wheelers (2W)
        category_alias_map[f"{k.lower()} (3w)"] = k   # For Three Wheelers (3W)

    all_subsegments = [s for segs in structure.values() for subs in segs.values() for s in subs]

    current_category = None
    current_segment = None
    current_subsegment = None
    rows_to_drop = []

    for i, row in df.iterrows():
        main_val = str(row['Main']).strip()
        main_val_lower = main_val.lower()

        # Check for category
        if main_val_lower in category_alias_map:
            current_category = category_alias_map[main_val_lower]
            current_segment = None
            current_subsegment = None
            continue

        # Check for segment
        match = re.match(r"^[A-Z] ?: (.+)", main_val)
        if match:
            current_segment = match.group(1).strip()
            current_subsegment = None
            continue

        # Sub-segment description row (e.g., "Micro: ...")
        if ':' in main_val:
            possible_name = main_val.split(':')[0].strip()
            if possible_name in all_subsegments:
                current_subsegment = possible_name
                rows_to_drop.append(i)
                continue

        # Assign structure to normal rows
        if current_category and current_segment and current_subsegment:
            df.at[i, 'Category'] = current_category
            df.at[i, 'Segment'] = current_segment
            df.at[i, 'Sub-Segment'] = current_subsegment

        # Also assign to total rows
        if 'Total' in main_val and current_category and current_segment and current_subsegment:
            df.at[i, 'Category'] = current_category
            df.at[i, 'Segment'] = current_segment
            df.at[i, 'Sub-Segment'] = current_subsegment

    # Drop subsegment description rows
    df.drop(index=rows_to_drop, inplace=True)

    # Filter rows with complete structure
    df_cleaned = df[df['Category'].notna() & df['Segment'].notna() & df['Sub-Segment'].notna()].copy()

    # Clean numeric columns
    numeric_cols = df_cleaned.columns[1:-3]
    for col in numeric_cols:
        df_cleaned[col] = (
            df_cleaned[col]
            .astype(str)
            .str.replace(",", "")
            .str.replace("-", "")
            .str.strip()
            .replace("NA", None)
            .replace("", None)
        )
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')

    return    df_cleaned

# clean_cmie_sheet_final_flexible_test(df_1)

# prompt: code for downloading clean_cmie_sheet_final_flexible_test(df_1)

def download_dataframe(df, filename="downloaded_dataframe.csv"):
  """Downloads a Pandas DataFrame as a CSV file."""
  df.to_csv(filename, index=False)  # Save to a CSV file (without index)

# Assuming df_cleaned is the cleaned DataFrame from clean_cmie_sheet_final_flexible_test
download_dataframe(clean_cmie_sheet_final_flexible_test(df_1), filename="clean_cmie_sheet_final_flexible_test.csv")

import pandas as pd

# Mapping sheet numbers to month names (df_1 is December, df_2 is November, ..., df_12 is January)
month_mapping = {
    1: 'December',
    2: 'November',
    3: 'October',
    4: 'September',
    5: 'August',
    6: 'July',
    7: 'June',
    8: 'May',
    9: 'April',
    10: 'March',
    11: 'February',
    12: 'January'
}

# Process and tag each cleaned monthly DataFrame
final_monthly_dfs = []

for i in range(1, 13):
    df_month = eval(f"clean_cmie_sheet_final_flexible_test(df_{i})").copy()
    df_month["Month"] = month_mapping[i]
    df_month["Year"] = 2024
    final_monthly_dfs.append(df_month)

# Concatenate all monthly cleaned frames into one
final_2024_df = pd.concat(final_monthly_dfs, ignore_index=True)

# Reorder and rename columns for clarity (optional)
# Rename 'Main' → 'Company', and reorder if needed
final_2024_df.rename(columns={'Main': 'Company'}, inplace=True)

# Optional column order
desired_cols = ['Year', 'Month', 'Company', 'Category', 'Segment', 'Sub-Segment']
# Add dynamic columns (like Production, Sales, Exports) based on available numeric columns
other_cols = [col for col in final_2024_df.columns if col not in desired_cols]
final_2024_df = final_2024_df[desired_cols + other_cols]

# Check result
final_2024_df.head()

final_2024_df

# prompt: code for saving final_2024_df as csv

final_2024_df.to_csv('final_2024_data.csv', index=False)

import pandas as pd

# Load the existing final_2024_data.csv
df = pd.read_csv("final_2024_data.csv")

# Define a function to standardize company names
def standardize_company_name(name):
    if pd.isna(name):
        return None
    # Basic cleaning: strip whitespace, convert to title case
    name = str(name).strip().title()
    # Remove common suffixes like 'Ltd', 'Limited', etc.
    suffixes = [' Ltd', ' Limited', ' Pvt Ltd', ' Private Limited']
    for suffix in suffixes:
        name = name.replace(suffix, '')
    # Placeholder for specific company name mappings (customize as needed)
    company_mapping = {
        'Tata Motors': 'Tata Motors',
        'Tata Motor': 'Tata Motors',  # Handle variations
        'Maruti Suzuki India': 'Maruti Suzuki',
        'Maruti Suzuki': 'Maruti Suzuki',
        'Hyundai Motor India': 'Hyundai Motor',
        'Hyundai Motors': 'Hyundai Motor',
        # Add more mappings based on your data
    }
    # Return standardized name if in mapping, else return cleaned name
    return company_mapping.get(name, name.strip())

# Create the Standard_Company column
df['Standard_Company'] = df['Company'].apply(standardize_company_name)

# Reorder columns to place Standard_Company after Company
desired_cols = ['Year', 'Month', 'Company', 'Standard_Company', 'Category', 'Segment', 'Sub-Segment']
other_cols = [col for col in df.columns if col not in desired_cols]
df = df[desired_cols + other_cols]

# Save the updated CSV
df.to_csv("final_2024_data_updated.csv", index=False)

print("Updated CSV saved as final_2024_data_updated.csv with Standard_Company column.")

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re



# Title and description
st.title("SIAM Vehicle Production, Sales, and Exports Dashboard (2024)")
st.markdown("Explore production, domestic sales, and exports data for Indian automobile manufacturers. Select companies or models, segments, and subsegments to view monthly or yearly trends.")

# Load the CSV file
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("final_2024_data_updated.csv")
        # Clean and preprocess data
        df = df.rename(columns={"1": "Production", "2": "Domestic Sales", "3": "Exports"})
        df["Production"] = pd.to_numeric(df["Production"], errors="coerce")
        df["Domestic Sales"] = pd.to_numeric(df["Domestic Sales"], errors="coerce")
        df["Exports"] = pd.to_numeric(df["Exports"], errors="coerce")
        # Convert Month to datetime for proper sorting
        df["Month"] = pd.to_datetime(df["Month"], format="%B", errors="coerce").dt.strftime("%B")
        # Create a Parent_Company column by cleaning the Company column
        df["Parent_Company"] = df["Company"].apply(
            lambda x: re.sub(r"\s*\(.*?\)|\s*OEM Model#", "", x).strip() if isinstance(x, str) else x
        )
        return df
    except FileNotFoundError:
        st.error("Error: 'final_2024_data_updated.csv' not found. Please ensure the file is in the same directory as the app.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

df = load_data()

# Check if data is loaded successfully
if df.empty:
    st.stop()

# Filter out total rows and keep only company-specific data
company_df = df[df["Standard_Company"].notna() & ~df["Company"].str.contains("Total", case=False)]

# Sidebar for filters
st.sidebar.header("Filter Options")

# View mode selection (Company or Model)
view_mode = st.sidebar.radio("Select View Mode", ["Company View", "Model View"])

# Company or Model selection based on view mode
if view_mode == "Company View":
    companies = sorted(company_df["Parent_Company"].unique())
    selected_companies = st.sidebar.multiselect("Select Companies", companies, default=[companies[0]] if companies else [])
    selected_models = company_df[company_df["Parent_Company"].isin(selected_companies)]["Standard_Company"].unique()
else:
    models = sorted(company_df["Standard_Company"].unique())
    selected_models = st.sidebar.multiselect("Select Models", models, default=[models[0]] if models else [])
    selected_companies = company_df[company_df["Standard_Company"].isin(selected_models)]["Parent_Company"].unique()

# Segment selection
segments = sorted(company_df["Segment"].unique())
selected_segments = st.sidebar.multiselect("Select Segments", segments, default=segments)

# Subsegment selection
subsegments = sorted(company_df["Sub-Segment"].unique())
selected_subsegments = st.sidebar.multiselect("Select Sub-Segments", subsegments, default=subsegments)

# View type selection (Monthly or Yearly)
view_type = st.sidebar.radio("Select View", ["Monthly", "Yearly"])

# Filter data based on selections
filtered_df = company_df[
    (company_df["Standard_Company"].isin(selected_models)) &
    (company_df["Segment"].isin(selected_segments)) &
    (company_df["Sub-Segment"].isin(selected_subsegments))
]

# Handle empty filtered data
if filtered_df.empty:
    st.warning("No data available for the selected filters. Please adjust your selections.")
else:
    # Aggregate data based on view mode and view type
    if view_mode == "Company View":
        group_cols = ["Parent_Company"]
        display_col = "Parent_Company"
    else:
        group_cols = ["Standard_Company"]
        display_col = "Standard_Company"

    if view_type == "Monthly":
        # Group by Month and Company/Model for monthly view
        group_cols = ["Month"] + group_cols
        grouped_df = filtered_df.groupby(group_cols).agg({
            "Production": "sum",
            "Domestic Sales": "sum",
            "Exports": "sum"
        }).reset_index()
        # Sort months in calendar order
        month_order = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        grouped_df["Month"] = pd.Categorical(grouped_df["Month"], categories=month_order, ordered=True)
        grouped_df = grouped_df.sort_values("Month")
        x_col = "Month"
        title_suffix = "Monthly (2024)"
    else:
        # Group by Year and Company/Model for yearly view
        group_cols = ["Year"] + group_cols
        grouped_df = filtered_df.groupby(group_cols).agg({
            "Production": "sum",
            "Domestic Sales": "sum",
            "Exports": "sum"
        }).reset_index()
        x_col = "Year"
        title_suffix = "Yearly (2024)"

    # Create subplots for Production, Domestic Sales, and Exports
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"Production ({title_suffix})",
            f"Domestic Sales ({title_suffix})",
            f"Exports ({title_suffix})"
        ],
        shared_xaxes=True,
        vertical_spacing=0.1
    )

    # Plot Production
    for entity in grouped_df[display_col].unique():
        entity_data = grouped_df[grouped_df[display_col] == entity]
        fig.add_trace(
            go.Bar(
                x=entity_data[x_col],
                y=entity_data["Production"],
                name=entity,
                legendgroup=entity,
                showlegend=True
            ),
            row=1, col=1
        )

    # Plot Domestic Sales
    for entity in grouped_df[display_col].unique():
        entity_data = grouped_df[grouped_df[display_col] == entity]
        fig.add_trace(
            go.Bar(
                x=entity_data[x_col],
                y=entity_data["Domestic Sales"],
                name=entity,
                legendgroup=entity,
                showlegend=False
            ),
            row=2, col=1
        )

    # Plot Exports
    for entity in grouped_df[display_col].unique():
        entity_data = grouped_df[grouped_df[display_col] == entity]
        fig.add_trace(
            go.Bar(
                x=entity_data[x_col],
                y=entity_data["Exports"],
                name=entity,
                legendgroup=entity,
                showlegend=False
            ),
            row=3, col=1
        )

    # Update layout
    fig.update_layout(
        height=800,
        width=1000,
        title_text=f"Vehicle Data by {view_type} {view_mode}",
        showlegend=True,
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )

    # Update axes labels
    fig.update_yaxes(title_text="Production (Vehicles)", row=1, col=1)
    fig.update_yaxes(title_text="Domestic Sales (Vehicles)", row=2, col=1)
    fig.update_yaxes(title_text="Exports (Vehicles)", row=3, col=1)
    fig.update_xaxes(title_text=x_col, row=3, col=1)

    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

    # Display filtered data table
    st.subheader("Filtered Data")
    if view_mode == "Company View":
        display_df = filtered_df[[
            "Year", "Month", "Parent_Company", "Standard_Company", "Segment", "Sub-Segment",
            "Production", "Domestic Sales", "Exports"
        ]].fillna(0)
    else:
        display_df = filtered_df[[
            "Year", "Month", "Standard_Company", "Segment", "Sub-Segment",
            "Production", "Domestic Sales", "Exports"
        ]].fillna(0)
    st.dataframe(display_df)
