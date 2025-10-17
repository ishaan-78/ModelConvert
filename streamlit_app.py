import streamlit as st
import pandas as pd
import numpy as np
from pro_forma_model_LLM import extract_summary_with_ollama, RealEstateProForma, RealEstateSummary, clean_currency
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Real Estate Pro Forma Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .summary-section {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def detect_sheets_and_extract_data(uploaded_file):
    """Detect relevant sheets and extract data from uploaded XLSX file."""
    warnings_list = []
    
    try:
        # Read all sheet names
        xl_file = pd.ExcelFile(uploaded_file)
        sheet_names = xl_file.sheet_names
        
        # Look for monthly cash flow sheet
        mcf_sheet = None
        mcf_keywords = ['monthly', 'cash flow', 'cf', 'monthly cf', 'cashflow', 'monthly cash flow']
        for sheet in sheet_names:
            if any(keyword in sheet.lower() for keyword in mcf_keywords):
                mcf_sheet = sheet
                break
        
        if not mcf_sheet:
            mcf_sheet = sheet_names[0]  # Default to first sheet
            warnings_list.append(f"⚠️ Monthly cash flow sheet not found. Using '{mcf_sheet}' instead.")
        
        # Look for waterfall sheet
        waterfall_sheet = None
        waterfall_keywords = ['waterfall', 'distribution', 'returns', 'waterfall distribution']
        for sheet in sheet_names:
            if any(keyword in sheet.lower() for keyword in waterfall_keywords):
                waterfall_sheet = sheet
                break
        
        if not waterfall_sheet:
            waterfall_sheet = sheet_names[1] if len(sheet_names) > 1 else sheet_names[0]
            warnings_list.append(f"⚠️ Waterfall sheet not found. Using '{waterfall_sheet}' instead.")
        
        # Read the sheets
        mcf_df = pd.read_excel(uploaded_file, sheet_name=mcf_sheet, header=None)
        waterfall_df = pd.read_excel(uploaded_file, sheet_name=waterfall_sheet, header=None)
        
        return mcf_df, waterfall_df, warnings_list, mcf_sheet, waterfall_sheet
        
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, None, [f"❌ Error reading file: {str(e)}"], None, None

def extract_summary_from_xlsx_dataframes(mcf_df, waterfall_df):
    """Extract summary data from XLSX DataFrames using direct parsing."""
    
    def find_row_index(df, search_term):
        """Find row index containing the search term in column 1."""
        for i, row in df.iterrows():
            if pd.notna(row.iloc[1]) and search_term.lower() in str(row.iloc[1]).lower():
                return i
        return None
    
    def extract_monthly_data(df, row_idx, start_col=2, end_col=133):
        """Extract monthly data from a specific row."""
        try:
            data = []
            for col in range(start_col, min(end_col, len(df.columns))):
                val = df.iloc[row_idx, col]
                if pd.notna(val) and val != 0:
                    # Clean the value by removing currency symbols and parentheses
                    val_str = str(val).replace('$', '').replace('(', '-').replace(')', '').replace(',', '')
                    try:
                        data.append(float(val_str))
                    except:
                        data.append(0.0)
                else:
                    data.append(0.0)
            return data
        except:
            return []
    
    # Clean the dataframes
    mcf_df = clean_currency(mcf_df)
    waterfall_df = clean_currency(waterfall_df)
    
    # Find row indices for monthly cash flow data
    gpr_row = find_row_index(mcf_df, "Gross Potential Rent")
    vacancy_row = find_row_index(mcf_df, "Physical Vacancy")
    bad_debt_row = find_row_index(mcf_df, "Bad Debt")
    other_income_row = find_row_index(mcf_df, "Total Other Income")
    egr_row = find_row_index(mcf_df, "Effective Gross Revenue")
    expenses_row = find_row_index(mcf_df, "Total Operating Expenses")
    noi_row = find_row_index(mcf_df, "Net Operating Income")
    interest_row = find_row_index(mcf_df, "Interest Payment")
    principal_row = find_row_index(mcf_df, "Principal Payment")
    cfads_row = find_row_index(mcf_df, "Cash Flow After Debt Service")
    
    # Extract acquisition data
    purchase_price_row = find_row_index(mcf_df, "Purchase Price")
    closing_costs_row = find_row_index(mcf_df, "Closing Costs")
    acquisition_fee_row = find_row_index(mcf_df, "Acquisition Fee")
    working_capital_row = find_row_index(mcf_df, "Working Capital Contributed")
    construction_row = find_row_index(mcf_df, "Construction Budget")
    
    # Extract loan data
    loan_funding_row = find_row_index(mcf_df, "Loan Funding")
    loan_fees_row = find_row_index(mcf_df, "Loan Fees")
    
    # Extract sale data
    sale_price_row = find_row_index(mcf_df, "Sale Price")
    
    # Build summary data
    summary_data = {
        # Monthly cash flow data
        "gross_potential_rent_monthly": extract_monthly_data(mcf_df, gpr_row) if gpr_row else [],
        "vacancy_monthly": [-abs(x) for x in extract_monthly_data(mcf_df, vacancy_row)] if vacancy_row else [],
        "bad_debt_monthly": [-abs(x) for x in extract_monthly_data(mcf_df, bad_debt_row)] if bad_debt_row else [],
        "other_income_monthly": extract_monthly_data(mcf_df, other_income_row) if other_income_row else [],
        "effective_gross_revenue_monthly": extract_monthly_data(mcf_df, egr_row) if egr_row else [],
        "total_expenses_monthly": [-abs(x) for x in extract_monthly_data(mcf_df, expenses_row)] if expenses_row else [],
        "net_operating_income_monthly": extract_monthly_data(mcf_df, noi_row) if noi_row else [],
        "interest_payment_monthly": [-abs(x) for x in extract_monthly_data(mcf_df, interest_row)] if interest_row else [],
        "principal_payment_monthly": [-abs(x) for x in extract_monthly_data(mcf_df, principal_row)] if principal_row else [],
        "cash_flow_after_debt_service_monthly": extract_monthly_data(mcf_df, cfads_row) if cfads_row else [],

        # Acquisition & costs
        "purchase_price": float(mcf_df.iloc[purchase_price_row, 2]) if purchase_price_row else None,
        "closing_costs": float(mcf_df.iloc[closing_costs_row, 2]) if closing_costs_row else None,
        "acquisition_fee": float(mcf_df.iloc[acquisition_fee_row, 2]) if acquisition_fee_row else None,
        "upfront_wc": float(mcf_df.iloc[working_capital_row, 2]) if working_capital_row else None,
        "construction_budget": float(mcf_df.iloc[construction_row, 2]) if construction_row else None,
        "hold_period_months": len(extract_monthly_data(mcf_df, gpr_row)) if gpr_row else 60,

        # Loan information
        "loan_amount": float(mcf_df.iloc[loan_funding_row, 2]) if loan_funding_row else None,
        "loan_fee_pct": 0.0,
        "loan_fee_amount": float(mcf_df.iloc[loan_fees_row, 2]) if loan_fees_row else None,
        "interest_rate_index": 0.0,
        "interest_rate_spread": 0.0,
        "interest_rate": 0.0,
        "io_period": 0,
        "amortization_years": 30,

        # Equity
        "lp_equity": 0.0,
        "gp_equity": 0.0,

        # Growth & expense assumptions
        "market_rent_growth": [],
        "annual_expense_growth": 0.02,
        "property_management_fee_pct": 0.03,

        # Revenue assumptions
        "initial_gpr": 0.0,
        "initial_other_income": 0.0,
        "initial_vacancy_pct": 0.0,
        "bad_debt_pct": 0.0,
        "initial_monthly_expenses": 0.0,

        # Sale
        "exit_cap_rate": 0.06,
        "sale_costs_pct": 0.01,

        # Waterfall
        "preferred_return": 0.08,
        "gp_promote_pct": 0.0,
        "gp_equity_split": 0.0,
    }
    
    # Calculate exit cap rate from sale price
    sale_price = float(mcf_df.iloc[sale_price_row, 2]) if sale_price_row else None
    if sale_price and summary_data["net_operating_income_monthly"]:
        final_noi = summary_data["net_operating_income_monthly"][-1] if summary_data["net_operating_income_monthly"] else 0
        summary_data["exit_cap_rate"] = (final_noi * 12) / sale_price if sale_price > 0 else 0.0
    
    return RealEstateSummary(**summary_data)

def validate_extracted_data(summary_model, warnings_list):
    """Validate extracted data and add warnings for missing critical information."""
    
    # Check for missing critical data
    critical_fields = {
        'purchase_price': 'Purchase Price',
        'loan_amount': 'Loan Amount',
        'gross_potential_rent_monthly': 'Monthly Cash Flow Data'
    }
    
    for field, display_name in critical_fields.items():
        value = getattr(summary_model, field, None)
        if not value or (isinstance(value, list) and len(value) == 0):
            warnings_list.append(f"⚠️ {display_name} not found or empty")
    
    # Check monthly data completeness
    monthly_fields = [
        'gross_potential_rent_monthly', 'net_operating_income_monthly', 
        'cash_flow_after_debt_service_monthly'
    ]
    
    monthly_data_lengths = []
    for field in monthly_fields:
        value = getattr(summary_model, field, [])
        if isinstance(value, list):
            monthly_data_lengths.append(len(value))
    
    if monthly_data_lengths and len(set(monthly_data_lengths)) > 1:
        warnings_list.append("⚠️ Monthly data arrays have different lengths - some data may be missing")
    
    # Check for zero or negative values where they shouldn't be
    if summary_model.purchase_price and summary_model.purchase_price > 0:
        warnings_list.append("⚠️ Purchase Price is positive - this may indicate incorrect data extraction")
    
    return warnings_list

def sanitize_dates(df: pd.DataFrame, date_format: str = "%m/%d/%Y", output_format: str = None) -> pd.DataFrame:
    """
    Parse object columns matching the date_format into datetime64 due to streamlit errors.
    If output_format is given, convert datetime columns to strings with that format.
    Ensures no object dtype with mixed types remain in parsed columns.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            parsed = pd.to_datetime(df[col], format=date_format, errors="coerce")
            # If parsing yields any non-null, treat as date column
            if parsed.notna().sum() > 0:
                if output_format is None:
                    # keep datetime dtype
                    df[col] = parsed
                else:
                    # convert to formatted string
                    df[col] = parsed.dt.strftime(output_format)
    # Replace pandas NaT / NA with None
    df = df.where(pd.notna(df), None)
    return df

def main():
    st.markdown('<h1 class="main-header"> Real Estate Pro Forma Analysis</h1>', unsafe_allow_html=True)
    
    # File upload section
    st.sidebar.header("📁 File Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload XLSX Workbook",
        type=['xlsx'],
        help="Upload an Excel workbook containing monthly cash flow and waterfall data"
    )
    
    if uploaded_file is None:
        st.info("Please upload an XLSX file In the Left-Hand Navigation Bar")
        return
    
    # Load data
    with st.spinner("Loading and processing data..."):
        try:
            if uploaded_file is not None:
                # Use uploaded XLSX file
                mcf_df, waterfall_df, warnings_list, mcf_sheet, waterfall_sheet = detect_sheets_and_extract_data(uploaded_file)
                
                if mcf_df is None:
                    return
                
                # Display warnings
                if warnings_list:
                    st.sidebar.header("⚠️ Warnings")
                    for warning in warnings_list:
                        st.sidebar.warning(warning)
                
                # Show detected sheets
                st.sidebar.success(f"✅ Monthly CF Sheet: {mcf_sheet}")
                st.sidebar.success(f"✅ Waterfall Sheet: {waterfall_sheet}")
                
                # Extract summary data from XLSX
                summary_model = extract_summary_from_xlsx_dataframes(mcf_df, waterfall_df)
                
                # Validate extracted data
                warnings_list = validate_extracted_data(summary_model, warnings_list)
                
                # Build pro forma with DataFrames
                pro_forma = RealEstateProForma.from_dataframes(summary_model, mcf_df, waterfall_df)
                pro_forma.run_model()

                st.success("✅ Data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Cash Flows", "Charts", "Raw Data"])
    
    with tab1:
        display_summary_tab(pro_forma)
    
    with tab2:
        display_cash_flows_tab(pro_forma)
    
    with tab3:
        display_charts_tab(pro_forma)
    
    with tab4:
        display_raw_data_tab(pro_forma)

def display_summary_tab(pro_forma):
    st.header("Investment Summary")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Purchase Price",
            value=f"${abs(pro_forma.purchase_price):,.0f}" if pro_forma.purchase_price else "N/A"
        )
        st.metric(
            label="Loan Amount",
            value=f"${pro_forma.loan_amount:,.0f}" if pro_forma.loan_amount else "N/A"
        )
    
    with col2:
        # Get IRR and equity multiple from return_metrics_df
        project_irr = None
        project_equity_multiple = None
        if pro_forma.return_metrics_df is not None:
            irr_row = pro_forma.return_metrics_df[pro_forma.return_metrics_df['Metric'] == 'Project-Level IRR']
            if not irr_row.empty:
                project_irr = irr_row['Value'].iloc[0]
            equity_row = pro_forma.return_metrics_df[pro_forma.return_metrics_df['Metric'] == 'Project-Level Equity Multiple']
            if not equity_row.empty:
                project_equity_multiple = equity_row['Value'].iloc[0]
        
        st.metric(
            label="Project IRR",
            value=f"{project_irr:.2%}" if project_irr is not None and not np.isnan(project_irr) else "N/A"
        )
        st.metric(
            label="Project Equity Multiple",
            value=f"{project_equity_multiple:.2f}x" if project_equity_multiple is not None and not np.isnan(project_equity_multiple) else "N/A"
        )
    
    with col3:
        # Get exit metrics from sale_metrics_df
        exit_cap_rate = None
        gross_sale_price = None
        if pro_forma.sale_metrics_df is not None:
            cap_rate_row = pro_forma.sale_metrics_df[pro_forma.sale_metrics_df['Metric'] == 'Exit Cap Rate']
            if not cap_rate_row.empty:
                exit_cap_rate = cap_rate_row['Value'].iloc[0]
            sale_price_row = pro_forma.sale_metrics_df[pro_forma.sale_metrics_df['Metric'] == 'Gross Sale Price']
            if not sale_price_row.empty:
                gross_sale_price = sale_price_row['Value'].iloc[0]
        
        st.metric(
            label="Exit Cap Rate",
            value=f"{exit_cap_rate:.2%}" if exit_cap_rate is not None and not np.isnan(exit_cap_rate) else "N/A"
        )
        st.metric(
            label="Gross Sale Price",
            value=f"${gross_sale_price:,.0f}" if gross_sale_price is not None and not np.isnan(gross_sale_price) else "N/A"
        )
    
    with col4:
        st.metric(
            label="Total NOI",
            value=f"${pro_forma.mcf_calculated_df['Net Operating Income'].sum():,.0f}" if len(pro_forma.mcf_calculated_df) > 0 else "N/A"
        )
        st.metric(
            label="Total CFADS",
            value=f"${pro_forma.mcf_calculated_df['Cash Flow After Debt Service'].sum():,.0f}" if len(pro_forma.mcf_calculated_df) > 0 else "N/A"
        )
    
    # Sale Details
    st.subheader("Sale Details and Exit Metrics")
    if pro_forma.sale_metrics_df is not None:
        st.dataframe(pro_forma.sale_metrics_df, width='stretch', hide_index=True)
    else:
        st.warning("Sale metrics data not available")
    
    # Project Returns
    st.subheader("Project-Level Return Metrics")
    if pro_forma.return_metrics_df is not None:
        st.dataframe(pro_forma.return_metrics_df, width='stretch', hide_index=True)
    else:
        st.warning("Return metrics data not available")
    
    # Waterfall Breakdown
    st.subheader("Waterfall Breakdown")
    if pro_forma.waterfall_breakdown_df is not None:
        st.dataframe(pro_forma.waterfall_breakdown_df, width='stretch', hide_index=True)
    else:
        st.warning("Waterfall breakdown data not available")

def display_cash_flows_tab(pro_forma):
    st.header("Monthly Cash Flows")
    
    if len(pro_forma.mcf_calculated_df) == 0:
        st.warning("No cash flow data available.")
        return
    
    # Summary statistics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Months", len(pro_forma.mcf_calculated_df))
        st.metric("Avg Monthly NOI", f"${pro_forma.mcf_calculated_df['Net Operating Income'].mean():,.0f}")
    
    with col2:
        st.metric("Avg Monthly CFADS", f"${pro_forma.mcf_calculated_df['Cash Flow After Debt Service'].mean():,.0f}")
        st.metric("Peak Monthly NOI", f"${pro_forma.mcf_calculated_df['Net Operating Income'].max():,.0f}")
    
    with col3:
        st.metric("Peak Monthly CFADS", f"${pro_forma.mcf_calculated_df['Cash Flow After Debt Service'].max():,.0f}")
        st.metric("Lowest Monthly NOI", f"${pro_forma.mcf_calculated_df['Net Operating Income'].min():,.0f}")
    
    with col4:
        st.metric("Lowest Monthly CFADS", f"${pro_forma.mcf_calculated_df['Cash Flow After Debt Service'].min():,.0f}")
        st.metric("Total NOI", f"${pro_forma.mcf_calculated_df['Net Operating Income'].sum():,.0f}")
    
    # Monthly Cash Flow Table - Show all entries
    st.subheader("Monthly Cash Flow Table")
    
    # Create a copy of the data for display with expenses as positive values
    display_data = pro_forma.mcf_calculated_df.copy()
    
    # Convert negative expense values to positive for display
    expense_columns = ['Vacancy', 'Bad Debt', 'Total Operating Expenses', 'Interest Payment', 'Principal Payment']
    for col in expense_columns:
        if col in display_data.columns:
            display_data[col] = display_data[col].abs()
    
    # Format the data for display
    formatted_data = display_data.copy()
    for col in formatted_data.columns:
        if col != 'Month':
            formatted_data[col] = formatted_data[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    
    st.dataframe(formatted_data, width='stretch', hide_index=True)
    
    # Show total count
    st.info(f"Showing all {len(pro_forma.mcf_calculated_df)} months of cash flow data")

def display_charts_tab(pro_forma):
    st.header("Financial Charts")
    
    if len(pro_forma.mcf_calculated_df) == 0:
        st.warning("No data available for charts.")
        return
    
    # Monthly NOI and CFADS chart
    st.subheader("Monthly NOI and Cash Flow After Debt Service")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pro_forma.mcf_calculated_df['Month'],
        y=pro_forma.mcf_calculated_df['Net Operating Income'],
        mode='lines+markers',
        name='Net Operating Income',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=pro_forma.mcf_calculated_df['Month'],
        y=pro_forma.mcf_calculated_df['Cash Flow After Debt Service'],
        mode='lines+markers',
        name='Cash Flow After Debt Service',
        line=dict(color='#ff7f0e', width=2)
    ))
    
    fig.update_layout(
        title="Monthly Cash Flow Trends",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Revenue breakdown chart
    st.subheader("Revenue Breakdown")
    
    revenue_cols = ['Gross Potential Rent', 'Other Income', 'Effective Gross Revenue']
    revenue_data = pro_forma.mcf_calculated_df[revenue_cols].iloc[:12]  # First 12 months
    
    fig2 = go.Figure()
    
    for col in revenue_cols:
        fig2.add_trace(go.Scatter(
            x=revenue_data.index + 1,
            y=revenue_data[col],
            mode='lines+markers',
            name=col,
            stackgroup='one'
        ))
    
    fig2.update_layout(
        title="Monthly Revenue Breakdown (First 12 Months)",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # Expenses breakdown
    st.subheader("Expenses Breakdown")
    
    expense_cols = ['Total Operating Expenses', 'Interest Payment', 'Principal Payment']
    expense_data = pro_forma.mcf_calculated_df[expense_cols].iloc[:12].abs()  # First 12 months, make positive
    
    fig3 = go.Figure()
    
    for col in expense_cols:
        fig3.add_trace(go.Bar(
            x=expense_data.index + 1,
            y=expense_data[col],
            name=col
        ))
    
    fig3.update_layout(
        title="Monthly Expenses Breakdown (First 12 Months)",
        xaxis_title="Month",
        yaxis_title="Amount ($)",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig3, use_container_width=True)

def display_raw_data_tab(pro_forma):
    st.header("📋 Raw Data")
    
    # Summary model data
    st.subheader("Summary Model Data")
    summary_data = {
        'Field': [
            'Purchase Price', 'Closing Costs', 'Acquisition Fee', 'Working Capital',
            'Construction Budget', 'Loan Amount', 'Interest Rate', 'Amortization Years',
            'Hold Period (Months)'
        ],
        'Value': [
            abs(pro_forma.purchase_price) if pro_forma.purchase_price else None,
            abs(pro_forma.closing_costs) if pro_forma.closing_costs else None,
            abs(pro_forma.acquisition_fee) if pro_forma.acquisition_fee else None,
            abs(pro_forma.upfront_wc) if pro_forma.upfront_wc else None,
            abs(pro_forma.construction_budget) if pro_forma.construction_budget else None,
            pro_forma.loan_amount,
            pro_forma.interest_rate,
            pro_forma.amortization_years,
            pro_forma.hold_period_months
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df['Value'] = summary_df['Value'].apply(lambda x: f"${x:,.0f}" if pd.notna(x) and isinstance(x, (int, float)) else str(x) if pd.notna(x) else "N/A")
    st.dataframe(summary_df, width='stretch', hide_index=True)
    
    # Monthly cash flow data
    st.subheader("Monthly Cash Flow Data")
    if len(pro_forma.mcf_calculated_df) > 0:
        # Create a copy with expenses as positive values for display
        display_raw_data = pro_forma.mcf_calculated_df.copy()
        expense_columns = ['Vacancy', 'Bad Debt', 'Total Operating Expenses', 'Interest Payment', 'Principal Payment']
        for col in expense_columns:
            if col in display_raw_data.columns:
                display_raw_data[col] = display_raw_data[col].abs()
        
        st.dataframe(display_raw_data, width='stretch')
    else:
        st.warning("No monthly cash flow data available.")

if __name__ == "__main__":
    main()
