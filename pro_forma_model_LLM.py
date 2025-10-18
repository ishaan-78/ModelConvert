# file: proforma_with_ollama.py
from typing import Optional, List
from pydantic import BaseModel, Field
# from ollama import chat
import pandas as pd
import numpy_financial as npf
from tabulate import tabulate
from openai import AzureOpenAI

class RealEstateSummary(BaseModel):
    # Monthly Cash Flow Data
    monthly_cash_flows: Optional[List[dict]] = Field(default_factory=list)  # List of monthly cash flow data
    gross_potential_rent_monthly: Optional[List[float]] = Field(default_factory=list)
    vacancy_monthly: Optional[List[float]] = Field(default_factory=list)
    bad_debt_monthly: Optional[List[float]] = Field(default_factory=list)
    other_income_monthly: Optional[List[float]] = Field(default_factory=list)
    effective_gross_revenue_monthly: Optional[List[float]] = Field(default_factory=list)
    total_expenses_monthly: Optional[List[float]] = Field(default_factory=list)
    net_operating_income_monthly: Optional[List[float]] = Field(default_factory=list)
    interest_payment_monthly: Optional[List[float]] = Field(default_factory=list)
    principal_payment_monthly: Optional[List[float]] = Field(default_factory=list)
    cash_flow_after_debt_service_monthly: Optional[List[float]] = Field(default_factory=list)
    
    # Acquisition & costs
    purchase_price: Optional[float]
    closing_costs: Optional[float]
    acquisition_fee: Optional[float]
    upfront_working_capital: Optional[float] = Field(alias="upfront_wc")
    total_construction_budget: Optional[float] = Field(alias="construction_budget")
    hold_period_months: Optional[int]

    # Loan
    loan_amount: Optional[float]
    loan_fee_pct: Optional[float]
    loan_fee_amount: Optional[float]
    interest_rate_index: Optional[float]
    interest_rate_spread: Optional[float]
    interest_rate: Optional[float]  # allow full interest rate if supplied
    interest_only_period_months: Optional[int] = Field(alias="io_period")
    amortization_years: Optional[int]

    # Equity
    lp_equity: Optional[float]
    gp_equity: Optional[float]

    # Growth & expense assumptions
    market_rent_growth: Optional[List[float]] = Field(default_factory=list) # e.g. [yr1, yr2, after]
    annual_expense_growth: Optional[float]
    property_management_fee_pct: Optional[float]

    # Revenue assumptions (optional)
    initial_gpr: Optional[float]
    initial_other_income: Optional[float]
    initial_vacancy_pct: Optional[float]
    bad_debt_pct: Optional[float]
    initial_monthly_expenses: Optional[float]

    # Sale
    exit_cap_rate: Optional[float]
    sale_costs_pct: Optional[float]

    # Waterfall (optional)
    preferred_return: Optional[float]
    gp_promote_pct: Optional[float]
    gp_equity_split: Optional[float]

    model_config = {"validate_by_name": True}


# -------------------------
# Ollama helper to parse messy CSV into validated pydantic model
# -------------------------
# def extract_summary_with_ollama(csv_path: str, model_name: str = "llama3.1") -> RealEstateSummary:
#     """Extract monthly cash flow data from CSV using Ollama or fallback to direct parsing."""
#     try:
#         csv_text = open(csv_path, "r", encoding="utf-8", errors="ignore").read()
#         prompt = f"""
# You are a data extractor for commercial real estate monthly cash flow CSVs.
# Read the CSV text and extract monthly cash flow data from the Monthly CF sheet.
# Return ONLY a JSON object that conforms to the following schema keys. Use numbers for numeric fields and arrays for monthly data:

# LP EQUITY AND GP EQUITY ARE FOUND IN THE WATERFALL SHEET.

# (Advise: 'LP'='Limited Partner' and 'GP'='General Partner')

# MONTHLY CASH FLOW DATA:
# gross_potential_rent_monthly (“Gross Potential Rent”, “GPR”)
# vacancy_monthly (“Physical Vacancy”, “Vacancy Loss”, negative)
# bad_debt_monthly (“Bad Debt”, “Credit Loss”, negative)
# other_income_monthly (“Total Other Income”, “Other Income”)
# effective_gross_revenue_monthly (“Effective Gross Revenue”, “EGR”)
# total_expenses_monthly (“Total Expenses”, “Operating Expenses”, negative)
# net_operating_income_monthly (“Net Operating Income”, “NOI”)
# interest_payment_monthly (“Interest Payment”, “Debt Interest”, negative)
# principal_payment_monthly (“Principal Payment”, “Debt Principal”, negative)
# cash_flow_after_debt_service_monthly (“Cash Flow After Debt Service”, “CFADS”)

# ACQUISITION & COSTS:
# purchase_price (“Purchase Price”, “Acquisition Price”, “Property Purchase”)
# closing_costs (“Closing Costs”, “Transaction Costs”)
# acquisition_fee (“Acquisition Fee”, “Sponsor Fee”)
# upfront_wc (“Working Capital Contributed”, “Initial Working Capital”, “Upfront WC”)
# construction_budget (“Construction Expenses”, “Total Construction Budget”, “CapEx Budget”)
# hold_period_months (“Hold Period”, “Investment Duration”)
# LOAN INFORMATION:
# loan_amount (“Loan Funding”, “Debt Proceeds”, “Senior Loan”)
# loan_fee_amount (“Loan Fees”, “Financing Fees”, “Origination Fees”)
# interest_rate (“Interest Rate”, “Loan Rate”)
# io_period (“Interest-Only Period”, “IO Period”)
# amortization_years (“Amortization Term”, “Loan Amortization”)

# EQUITY & SALE:
# project_irr (“Project IRR”, “Project-level IRR”)
# lp_equity (“LP Equity”, “Limited Partner Equity”, “Investor Equity”)
# gp_equity (“GP Equity”, “General Partner Equity”, “Sponsor Equity”)
# exit_cap_rate (“Exit Cap Rate”, “Terminal Cap Rate”) (NOT TO BE CONFUSED WITH GOING-IN CAP RATE OR OTHER CAP RATES)
# sale_costs_pct (“Costs of Sale”, “Disposition Costs”, “Sales Costs Percentage”)

# If you cannot find a value, put null. Extract monthly data as arrays of numbers.
# CSV:
# {csv_text[:6000]}
# """
#         response = chat(
#             messages=[{"role": "user", "content": prompt}],
#             model=model_name,
#             format=RealEstateSummary.model_json_schema()
#         )

#         # Ollama returns JSON in response.message.content per your example
#         raw = response.message.content
#         # Validate/parse into the pydantic model
#         summary_model = RealEstateSummary.model_validate_json(raw)
#         return summary_model
#     except Exception as e:
#         raise Exception(f"Ollama extraction failed: {e}")
    

def extract_summary_with_azure_from_text(csv_text: str) -> RealEstateSummary:
    """
    Use Azure OpenAI (via openai.OpenAI client) to parse CSV text into the RealEstateSummary JSON.
    Assumes Streamlit (or environment) has the Azure secrets set:
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_ENDPOINT  (e.g. https://your-resource-name.openai.azure.com)
      - AZURE_OPENAI_DEPLOYMENT (deployment name)
      - AZURE_OPENAI_API_VERSION (optional; default will be used if not set)
    """
    # Build prompt (similar to your Ollama prompt)
    prompt = f"""
You are a data extractor for commercial real estate monthly cash flow CSVs.
Read the CSV text and extract the requested values and monthly arrays.
Return ONLY a single JSON object that complies with the following pydantic schema keys:
{json.dumps(RealEstateSummary.model_json_schema(), indent=0)}

CSV:
{csv_text[:15000]}
"""

    # Read Azure credentials from environment (Streamlit will set these via st.secrets)
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")  # optional

    if not (api_key and endpoint and deployment):
        raise RuntimeError("Azure OpenAI credentials (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) must be set in environment.")

    # Build client (use base_url that points to your resource path)
    base_url = endpoint.rstrip("/") + "/openai/v1/"
    client = OpenAI(api_key=api_key, base_url=base_url)  # matches Azure examples

    # Call chat/completions (use deployment name as model)
    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a JSON extractor that outputs only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=4000,
    )

    # Get text result (new SDK uses choices[0].message.content)
    raw_text = None
    try:
        raw_text = resp.choices[0].message.content
    except Exception:
        # Fallback: try response.output_text or .choices[0].text
        raw_text = getattr(resp, "output_text", None) or (resp.choices[0].get("text") if resp.choices else None)

    if not raw_text:
        raise RuntimeError("Azure OpenAI returned no usable text. Check deployment and model response.")

    # Validate JSON via pydantic
    # If model returned JSON wrapped in markdown/code fences, strip them
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        # remove triple backticks and optional 'json'
        cleaned = "\n".join(cleaned.splitlines()[1:-1])

    # If Azure returns a JSON object string inside other text, attempt to find the first { ... }
    if "{" in cleaned and "}" in cleaned:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        cleaned = cleaned[first:last+1]

    # Validate / parse using pydantic
    summary_model = RealEstateSummary.model_validate_json(cleaned)
    return summary_model




def clean_currency(df):
    """Strip $, commas, and parentheses from all string cells and convert to float."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r'[\$,]', '', regex=True)
                .str.replace(r'\((.*?)\)', r'-\1', regex=True)
            )
            # Convert to numeric, keeping non-numeric values as strings
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Keep non-numeric values as strings if conversion fails
                pass
    return df

class RealEstateProForma:
    def __init__(self, summary_model: RealEstateSummary, mcf_path: str, waterfall_path: str):
        # store raw sources
        self.summary_model = summary_model
        self.mcf_df_raw = pd.read_excel(mcf_path, header=None)
        self.mcf_df_raw = clean_currency(self.mcf_df_raw)
        self.waterfall_df = pd.read_excel(waterfall_path, header=None)
        self.waterfall_df = clean_currency(self.waterfall_df)

        # placeholders
        self.sources_uses_df = None
        self.sale_metrics_df = None
        self.return_metrics_df = None
        self.waterfall_breakdown_df = None
        self.mcf_calculated_df = None

        self._load_assumptions_from_model()

    @classmethod
    def from_summary_model(cls, summary_model: RealEstateSummary, mcf_path: str, waterfall_path: str):
        return cls(summary_model, mcf_path, waterfall_path)
    
    @classmethod
    def from_dataframes(cls, summary_model: RealEstateSummary, mcf_df: pd.DataFrame, waterfall_df: pd.DataFrame):
        """Create instance from DataFrames instead of file paths."""
        instance = cls.__new__(cls)
        instance.summary_model = summary_model
        instance.mcf_df_raw = mcf_df.copy()
        instance.mcf_df_raw = clean_currency(instance.mcf_df_raw)
        instance.waterfall_df = waterfall_df.copy()
        instance.waterfall_df = clean_currency(instance.waterfall_df)
        
        # placeholders
        instance.sources_uses_df = None
        instance.sale_metrics_df = None
        instance.return_metrics_df = None
        instance.waterfall_breakdown_df = None
        instance.mcf_calculated_df = None
        
        instance._load_assumptions_from_model()
        return instance

    def _get_model_value(self, attr: str, default=None):
        v = getattr(self.summary_model, attr, None)
        if v is None:
            return default
        return v

    def _load_assumptions_from_model(self):
        """Load inputs from pydantic model, with safe fallbacks and simple derived calculations."""
        s = self.summary_model

        # Direct mapping
        self.purchase_price = self._get_model_value("purchase_price", 0.0)
        self.closing_costs = self._get_model_value("closing_costs", 0.0)
        self.acquisition_fee = self._get_model_value("acquisition_fee", 0.0)
        self.upfront_wc = self._get_model_value("upfront_working_capital", self._get_model_value("upfront_wc", 0.0))
        self.construction_budget = self._get_model_value("total_construction_budget", self._get_model_value("construction_budget", 0.0))
        self.hold_period_months = int(self._get_model_value("hold_period_months", 60))

        # Loan
        self.loan_amount = self._get_model_value("loan_amount", 0.0)
        self.loan_fee_pct = self._get_model_value("loan_fee_pct", 0.0)
        # prefer explicit loan_fee_amount if provided
        self.loan_fee_amount = self._get_model_value("loan_fee_amount", None)
        if self.loan_fee_amount is None and self.loan_amount and self.loan_fee_pct:
            self.loan_fee_amount = self.loan_amount * self.loan_fee_pct

        # Interest: prefer full rate if given, otherwise sum index + spread
        self.interest_rate = self._get_model_value("interest_rate", None)
        if self.interest_rate is None:
            idx = self._get_model_value("interest_rate_index", 0.0) or 0.0
            spr = self._get_model_value("interest_rate_spread", 0.0) or 0.0
            self.interest_rate = idx + spr

        self.io_period = int(self._get_model_value("interest_only_period_months", self._get_model_value("io_period", 0) or 0))
        self.amortization_years = int(self._get_model_value("amortization_years", 30))

        # Equity
        self.lp_equity = self._get_model_value("lp_equity", 0.0)
        self.gp_equity = self._get_model_value("gp_equity", 0.0)
        self.total_equity = (self.lp_equity or 0.0) + (self.gp_equity or 0.0)

        # Load monthly cash flow data if available
        self._load_monthly_cash_flow_data()

        # Revenue & Expense from summary model if present, otherwise fall back to raw MCF
        self.initial_gpr = self._get_model_value("initial_gpr", None)
        self.initial_other_income = self._get_model_value("initial_other_income", None)
        self.initial_vacancy_pct = self._get_model_value("initial_vacancy_pct", None)
        self.bad_debt_pct = self._get_model_value("bad_debt_pct", None)
        self.initial_monthly_expenses = self._get_model_value("initial_monthly_expenses", None)

        # If any of those are missing, try reading from the MCF raw file using your previous heuristics
        try:
            if self.initial_gpr is None:
                self.initial_gpr = float(self.mcf_df_raw.iloc[5, 3])
        except Exception:
            self.initial_gpr = self.initial_gpr or 0.0

        try:
            if self.initial_other_income is None:
                # safe attempt to sum possible other income rows (if present)
                vals = []
                for r in [10,11,12,13,14]:
                    try:
                        vals.append(float(self.mcf_df_raw.iloc[r, 3]))
                    except Exception:
                        pass
                self.initial_other_income = sum(vals) if vals else (self.initial_other_income or 0.0)
        except Exception:
            self.initial_other_income = self.initial_other_income or 0.0

        try:
            if self.initial_vacancy_pct is None:
                self.initial_vacancy_pct = abs(float(self.mcf_df_raw.iloc[6, 3]) / (self.initial_gpr or 1.0))
            if self.bad_debt_pct is None:
                self.bad_debt_pct = abs(float(self.mcf_df_raw.iloc[7, 3]) / (self.initial_gpr or 1.0))
        except Exception:
            self.initial_vacancy_pct = self.initial_vacancy_pct or 0.0
            self.bad_debt_pct = self.bad_debt_pct or 0.0

        try:
            if self.initial_monthly_expenses is None:
                self.initial_monthly_expenses = abs(sum(self.mcf_df_raw.iloc[i, 3] for i in range(19, 29) if not pd.isna(self.mcf_df_raw.iloc[i, 3])))
        except Exception:
            self.initial_monthly_expenses = self.initial_monthly_expenses or 0.0

        # Growths
        mrg = self._get_model_value("market_rent_growth", [])
        # Expect array [yr1, yr2, after] or fallback to separate fields
        if mrg and isinstance(mrg, list):
            self.annual_rent_growth_yr1 = mrg[0] if len(mrg) > 0 else 0.0
            self.annual_rent_growth_yr2 = mrg[1] if len(mrg) > 1 else 0.0
            self.annual_rent_growth_after = mrg[2] if len(mrg) > 2 else (mrg[1] if len(mrg) > 1 else 0.0)
        else:
            self.annual_rent_growth_yr1 = 0.0
            self.annual_rent_growth_yr2 = 0.0
            self.annual_rent_growth_after = 0.0

        self.annual_expense_growth = float(self._get_model_value("annual_expense_growth", 0.02))
        self.mgmt_fee_pct = float(self._get_model_value("property_management_fee_pct", 0.03))

        # Sale and waterfall
        self.exit_cap_rate = float(self._get_model_value("exit_cap_rate", 0.01))
        self.sale_costs_pct = float(self._get_model_value("sale_costs_pct", 0.01))
        self.pref_return = float(self._get_model_value("preferred_return", 0.08))
        self.gp_promote_pct = float(self._get_model_value("gp_promote_pct", 0.0))
        self.gp_equity_split = float(self._get_model_value("gp_equity_split", 0.0))

    def _load_monthly_cash_flow_data(self):
        """Load monthly cash flow data from the model if available."""
        # Store monthly data from the model
        self.gross_potential_rent_monthly = self._get_model_value("gross_potential_rent_monthly", [])
        self.vacancy_monthly = self._get_model_value("vacancy_monthly", [])
        self.bad_debt_monthly = self._get_model_value("bad_debt_monthly", [])
        self.other_income_monthly = self._get_model_value("other_income_monthly", [])
        self.effective_gross_revenue_monthly = self._get_model_value("effective_gross_revenue_monthly", [])
        self.total_expenses_monthly = self._get_model_value("total_expenses_monthly", [])
        self.net_operating_income_monthly = self._get_model_value("net_operating_income_monthly", [])
        self.interest_payment_monthly = self._get_model_value("interest_payment_monthly", [])
        self.principal_payment_monthly = self._get_model_value("principal_payment_monthly", [])
        self.cash_flow_after_debt_service_monthly = self._get_model_value("cash_flow_after_debt_service_monthly", [])

    # --- keep the rest of your original methods, slightly adapted to reference these attributes ---
    def run_model(self):
        self._calculate_monthly_cash_flows()
        self._calculate_sale_and_exit_metrics()
        self._calculate_project_returns()
        self._build_output_tables()

    def _calculate_monthly_cash_flows(self):
        # Check if we have monthly data from the model
        if (self.gross_potential_rent_monthly and 
            self.net_operating_income_monthly and 
            self.cash_flow_after_debt_service_monthly):
            # Use the monthly data from the model
            months = list(range(0, len(self.gross_potential_rent_monthly)))
            
            self.mcf_calculated_df = pd.DataFrame({
                'Month': months,
                'Gross Potential Rent': self.gross_potential_rent_monthly,
                'Vacancy': self.vacancy_monthly if self.vacancy_monthly else [0] * len(months),
                'Bad Debt': self.bad_debt_monthly if self.bad_debt_monthly else [0] * len(months),
                'Other Income': self.other_income_monthly if self.other_income_monthly else [0] * len(months),
                'Effective Gross Revenue': self.effective_gross_revenue_monthly if self.effective_gross_revenue_monthly else [0] * len(months),
                'Total Operating Expenses': self.total_expenses_monthly if self.total_expenses_monthly else [0] * len(months),
                'Net Operating Income': self.net_operating_income_monthly,
                'Interest Payment': self.interest_payment_monthly if self.interest_payment_monthly else [0] * len(months),
                'Principal Payment': self.principal_payment_monthly if self.principal_payment_monthly else [0] * len(months),
                'Cash Flow After Debt Service': self.cash_flow_after_debt_service_monthly
            })
            
            # Calculate ending loan balance from principal payments
            if self.principal_payment_monthly:
                self.ending_loan_balance = (self.loan_amount or 0.0) + sum(self.principal_payment_monthly)
            else:
                self.ending_loan_balance = self.loan_amount or 0.0
        else:
            # Fall back to calculated monthly cash flows
            months = list(range(1, self.hold_period_months + 1))
            gpr_list, vacancy_list, bad_debt_list, other_income_list, egr_list = [], [], [], [], []
            expenses_list, noi_list, interest_list, principal_list, cfa_ds_list = [], [], [], [], []

            loan_balance = self.loan_amount or 0.0
            current_gpr = self.initial_gpr or 0.0
            current_other_income = self.initial_other_income or 0.0
            current_expenses = self.initial_monthly_expenses or 0.0

            for m in months:
                year = (m - 1) // 12 + 1
                if m > 1:
                    if year == 1:
                        growth_rate = (1 + (self.annual_rent_growth_yr1 or 0.0))**(1/12)
                    elif year == 2:
                        growth_rate = (1 + (self.annual_rent_growth_yr2 or 0.0))**(1/12)
                    else:
                        growth_rate = (1 + (self.annual_rent_growth_after or 0.0))**(1/12)
                    current_gpr *= growth_rate
                    current_other_income *= growth_rate

                gpr_list.append(current_gpr)
                other_income_list.append(current_other_income)

                vacancy = current_gpr * (self.initial_vacancy_pct or 0.0)
                bad_debt = current_gpr * (self.bad_debt_pct or 0.0)
                vacancy_list.append(-vacancy)
                bad_debt_list.append(-bad_debt)

                egr = current_gpr - vacancy - bad_debt + current_other_income
                egr_list.append(egr)

                if m > 1 and (m - 1) % 12 == 0:
                    current_expenses *= (1 + (self.annual_expense_growth or 0.0))

                mgmt_fee = egr * (self.mgmt_fee_pct or 0.0) / 12
                total_expenses = (current_expenses - (self.initial_monthly_expenses * ((self.mgmt_fee_pct or 0.0)/12))) + mgmt_fee
                expenses_list.append(-total_expenses)

                noi = egr - total_expenses
                noi_list.append(noi)

                interest = 0.0
                principal = 0.0
                if m <= (self.io_period or 0):
                    interest = loan_balance * (self.interest_rate or 0.0) / 12
                else:
                    pmt = -npf.pmt((self.interest_rate or 0.0) / 12, (self.amortization_years or 30) * 12, loan_balance)
                    # ipmt/ppmt requires integer period input; use (m - io_period)
                    period = m - (self.io_period or 0)
                    try:
                        interest = npf.ipmt((self.interest_rate or 0.0) / 12, period, (self.amortization_years or 30) * 12, loan_balance)
                        principal = npf.ppmt((self.interest_rate or 0.0) / 12, period, (self.amortization_years or 30) * 12, loan_balance)
                        loan_balance += principal  # principal is negative, so this reduces the balance
                    except Exception:
                        interest = loan_balance * (self.interest_rate or 0.0) / 12

                interest_list.append(interest)
                principal_list.append(principal)
                cfa_ds_list.append(noi + interest + principal)

            self.ending_loan_balance = loan_balance

            self.mcf_calculated_df = pd.DataFrame({
                'Month': months,
                'Gross Potential Rent': gpr_list,
                'Vacancy': vacancy_list,
                'Bad Debt': bad_debt_list,
                'Other Income': other_income_list,
                'Effective Gross Revenue': egr_list,
                'Total Operating Expenses': expenses_list,
                'Net Operating Income': noi_list,
                'Interest Payment': interest_list,
                'Principal Payment': principal_list,
                'Cash Flow After Debt Service': cfa_ds_list
            })

    def _calculate_sale_and_exit_metrics(self):
        # Use exit T12 NOI
        if len(self.mcf_calculated_df) >= 12:
            t12_noi_exit = self.mcf_calculated_df['Net Operating Income'].iloc[-12:].sum()
        else:
            t12_noi_exit = self.mcf_calculated_df['Net Operating Income'].sum()
        self.sale_price = t12_noi_exit / (self.exit_cap_rate or 1.0)
        self.costs_of_sale = self.sale_price * (self.sale_costs_pct or 0.0)
        self.net_sale_proceeds = self.sale_price - self.costs_of_sale
        self.net_proceeds_to_equity = self.net_sale_proceeds - (self.ending_loan_balance or 0.0)

    def _calculate_project_returns(self):
        initial_investment = - (self.total_equity or 0.0)
        interim_cfs = self.mcf_calculated_df['Cash Flow After Debt Service'].tolist()
        levered_cfs = [initial_investment] + interim_cfs
        levered_cfs[-1] += self.net_proceeds_to_equity or 0.0

        try:
            irr_monthly = npf.irr(levered_cfs)
            self.project_irr = (irr_monthly or 0.0) * 12
        except Exception:
            self.project_irr = 0.0

        try:
            num = sum(cf for cf in levered_cfs if cf > 0)
            denom = abs(sum(cf for cf in levered_cfs if cf < 0))
            self.project_em = num / denom if denom != 0 else 0.0
        except Exception:
            self.project_em = 0.0

        avg_annual_cfads = self.mcf_calculated_df['Cash Flow After Debt Service'].sum() / (self.hold_period_months / 12)
        self.avg_coc = avg_annual_cfads / (self.total_equity or 1.0)

    def _build_output_tables(self):
        """Constructs the final DataFrames for display."""
        # --- Sources (each row has same length) ---
        sources = pd.DataFrame({
            'Sources': ['Acquisition Loan', 'LP Equity', 'GP Equity', 'Total Sources'],
            'Amount': [
                self.loan_amount or 0.0,
                self.lp_equity or 0.0,
                self.gp_equity or 0.0,
                (self.loan_amount or 0.0) + (self.total_equity or 0.0)
            ]
        })

        # --- Uses (each row has same length) ---
        total_uses = (
            (self.purchase_price or 0.0)
            + (self.acquisition_fee or 0.0)
            + (self.closing_costs or 0.0)
            + (self.upfront_wc or 0.0)
            + (self.loan_fee_amount or 0.0)
            + (self.construction_budget or 0.0)
        )

        uses = pd.DataFrame({
            'Uses': [
                'Purchase Price', 'Acquisition Fee', 'Closing Costs',
                'Up-Front Working Capital', 'Loan Fees', 'Construction Budget', 'Total Uses'
            ],
            'Amount_Uses': [
                self.purchase_price or 0.0,
                self.acquisition_fee or 0.0,
                self.closing_costs or 0.0,
                self.upfront_wc or 0.0,
                self.loan_fee_amount or 0.0,
                self.construction_budget or 0.0,
                total_uses
            ]
        })

        self.sources_uses_df = sources
        self.uses_df = uses

        # --- Sale Details ---
        self.sale_metrics_df = pd.DataFrame([
            {'Metric': 'Exit T-12 NOI', 'Value': self.mcf_calculated_df['Net Operating Income'].iloc[-12:].sum() if len(self.mcf_calculated_df) >= 12 else self.mcf_calculated_df['Net Operating Income'].sum()},
            {'Metric': 'Exit Cap Rate', 'Value': self.exit_cap_rate},
            {'Metric': 'Gross Sale Price', 'Value': self.sale_price},
            {'Metric': 'Costs of Sale', 'Value': -self.costs_of_sale},
            {'Metric': 'Net Sale Price', 'Value': self.net_sale_proceeds},
            {'Metric': 'Loan Payoff', 'Value': -self.ending_loan_balance},
            {'Metric': 'Net Proceeds to Equity', 'Value': self.net_proceeds_to_equity}
        ])

        # --- Return Metrics ---
        self.return_metrics_df = pd.DataFrame([
            {'Metric': 'Project-Level IRR', 'Value': self.project_irr},
            {'Metric': 'Project-Level Equity Multiple', 'Value': self.project_em},
            {'Metric': 'Average Cash-on-Cash Return', 'Value': self.avg_coc}
        ])

        # --- Waterfall (keeps your existing logic) ---
        lp_share = 1 - (self.gp_equity_split or 0.0)

        lp_capital_returned = min(self.net_proceeds_to_equity or 0.0, self.lp_equity or 0.0)
        gp_capital_returned = min(max(0, (self.net_proceeds_to_equity or 0.0) - lp_capital_returned), self.gp_equity or 0.0)
        remaining_profit = (self.net_proceeds_to_equity or 0.0) - lp_capital_returned - gp_capital_returned

        lp_pref_due = (self.lp_equity or 0.0) * (((1 + (self.pref_return or 0.0))**(self.hold_period_months/12)) - 1)
        lp_pref_paid = min(remaining_profit, lp_pref_due)
        remaining_profit -= lp_pref_paid

        promote_to_gp = remaining_profit * (self.gp_promote_pct or 0.0)
        profit_to_lp = remaining_profit * (1 - (self.gp_promote_pct or 0.0)) * lp_share
        profit_to_gp_pro_rata = remaining_profit * (1 - (self.gp_promote_pct or 0.0)) * (self.gp_equity_split or 0.0)

        total_lp_dist = (self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * lp_share
                        + lp_capital_returned + lp_pref_paid + profit_to_lp)
        total_gp_dist = (self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * (self.gp_equity_split or 0.0)
                        + gp_capital_returned + promote_to_gp + profit_to_gp_pro_rata)

        self.waterfall_breakdown_df = pd.DataFrame({
            'Distribution Tier': ['Interim Cash Flow', 'Return of Capital', 'Preferred Return', 'Promote Split', 'Total Distributions'],
            'To LP': [
                self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * lp_share,
                lp_capital_returned, lp_pref_paid, profit_to_lp, total_lp_dist
            ],
            'To GP': [
                self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * (self.gp_equity_split or 0.0),
                gp_capital_returned, 0.0, promote_to_gp + profit_to_gp_pro_rata, total_gp_dist
            ]
        })


if __name__ == "__main__":
    MCF_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Monthly CF.csv'
    WATERFALL_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Waterfall.csv'

    # 1) Use Ollama + pydantic schema to extract monthly cash flow data
    print("Extracting monthly cash flow data with ollama...")
    # summary_model = extract_summary_with_ollama(MCF_FILE, model_name="llama3.1")
    summary_model = extract_summary_with_azure_from_text(MCF_FILE)

    # 2) Build the pro forma from the validated model
    pro_forma = RealEstateProForma.from_summary_model(summary_model, MCF_FILE, WATERFALL_FILE)
    pro_forma.run_model()

    # 3) Display
    pro_forma.display_outputs()
