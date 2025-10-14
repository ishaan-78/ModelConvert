# file: proforma_with_ollama.py
from typing import Optional, List
from pydantic import BaseModel, Field
from ollama import chat
import pandas as pd
import numpy_financial as npf
from tabulate import tabulate
import json

# -------------------------
# Pydantic schema for summary
# -------------------------
class RealEstateSummary(BaseModel):
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

    class Config:
        # allow population by both alias and field name
        allow_population_by_field_name = True


# -------------------------
# Ollama helper to parse messy CSV into validated pydantic model
# -------------------------
def extract_summary_with_ollama(csv_path: str, model_name: str = "llama3.1") -> RealEstateSummary:
    csv_text = open(csv_path, "r", encoding="utf-8", errors="ignore").read()
    prompt = f"""
You are a data extractor for commercial real estate pro forma CSVs.
Read the CSV text (it can contain merged cells or headers in different columns).
Return ONLY a JSON object that conforms to the following schema keys. Use numbers for numeric fields and arrays for multi-year growth:
- purchase_price
- closing_costs
- acquisition_fee
- upfront_wc (or upfront_working_capital)
- construction_budget (or total_construction_budget)
- hold_period_months
- loan_amount
- loan_fee_pct
- loan_fee_amount
- interest_rate_index
- interest_rate_spread
- interest_rate
- io_period (Interest Only Period in months)
- amortization_years
- lp_equity
- gp_equity
- market_rent_growth (array [yr1, yr2, after])
- annual_expense_growth
- property_management_fee_pct
- initial_gpr
- initial_other_income
- initial_vacancy_pct
- bad_debt_pct
- initial_monthly_expenses
- exit_cap_rate
- sale_costs_pct
- preferred_return
- gp_promote_pct
- gp_equity_split

If you cannot find a value, put null. Standardize common label variations (e.g., "Total Const. Budget" -> "construction_budget").
CSV:
{csv_text[:4000]}
"""
    response = chat(
        messages=[{"role": "user", "content": prompt}],
        model=model_name,
        format=RealEstateSummary.model_json_schema()
    )

    # Ollama returns JSON in response.message.content per your example
    raw = response.message.content
    # Validate/parse into the pydantic model
    summary_model = RealEstateSummary.model_validate_json(raw)
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
            df[col] = pd.to_numeric(df[col], errors='ignore')
    return df

class RealEstateProForma:
    def __init__(self, summary_model: RealEstateSummary, mcf_path: str, waterfall_path: str):
        # store raw sources
        self.summary_model = summary_model
        self.mcf_df_raw = pd.read_csv(mcf_path, header=None)
        self.mcf_df_raw = clean_currency(self.mcf_df_raw)
        self.waterfall_df = pd.read_csv(waterfall_path, header=None)
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
        self.exit_cap_rate = float(self._get_model_value("exit_cap_rate", 0.06))
        self.sale_costs_pct = float(self._get_model_value("sale_costs_pct", 0.01))
        self.pref_return = float(self._get_model_value("preferred_return", 0.08))
        self.gp_promote_pct = float(self._get_model_value("gp_promote_pct", 0.0))
        self.gp_equity_split = float(self._get_model_value("gp_equity_split", 0.0))

    # --- keep the rest of your original methods, slightly adapted to reference these attributes ---
    def run_model(self):
        self._calculate_monthly_cash_flows()
        self._calculate_sale_and_exit_metrics()
        self._calculate_project_returns()
        self._build_output_tables()

    def _calculate_monthly_cash_flows(self):
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
            self.project_em = sum(cf for cf in levered_cfs if cf > 0) / abs(initial_investment) if initial_investment != 0 else 0.0
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

        # Save both tables — previously you stored a single DataFrame; now store separately
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


    def display_outputs(self):
        print("## Sources and Uses Summary")
        print(tabulate(self.sources_uses_df, headers='keys', tablefmt='pipe', showindex=False, numalign="right", stralign="left", floatfmt=",.0f"))
        print("\n" + "="*80 + "\n")

        print("## Sale Details and Exit Metrics")
        formatted_sale = self.sale_metrics_df.copy()
        formatted_sale.loc[formatted_sale['Metric']=='Exit Cap Rate','Value'] = formatted_sale.loc[formatted_sale['Metric']=='Exit Cap Rate','Value'] * 100
        formatted_sale['Value'] = formatted_sale.apply(lambda row: f"{row['Value']:.2f}%" if row['Metric']=='Exit Cap Rate' else f"${row['Value']:,.0f}", axis=1)
        print(tabulate(formatted_sale, headers='keys', tablefmt='pipe', showindex=False, numalign="right", stralign="left"))
        print("\n" + "="*80 + "\n")

        print("## Project-Level Return Metrics")
        formatted_returns = self.return_metrics_df.copy()
        formatted_returns['Value'] = formatted_returns['Value'].apply(lambda x: f"{x:.2%}" if 'IRR' in x or 'Return' in x else f"{x:.2f}x")
        print(tabulate(formatted_returns, headers='keys', tablefmt='pipe', showindex=False, numalign="right", stralign="left"))
        print("\n" + "="*80 + "\n")

        print("## Waterfall Breakdown")
        print(tabulate(self.waterfall_breakdown_df, headers='keys', tablefmt='pipe', showindex=False, numalign="right", stralign="left", floatfmt=",.0f"))
        print("\n" + "="*80 + "\n")

        print("## Monthly Cash Flows (First 12 Months)")
        print(tabulate(self.mcf_calculated_df.head(12), headers='keys', tablefmt='pipe', showindex=False, numalign="right", floatfmt=",.0f"))


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    SUMMARY_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Summary.csv'
    MCF_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Monthly CF.csv'
    WATERFALL_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Waterfall.csv'

    # 1) Use Ollama + pydantic schema to extract a normalized JSON summary
    print("Extracting normalized summary with ollama...")
    summary_model = extract_summary_with_ollama(SUMMARY_FILE, model_name="llama3.1")

    # 2) Build the pro forma from the validated model
    pro_forma = RealEstateProForma.from_summary_model(summary_model, MCF_FILE, WATERFALL_FILE)
    pro_forma.run_model()

    # 3) Display
    pro_forma.display_outputs()
