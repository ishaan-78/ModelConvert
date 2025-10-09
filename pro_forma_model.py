import pandas as pd
import numpy_financial as npf
from tabulate import tabulate

class RealEstateProForma:
    """
    A class to model a real estate acquisition pro forma based on provided CSV files.
    """
    def __init__(self, summary_path, mcf_path, waterfall_path):
        """
        Initializes the model by loading all necessary assumptions.
        """
        self.summary_df = pd.read_csv(summary_path, header=None)
        self.mcf_df_raw = pd.read_csv(mcf_path, header=None)
        self.waterfall_df = pd.read_csv(waterfall_path, header=None)
        
        # --- Store raw dataframes for display purposes ---
        self.sources_uses_df = None
        self.sale_metrics_df = None
        self.return_metrics_df = None
        self.waterfall_breakdown_df = None
        
        # --- Core Calculated DataFrame ---
        self.mcf_calculated_df = None

        self._load_assumptions()

    def _get_value(self, df, key, col=1):
        """Helper function to find a value in a key-value style DataFrame."""
        try:
            # Find the row index where the key is located in the first column
            row_index = df[df[0] == key].index[0]
            # Get the value from the specified column
            value = df.iloc[row_index, col]
            # Convert to numeric if possible, handling potential errors
            return pd.to_numeric(value, errors='coerce')
        except (IndexError, KeyError):
            # Return None or a default value if the key is not found
            return None

    def _load_assumptions(self):
        """Loads and parses all model inputs from the summary DataFrame."""
        # Acquisition & Property Details
        self.purchase_price = self._get_value(self.summary_df, 'Purchase Price')
        self.closing_costs = self._get_value(self.summary_df, 'Closing Costs')
        self.acquisition_fee = self._get_value(self.summary_df, 'Acquisition Fee')
        self.upfront_wc = self._get_value(self.summary_df, 'Up-Front Working Capital')
        self.construction_budget = self._get_value(self.summary_df, 'Total Construction Budget')
        self.hold_period_months = int(self._get_value(self.summary_df, 'Hold Period'))

        # Loan Information
        self.loan_amount = self._get_value(self.summary_df, 'Loan Amount')
        self.loan_fee_pct = self._get_value(self.summary_df, 'Loan Fee (as Percentage of Proceeds)')
        self.loan_fee_amount = self.loan_amount * self.loan_fee_pct
        self.interest_rate = self._get_value(self.summary_df, 'Interest Rate Index') + self._get_value(self.summary_df, 'Interest Rate Spread')
        self.io_period = int(self._get_value(self.summary_df, 'Interest Only Period'))
        self.amortization_years = int(self._get_value(self.summary_df, 'Amortization'))
        
        # Equity
        self.lp_equity = self._get_value(self.summary_df, 'LP Equity')
        self.gp_equity = self._get_value(self.summary_df, 'GP Equity')
        self.total_equity = self.lp_equity + self.gp_equity

        # Revenue & Expense Assumptions from Monthly CF Raw Data
        self.initial_gpr = self.mcf_df_raw.iloc[5, 3] # Gross Potential Rent at Month 1
        self.initial_other_income = self.mcf_df_raw.iloc[10, 3] + self.mcf_df_raw.iloc[11, 3] + self.mcf_df_raw.iloc[12, 3] + self.mcf_df_raw.iloc[13, 3] + self.mcf_df_raw.iloc[14, 3]
        self.initial_vacancy_pct = abs(self.mcf_df_raw.iloc[6, 3] / self.initial_gpr)
        self.bad_debt_pct = abs(self.mcf_df_raw.iloc[7, 3] / self.initial_gpr)
        
        # Using raw expense totals from month 1
        self.initial_monthly_expenses = abs(sum(self.mcf_df_raw.iloc[i, 3] for i in range(19, 29)))

        self.annual_rent_growth_yr1 = self._get_value(self.summary_df, 'Market Rent Growth', 1)
        self.annual_rent_growth_yr2 = self._get_value(self.summary_df, 'Market Rent Growth', 2)
        self.annual_rent_growth_after = self._get_value(self.summary_df, 'Market Rent Growth', 3)
        self.annual_expense_growth = self._get_value(self.summary_df, 'Annual Expense Growth')
        
        # Sale Assumptions
        self.exit_cap_rate = self._get_value(self.summary_df, 'Exit Cap Rate')
        self.sale_costs_pct = self._get_value(self.summary_df, 'Closing Costs (as % of Sale Price)')

        # Waterfall Assumptions
        self.pref_return = self._get_value(self.waterfall_df, 'Preferred Return')
        self.gp_promote_pct = self._get_value(self.waterfall_df, 'Hurdle #2', 2)
        self.gp_equity_split = self._get_value(self.waterfall_df, 'GP Equity', 1)

    def run_model(self):
        """Orchestrates the calculation of all financial components."""
        self._calculate_monthly_cash_flows()
        self._calculate_sale_and_exit_metrics()
        self._calculate_project_returns()
        self._build_output_tables()

    def _calculate_monthly_cash_flows(self):
        """Generates the monthly cash flow statement for the hold period."""
        # Prepare lists to hold monthly data
        months = list(range(1, self.hold_period_months + 1))
        gpr_list, vacancy_list, bad_debt_list, other_income_list, egr_list = [], [], [], [], []
        expenses_list, noi_list, interest_list, principal_list, cfa_ds_list = [], [], [], [], []
        
        loan_balance = self.loan_amount
        current_gpr = self.initial_gpr
        current_other_income = self.initial_other_income
        current_expenses = self.initial_monthly_expenses

        for m in months:
            year = (m - 1) // 12 + 1
            
            # --- Revenue ---
            if m > 1:
                if year == 1:
                    growth_rate = (1 + self.annual_rent_growth_yr1)**(1/12)
                elif year == 2:
                    growth_rate = (1 + self.annual_rent_growth_yr2)**(1/12)
                else:
                    growth_rate = (1 + self.annual_rent_growth_after)**(1/12)
                current_gpr *= growth_rate
                current_other_income *= growth_rate

            gpr_list.append(current_gpr)
            other_income_list.append(current_other_income)
            
            # Simplified vacancy for demonstration
            vacancy = current_gpr * self.initial_vacancy_pct
            bad_debt = current_gpr * self.bad_debt_pct
            vacancy_list.append(-vacancy)
            bad_debt_list.append(-bad_debt)
            
            egr = current_gpr - vacancy - bad_debt + current_other_income
            egr_list.append(egr)

            # --- Expenses ---
            if m > 1 and (m - 1) % 12 == 0:
                current_expenses *= (1 + self.annual_expense_growth)
            
            # Management fee as a % of EGR
            mgmt_fee_pct = self._get_value(self.summary_df, 'Property Management Fee %')
            mgmt_fee = egr * mgmt_fee_pct / 12
            total_expenses = (current_expenses - (self.initial_monthly_expenses * (mgmt_fee_pct/12))) + mgmt_fee
            expenses_list.append(-total_expenses)
            
            noi = egr - total_expenses
            noi_list.append(noi)

            # --- Debt Service ---
            interest = 0
            principal = 0
            if m <= self.io_period:
                interest = loan_balance * self.interest_rate / 12
            else:
                pmt = -npf.pmt(self.interest_rate / 12, self.amortization_years * 12, loan_balance)
                interest = npf.ipmt(self.interest_rate / 12, m - self.io_period, self.amortization_years * 12, loan_balance)
                principal = npf.ppmt(self.interest_rate / 12, m - self.io_period, self.amortization_years * 12, loan_balance)
                loan_balance += principal # principal is negative
            
            interest_list.append(interest)
            principal_list.append(principal)
            
            cfa_ds_list.append(noi + interest + principal)
            
        self.ending_loan_balance = loan_balance

        # Create DataFrame
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
        """Calculates sale proceeds and final exit numbers."""
        t12_noi_exit = self.mcf_calculated_df['Net Operating Income'].iloc[-12:].sum()
        self.sale_price = t12_noi_exit / self.exit_cap_rate
        self.costs_of_sale = self.sale_price * self.sale_costs_pct
        self.net_sale_proceeds = self.sale_price - self.costs_of_sale
        self.net_proceeds_to_equity = self.net_sale_proceeds - self.ending_loan_balance
        
    def _calculate_project_returns(self):
        """Calculates project-level IRR, EM, and other return metrics."""
        # Levered Cash Flow
        initial_investment = -self.total_equity
        interim_cfs = self.mcf_calculated_df['Cash Flow After Debt Service'].tolist()
        levered_cfs = [initial_investment] + interim_cfs
        levered_cfs[-1] += self.net_proceeds_to_equity
        
        self.project_irr = npf.irr(levered_cfs) * 12
        self.project_em = sum(cf for cf in levered_cfs if cf > 0) / abs(initial_investment)
        
        # Average Cash-on-Cash
        avg_annual_cfads = self.mcf_calculated_df['Cash Flow After Debt Service'].sum() / (self.hold_period_months / 12)
        self.avg_coc = avg_annual_cfads / self.total_equity

    def _build_output_tables(self):
        """Constructs the final DataFrames for display."""
        # --- Sources and Uses ---
        total_uses = self.purchase_price + self.acquisition_fee + self.closing_costs + self.upfront_wc + self.loan_fee_amount + self.construction_budget
        self.sources_uses_df = pd.DataFrame({
            'Sources': ['Acquisition Loan', 'LP Equity', 'GP Equity', 'Total Sources'],
            'Amount': [self.loan_amount, self.lp_equity, self.gp_equity, self.loan_amount + self.total_equity],
            'Uses': ['Purchase Price', 'Acquisition Fee', 'Closing Costs', 'Up-Front Working Capital', 'Loan Fees', 'Construction Budget', 'Total Uses'],
            'Amount_Uses': [self.purchase_price, self.acquisition_fee, self.closing_costs, self.upfront_wc, self.loan_fee_amount, self.construction_budget, total_uses]
        })

        # --- Sale Details ---
        self.sale_metrics_df = pd.DataFrame([
            {'Metric': 'Exit T-12 NOI', 'Value': self.mcf_calculated_df['Net Operating Income'].iloc[-12:].sum()},
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
        
        # --- Waterfall (Simplified for this example) ---
        # This is a basic waterfall. Real ones can be much more complex.
        lp_share = 1 - self.gp_equity_split
        
        # 1. Return of Capital
        lp_capital_returned = min(self.net_proceeds_to_equity, self.lp_equity)
        gp_capital_returned = min(max(0, self.net_proceeds_to_equity - lp_capital_returned), self.gp_equity)
        remaining_profit = self.net_proceeds_to_equity - lp_capital_returned - gp_capital_returned

        # 2. Preferred Return (simplified as a lump sum for this table)
        # A full model would accrue this monthly.
        lp_pref_due = self.lp_equity * ((1 + self.pref_return)**(self.hold_period_months/12) - 1)
        lp_pref_paid = min(remaining_profit, lp_pref_due)
        remaining_profit -= lp_pref_paid

        # 3. Promote Split
        promote_to_gp = remaining_profit * self.gp_promote_pct
        profit_to_lp = remaining_profit * (1- self.gp_promote_pct) * lp_share
        profit_to_gp_pro_rata = remaining_profit * (1 - self.gp_promote_pct) * self.gp_equity_split
        
        total_lp_dist = self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * lp_share + lp_capital_returned + lp_pref_paid + profit_to_lp
        total_gp_dist = self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * self.gp_equity_split + gp_capital_returned + promote_to_gp + profit_to_gp_pro_rata

        self.waterfall_breakdown_df = pd.DataFrame({
            'Distribution Tier': ['Interim Cash Flow', 'Return of Capital', 'Preferred Return', 'Promote Split', 'Total Distributions'],
            'To LP': [self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * lp_share, lp_capital_returned, lp_pref_paid, profit_to_lp, total_lp_dist],
            'To GP': [self.mcf_calculated_df['Cash Flow After Debt Service'].sum() * self.gp_equity_split, gp_capital_returned, 0, promote_to_gp + profit_to_gp_pro_rata, total_gp_dist]
        })

    def display_outputs(self):
        """Prints all the generated tables in a clean format."""
        
        print("## Sources and Uses Summary")
        print(tabulate(self.sources_uses_df, headers='keys', tablefmt='pipe', showindex=False, numalign="right", stralign="left", floatfmt=",.0f"))
        print("\n" + "="*80 + "\n")

        print("## Sale Details and Exit Metrics")
        self.sale_metrics_df.loc[self.sale_metrics_df['Metric'] == 'Exit Cap Rate', 'Value'] *= 100
        formatted_sale = self.sale_metrics_df.copy()
        formatted_sale['Value'] = formatted_sale.apply(
            lambda row: f"{row['Value']:.2f}%" if row['Metric'] == 'Exit Cap Rate' else f"${row['Value']:,.0f}",
            axis=1
        )
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

if __name__ == '__main__':
    # --- Define file paths ---
    # Make sure these CSV files are in the same directory as your script,
    # or provide the full path to them.
    SUMMARY_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Summary.csv'
    MCF_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Monthly CF.csv'
    WATERFALL_FILE = 'The+Landing+at+Avila+Acquisition+Model.xlsx - Waterfall.csv'
    
    # --- Run the model ---
    pro_forma = RealEstateProForma(
        summary_path=SUMMARY_FILE,
        mcf_path=MCF_FILE,
        waterfall_path=WATERFALL_FILE
    )
    pro_forma.run_model()
    
    # --- Display the results ---
    pro_forma.display_outputs()