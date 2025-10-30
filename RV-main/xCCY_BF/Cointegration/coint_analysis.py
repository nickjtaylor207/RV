import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings





class CointegrationAnalyzer:

    def __init__(self, excel_path):
        self.excel_path = excel_path
        self.sheets = {}
        self.load_data()


    def load_data(self):
        """Load all sheets from Excel file"""
        xl_file = pd.ExcelFile(self.excel_path)
        print(f"Loading data from: {self.excel_path}")
        print(f"Available sheets: {xl_file.sheet_names}\n")
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(xl_file, sheet_name=sheet_name, index_col=0)
            self.sheets[sheet_name] = df
            print(f"  {sheet_name}: {len(df)} pairs")

    def get_summary_statistics(self):
        summary_data = []
        for tenor, df in self.sheets.items():
            if tenor == 'Summary':
                continue
            summary = {
                'Tenor': tenor,
                'Total_Pairs': len(df),
                'Avg_Score': df['cointegration_score'].mean() if 'cointegration_score' in df.columns else np.nan,
                'Median_Score': df['cointegration_score'].median() if 'cointegration_score' in df.columns else np.nan,
                'Excellent_Count': (df['cointegration_strength'] == 'Excellent').sum() if 'cointegration_strength' in df.columns else 0,
                'Strong_Count': (df['cointegration_strength'].isin(['Very Strong', 'Strong'])).sum() if 'cointegration_strength' in df.columns else 0,
                'Avg_EG_pvalue': df['eg_eg_p_value'].mean() if 'eg_eg_p_value' in df.columns else np.nan,
                'Avg_ADF_pvalue': df['adf_p_value'].mean() if 'adf_p_value' in df.columns else np.nan,
                'Avg_Halflife': df['halflife_halflife'].median() if 'halflife_halflife' in df.columns else np.nan,
                'OOS_Success_Rate': (df['oos_oos_stationary'] == True).mean() * 100 if 'oos_oos_stationary' in df.columns else np.nan}
            summary_data.append(summary)
        summary_df = pd.DataFrame(summary_data)
        return summary_df.round(3)


path = r'C:\Users\Nick Taylor\Desktop\RV_xCCY_Project\xCCY_BF\Data\BF_Cointegration_All_5Y_3Oct.xlsx'

analyzer = CointegrationAnalyzer(path)

print(analyzer.get_summary_statistics())













