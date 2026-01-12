import sys
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import pickle
import json
import re  # NEW: For variable name sanitization
from datetime import datetime
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# --- SCIENTIFIC ENGINE (Integrated) ---
try:
    from semopy import Model, calc_stats
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    SEM_AVAILABLE = True
except ImportError:
    SEM_AVAILABLE = False

# NEW: For reliability analysis
try:
    import pingouin as pg
    PINGOUIN_AVAILABLE = True
except:
    PINGOUIN_AVAILABLE = False

# NEW: Helper functions for reliability analysis (FIX #2)
def sanitize_var_name(name):
    """Convert variable names to valid Python identifiers for semopy (FIX #4)"""
    name = str(name).strip()
    # Replace special chars with underscore
    name = re.sub(r'[^\w]', '_', name)
    # Remove leading digits
    name = re.sub(r'^(\d+)', r'var\1', name)
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    name = name.strip('_')
    return name

def calculate_cronbach_alpha(data):
    """Calculate Cronbach's Alpha for reliability (FIX #2)"""
    if PINGOUIN_AVAILABLE:
        try:
            return pg.cronbach_alpha(data=data)
        except:
            pass
    # Manual calculation fallback
    n_items = data.shape[1]
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    return (alpha, None)

def calculate_composite_reliability(loadings):
    """Calculate Composite Reliability (CR) from factor loadings (FIX #2)"""
    sum_loadings = loadings.sum()
    sum_error_var = (1 - loadings ** 2).sum()
    cr = (sum_loadings ** 2) / ((sum_loadings ** 2) + sum_error_var)
    return cr

def calculate_ave(loadings):
    """Calculate Average Variance Extracted (AVE) (FIX #2)"""
    squared_loadings = loadings ** 2
    sum_squared = squared_loadings.sum()
    sum_error = (1 - squared_loadings).sum()
    ave = sum_squared / (sum_squared + sum_error)
    return ave

def map_method_names(method_name, method_type='extraction'):
    """Map display names back to library parameter names"""
    extraction_map = {
        "Principal Axis Factoring": "principal",
        "Minimum Residual (MinRes)": "minres",
        "Maximum Likelihood": "ml"
    }
    rotation_map = {
        "Varimax (Orthogonal)": "varimax",
        "Promax (Oblique)": "promax",
        "Oblimin (Oblique)": "oblimin",
        "None": "none"
    }
    if method_type == 'extraction':
        return extraction_map.get(method_name, method_name.lower().split()[0] if ' ' in method_name else method_name.lower())
    else:
        return rotation_map.get(method_name, method_name.lower().split()[0] if ' ' in method_name else method_name.lower())

# --- MODERN ACADEMIC THEME ---
STYLESHEET = """
QMainWindow { background-color: #f0f2f5; }
QSplitter { background-color: #d1d4d7; }
QSplitter::handle { background-color: #b0b3b7; width: 4px; height: 4px; }
QSplitter::handle:hover { background-color: #3498db; }
QSplitter::handle:pressed { background-color: #2980b9; }
QFrame#Sidebar { background-color: #2c3e50; border-right: 1px solid #1a252f; }
QLabel#SidebarTitle { color: #ecf0f1; font-size: 18px; font-weight: bold; padding: 20px 10px; }
QPushButton#NavBtn { 
    background-color: transparent; color: #bdc3c7; text-align: left; 
    padding: 15px 20px; border: none; font-size: 13px; font-weight: 500;
}
QPushButton#NavBtn:hover { background-color: #34495e; color: white; }
QPushButton#NavBtn:checked { background-color: #3498db; color: white; font-weight: bold; }
QWidget#ConfigPanel { background-color: white; border-right: 1px solid #d1d4d7; }
QTextEdit#ReportArea { background-color: white; border: none; font-family: 'Consolas', 'Courier New'; font-size: 11px; line-height: 1.6; padding: 15px; }
QPushButton#ActionBtn { background-color: #3498db; color: white; padding: 12px; border-radius: 5px; font-weight: bold; font-size: 14px; }
QPushButton#ActionBtn:hover { background-color: #2980b9; }
QTableWidget { background-color: white; gridline-color: #d1d4d7; selection-background-color: #3498db; }
QTableWidget::item { padding: 5px; }
QHeaderView::section { background-color: #34495e; color: white; padding: 8px; font-weight: bold; border: none; }
"""


def safe_float(value, default=0.0):
    """Safely convert a value to float, handling '-' and other invalid strings"""
    try:
        if value == '-' or value == '' or value is None:
            return default
        return float(str(value).replace('-', str(default)))
    except (ValueError, TypeError, AttributeError):
        return default


class VariableSelector(QDialog):
    """Professional dialog for manual variable selection."""
    def __init__(self, items, title="Select Variables", parent=None, multi=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(450, 550)
        layout = QVBoxLayout(self)
        
        # Info label
        info = QLabel("Select variables for analysis (check boxes to include):")
        info.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(info)
        
        self.search = QLineEdit()
        self.search.setPlaceholderText("🔍 Search variables...")
        self.search.textChanged.connect(self.filter_items)
        layout.addWidget(self.search)
        
        # Select/Deselect all buttons
        btn_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.clicked.connect(self.select_all)
        self.deselect_all_btn = QPushButton("Deselect All")
        self.deselect_all_btn.clicked.connect(self.deselect_all)
        btn_layout.addWidget(self.select_all_btn)
        btn_layout.addWidget(self.deselect_all_btn)
        layout.addLayout(btn_layout)
        
        self.list = QListWidget()
        if multi:
            self.list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        
        # DEFAULT: All items UNCHECKED
        for item in items:
            l_item = QListWidgetItem(item)
            l_item.setFlags(l_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            l_item.setCheckState(Qt.CheckState.Unchecked)
            self.list.addItem(l_item)
        layout.addWidget(self.list)
        
        # Count label
        self.count_label = QLabel("0 variables selected")
        self.count_label.setStyleSheet("color: #3498db; font-weight: bold;")
        layout.addWidget(self.count_label)
        self.list.itemChanged.connect(self.update_count)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
        
        self.all_items = items
        self.update_count()

    def select_all(self):
        for i in range(self.list.count()):
            if not self.list.item(i).isHidden():
                self.list.item(i).setCheckState(Qt.CheckState.Checked)
    
    def deselect_all(self):
        for i in range(self.list.count()):
            if not self.list.item(i).isHidden():
                self.list.item(i).setCheckState(Qt.CheckState.Unchecked)
    
    def update_count(self):
        count = sum(1 for i in range(self.list.count()) 
                   if self.list.item(i).checkState() == Qt.CheckState.Checked)
        self.count_label.setText(f"{count} variable{'s' if count != 1 else ''} selected")

    def filter_items(self, text):
        for i in range(self.list.count()):
            item = self.list.item(i)
            item.setHidden(text.lower() not in item.text().lower())

    def get_selected(self):
        return [self.list.item(i).text() for i in range(self.list.count()) 
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

class DataEditor(QDialog):
    """In-app data editor."""
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Editor")
        self.setMinimumSize(900, 600)
        self.data = data.copy()
        
        layout = QVBoxLayout(self)
        
        # Toolbar
        toolbar = QHBoxLayout()
        self.add_col_btn = QPushButton("Add Column")
        self.add_col_btn.clicked.connect(self.add_column)
        self.del_col_btn = QPushButton("Delete Column")
        self.del_col_btn.clicked.connect(self.delete_column)
        self.add_row_btn = QPushButton("Add Row")
        self.add_row_btn.clicked.connect(self.add_row)
        self.del_row_btn = QPushButton("Delete Row")
        self.del_row_btn.clicked.connect(self.delete_row)
        
        toolbar.addWidget(self.add_col_btn)
        toolbar.addWidget(self.del_col_btn)
        toolbar.addWidget(self.add_row_btn)
        toolbar.addWidget(self.del_row_btn)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Table
        self.table = QTableWidget()
        self.load_data()
        layout.addWidget(self.table)
        
        # Buttons
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.save_changes)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
    
    def load_data(self):
        self.table.setRowCount(len(self.data))
        self.table.setColumnCount(len(self.data.columns))
        self.table.setHorizontalHeaderLabels(self.data.columns.tolist())
        
        for i in range(len(self.data)):
            for j, col in enumerate(self.data.columns):
                item = QTableWidgetItem(str(self.data.iloc[i, j]))
                self.table.setItem(i, j, item)
    
    def add_column(self):
        name, ok = QInputDialog.getText(self, "Add Column", "Column name:")
        if ok and name:
            self.data[name] = np.nan
            self.load_data()
    
    def delete_column(self):
        col = self.table.currentColumn()
        if col >= 0:
            col_name = self.data.columns[col]
            self.data = self.data.drop(columns=[col_name])
            self.load_data()
    
    def add_row(self):
        self.data.loc[len(self.data)] = [np.nan] * len(self.data.columns)
        self.load_data()
    
    def delete_row(self):
        row = self.table.currentRow()
        if row >= 0:
            self.data = self.data.drop(self.data.index[row]).reset_index(drop=True)
            self.load_data()
    
    def save_changes(self):
        for i in range(self.table.rowCount()):
            for j in range(self.table.columnCount()):
                item = self.table.item(i, j)
                if item:
                    try:
                        val = float(item.text())
                    except:
                        val = item.text()
                    self.data.iloc[i, j] = val
        self.accept()
    
    def get_data(self):
        return self.data

class AnalysisThread(QThread):
    finished = pyqtSignal(dict)
    
    def __init__(self, data, mode, settings):
        super().__init__()
        self.original_data = data.copy()
        self.mode = mode
        self.settings = settings
    
    def run(self):
        try:
            out = {'mode': self.mode}
            
            if self.mode == 'REGRESSION':
                dv = self.settings['dv']
                ivs = self.settings['ivs']
                reg_type = self.settings.get('type', 'standard')
                
                # Prepare base data
                cols = [dv] + ivs
                
                # Add moderator variables if moderation
                if reg_type == 'moderation':
                    moderator = self.settings['moderator']
                    cols.append(moderator)
                
                df = self.original_data[cols].dropna()
                
                if df.empty:
                    raise ValueError("No valid data after removing missing values.")
                
                out['n'] = len(df)
                out['dv'] = dv
                out['ivs'] = ivs
                out['reg_type'] = reg_type
                
                if reg_type == 'standard':
                    # Standard regression
                    X = df[ivs]
                    y = df[dv]
                    
                    X_sm = sm.add_constant(X)
                    model = sm.OLS(y, X_sm).fit()

                    # Calculate standardized coefficients
                    X_std = (X - X.mean()) / X.std()
                    y_std = (y - y.mean()) / y.std()
                    X_std_sm = sm.add_constant(X_std)
                    model_std = sm.OLS(y_std, X_std_sm).fit()

                    out['r_squared'] = model.rsquared
                    out['adj_r_squared'] = model.rsquared_adj
                    out['f_stat'] = model.fvalue
                    out['f_pvalue'] = model.f_pvalue
                    out['params'] = model.params
                    out['standardized_params'] = model_std.params
                    out['pvalues'] = model.pvalues
                    out['std_err'] = model.bse
                    out['conf_int'] = model.conf_int()
                    out['tvalues'] = model.tvalues
                    
                    residuals = model.resid
                    fitted = model.fittedvalues
                    
                elif reg_type == 'hierarchical':
                    # Hierarchical regression
                    blocks = self.settings['blocks']
                    y = df[dv]
                    
                    models = []
                    out['blocks'] = []
                    
                    cumulative_vars = []
                    for block_idx, block_vars in enumerate(blocks):
                        cumulative_vars.extend(block_vars)
                        X = df[cumulative_vars]
                        X_sm = sm.add_constant(X)
                        model = sm.OLS(y, X_sm).fit()
                        models.append(model)
                        
                        # Standardized
                        X_std = (X - X.mean()) / X.std()
                        X_std_sm = sm.add_constant(X_std)
                        model_std = sm.OLS(y, X_std_sm).fit()
                        
                        block_info = {
                            'block': block_idx + 1,
                            'variables': cumulative_vars.copy(),
                            'new_variables': block_vars,
                            'r_squared': model.rsquared,
                            'adj_r_squared': model.rsquared_adj,
                            'f_stat': model.fvalue,
                            'f_pvalue': model.f_pvalue,
                            'params': model.params,
                            'standardized_params': model_std.params,
                            'pvalues': model.pvalues,
                            'std_err': model.bse,
                            'tvalues': model.tvalues
                        }
                        
                        # Calculate R-squared change
                        if block_idx > 0:
                            r2_change = model.rsquared - models[block_idx - 1].rsquared
                            # F-test for R-squared change
                            df1 = len(block_vars) - len(blocks[block_idx - 1])
                            df2 = len(df) - len(block_vars) - 1
                            f_change = (r2_change / df1) / ((1 - model.rsquared) / df2)
                            p_change = 1 - stats.f.cdf(f_change, df1, df2)
                            
                            block_info['r2_change'] = r2_change
                            block_info['f_change'] = f_change
                            block_info['p_change'] = p_change
                        
                        out['blocks'].append(block_info)
                    
                    # Use final model for diagnostics
                    model = models[-1]
                    residuals = model.resid
                    fitted = model.fittedvalues
                    X = df[blocks[-1]]
                    
                elif reg_type == 'moderation':
                    # Moderation analysis
                    moderator = self.settings['moderator']
                    focal_predictor = ivs[0]  # First IV is the focal predictor
                    
                    # Center variables
                    df_centered = df.copy()
                    df_centered[focal_predictor] = df[focal_predictor] - df[focal_predictor].mean()
                    df_centered[moderator] = df[moderator] - df[moderator].mean()
                    
                    # Create interaction term
                    interaction_name = f"{focal_predictor} × {moderator}"
                    df_centered[interaction_name] = df_centered[focal_predictor] * df_centered[moderator]
                    
                    # Model without interaction
                    X1 = df_centered[[focal_predictor, moderator]]
                    X1_sm = sm.add_constant(X1)
                    model1 = sm.OLS(df[dv], X1_sm).fit()
                    
                    # Model with interaction
                    X2 = df_centered[[focal_predictor, moderator, interaction_name]]
                    X2_sm = sm.add_constant(X2)
                    model2 = sm.OLS(df[dv], X2_sm).fit()
                    
                    out['moderator'] = moderator
                    out['focal_predictor'] = focal_predictor
                    out['interaction_term'] = interaction_name
                    
                    # Model 1 (without interaction)
                    out['model1'] = {
                        'r_squared': model1.rsquared,
                        'adj_r_squared': model1.rsquared_adj,
                        'params': model1.params,
                        'pvalues': model1.pvalues,
                        'std_err': model1.bse,
                        'tvalues': model1.tvalues
                    }
                    
                    # Model 2 (with interaction)
                    out['model2'] = {
                        'r_squared': model2.rsquared,
                        'adj_r_squared': model2.rsquared_adj,
                        'params': model2.params,
                        'pvalues': model2.pvalues,
                        'std_err': model2.bse,
                        'tvalues': model2.tvalues,
                        'conf_int': model2.conf_int()
                    }
                    
                    # R-squared change
                    r2_change = model2.rsquared - model1.rsquared
                    df1 = 1
                    df2 = len(df) - 4  # constant + 2 main effects + interaction
                    f_change = (r2_change / df1) / ((1 - model2.rsquared) / df2)
                    p_change = 1 - stats.f.cdf(f_change, df1, df2)
                    
                    out['r2_change'] = r2_change
                    out['f_change'] = f_change
                    out['p_change'] = p_change
                    
                    model = model2
                    residuals = model2.resid
                    fitted = model2.fittedvalues
                    X = X2
                
                # Common assumption tests
                X_sm = sm.add_constant(X)
                
                # Normality
                if len(residuals) <= 5000:
                    shapiro_stat, shapiro_p = stats.shapiro(residuals)
                    out['normality_test'] = 'Shapiro-Wilk'
                    out['normality_stat'] = shapiro_stat
                    out['normality_p'] = shapiro_p
                else:
                    ks_stat, ks_p = stats.kstest(residuals, 'norm')
                    out['normality_test'] = 'Kolmogorov-Smirnov'
                    out['normality_stat'] = ks_stat
                    out['normality_p'] = ks_p
                
                # Homoscedasticity
                bp_test = het_breuschpagan(residuals, X_sm)
                out['bp_stat'] = bp_test[0]
                out['bp_p'] = bp_test[1]
                
                # VIF
                if X.shape[1] > 1:
                    vif_data = pd.DataFrame()
                    vif_data['Variable'] = X.columns
                    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
                    out['vif'] = vif_data
                else:
                    out['vif'] = None
                
                # Durbin-Watson
                dw_stat = durbin_watson(residuals)
                out['dw_stat'] = dw_stat
                
                out['residuals'] = residuals.values
                out['fitted'] = fitted.values
                
            elif self.mode == 'DESCRIPTIVES':
                df = self.original_data.select_dtypes(include=[np.number])
                if df.empty:
                    raise ValueError("No numeric variables found.")
                
                out['n'] = len(df)
                out['variables'] = df.columns.tolist()
                
                # Calculate statistics for each variable
                stats_dict = {}
                for col in df.columns:
                    col_data = df[col].dropna()
                    stats_dict[col] = {
                        'N': len(col_data),
                        'Missing': df[col].isnull().sum(),
                        'Mean': col_data.mean(),
                        'SD': col_data.std(),
                        'Min': col_data.min(),
                        'Q1': col_data.quantile(0.25),
                        'Median': col_data.median(),
                        'Q3': col_data.quantile(0.75),
                        'Max': col_data.max(),
                        'Skewness': col_data.skew(),
                        'Kurtosis': col_data.kurtosis()
                    }
                
                out['stats'] = stats_dict
                
            elif self.mode == 'EFA':
                df = self.original_data.select_dtypes(include=[np.number]).dropna()
                
                if df.empty:
                    raise ValueError("No valid numeric data after removing missing values.")
                
                df = df.loc[:, df.std() > 0]
                
                if len(df.columns) < 3:
                    raise ValueError("EFA requires at least 3 variables with non-zero variance.")
                
                n_factors = self.settings.get('n_factors', 0)
                # Auto-determine if 0
                if n_factors == 0:
                    fa_temp = FactorAnalyzer(rotation=None)
                    fa_temp.fit(df)
                    ev_temp, _ = fa_temp.get_eigenvalues()
                    n_factors = sum(ev_temp > 1.0)
                    n_factors = max(1, n_factors)
                
                if n_factors > len(df.columns):
                    n_factors = len(df.columns)
                
                fa = FactorAnalyzer(n_factors=n_factors, 
                                    method=self.settings['extr'], 
                                    rotation=self.settings['rot'])
                fa.fit(df)
                ev, _ = fa.get_eigenvalues()
                var_stats = fa.get_factor_variance()
                kmo_all, kmo_model = calculate_kmo(df)
                chi, p = calculate_bartlett_sphericity(df)
                
                out['n'] = len(df)
                out['n_factors'] = n_factors
                out['variable_names'] = df.columns.tolist()  # Store full names
                out['loadings'] = pd.DataFrame(fa.loadings_, 
                                              index=df.columns,
                                              columns=[f'Factor {i+1}' for i in range(n_factors)])
                out['eigenvalues'] = ev
                out['original_df'] = df
                out['communalities'] = pd.Series(fa.get_communalities(), index=df.columns)
                out['pct_var'] = var_stats[1] * 100
                out['cum_var'] = var_stats[2] * 100
                out['actual_factors'] = len(var_stats[1])
                out['kmo'] = kmo_model
                out['bartlett_p'] = p
                out['extraction'] = self.settings['extr']
                out['rotation'] = self.settings['rot']
                
                # NEW: Store original data for reliability calculation (FIX #2)
                out['original_df'] = self.original_data[df.columns].dropna()
                

            elif self.mode == 'CORRELATION':
                vars_selected = self.settings['variables']
                df = self.original_data[vars_selected].dropna()
                method = self.settings['method'].lower()

                if df.empty:
                    raise ValueError("No valid data after removing missing values.")

                if len(vars_selected) < 2:
                    raise ValueError("Correlation analysis requires at least 2 variables.")

                # Calculate correlation matrix
                corr_matrix = df.corr(method=method)

                # Calculate p-values
                n = len(df)
                p_matrix = pd.DataFrame(np.zeros((len(vars_selected), len(vars_selected))),
                                       columns=vars_selected, index=vars_selected)

                for i, var1 in enumerate(vars_selected):
                    for j, var2 in enumerate(vars_selected):
                        if i != j:
                            if method == 'pearson':
                                _, p = stats.pearsonr(df[var1], df[var2])
                            elif method == 'spearman':
                                _, p = stats.spearmanr(df[var1], df[var2])
                            else:  # kendall
                                _, p = stats.kendalltau(df[var1], df[var2])
                            p_matrix.iloc[i, j] = p
                        else:
                            p_matrix.iloc[i, j] = 0.0

                # Calculate descriptive statistics
                desc_stats = {}
                for var in vars_selected:
                    desc_stats[var] = {
                        'mean': df[var].mean(),
                        'sd': df[var].std(),
                        'min': df[var].min(),
                        'max': df[var].max()
                    }

                out['corr_matrix'] = corr_matrix
                out['p_matrix'] = p_matrix
                out['n'] = n
                out['method'] = method
                out['variables'] = vars_selected
                out['descriptives'] = desc_stats

            elif self.mode == 'TTEST':
                test_type = self.settings['test_type']

                if test_type == 'Independent Samples':
                    dv = self.settings['dv']
                    group_var = self.settings['group']
                    df = self.original_data[[dv, group_var]].dropna()

                    if df.empty:
                        raise ValueError('No valid data after removing missing values.')

                    groups = df[group_var].unique()
                    if len(groups) != 2:
                        raise ValueError(f'Independent t-test requires exactly 2 groups, found {len(groups)}')

                    group1 = df[df[group_var] == groups[0]][dv]
                    group2 = df[df[group_var] == groups[1]][dv]

                    levene_stat, levene_p = stats.levene(group1, group2)
                    t_stat, p_val = stats.ttest_ind(group1, group2)
                    pooled_std = np.sqrt(((len(group1)-1)*group1.std()**2 + (len(group2)-1)*group2.std()**2) / (len(group1)+len(group2)-2))
                    cohens_d = (group1.mean() - group2.mean()) / pooled_std

                    out['test_type'] = 'Independent Samples'
                    out['t_statistic'] = t_stat
                    out['p_value'] = p_val
                    out['df'] = len(group1) + len(group2) - 2
                    out['group1_name'] = str(groups[0])
                    out['group2_name'] = str(groups[1])
                    out['group1_mean'] = group1.mean()
                    out['group2_mean'] = group2.mean()
                    out['group1_sd'] = group1.std()
                    out['group2_sd'] = group2.std()
                    out['group1_n'] = len(group1)
                    out['group2_n'] = len(group2)
                    out['levene_stat'] = levene_stat
                    out['levene_p'] = levene_p
                    out['cohens_d'] = cohens_d
                    out['dv'] = dv
                    out['group_var'] = group_var

                elif test_type == 'Paired Samples':
                    var1 = self.settings['var1']
                    var2 = self.settings['var2']
                    df = self.original_data[[var1, var2]].dropna()

                    if df.empty:
                        raise ValueError('No valid data after removing missing values.')

                    t_stat, p_val = stats.ttest_rel(df[var1], df[var2])
                    diff = df[var1] - df[var2]
                    cohens_d = diff.mean() / diff.std()

                    out['test_type'] = 'Paired Samples'
                    out['t_statistic'] = t_stat
                    out['p_value'] = p_val
                    out['df'] = len(df) - 1
                    out['var1'] = var1
                    out['var2'] = var2
                    out['var1_mean'] = df[var1].mean()
                    out['var2_mean'] = df[var2].mean()
                    out['var1_sd'] = df[var1].std()
                    out['var2_sd'] = df[var2].std()
                    out['mean_diff'] = diff.mean()
                    out['sd_diff'] = diff.std()
                    out['n'] = len(df)
                    out['cohens_d'] = cohens_d

                elif test_type == 'One Sample':
                    var = self.settings['var']
                    test_value = self.settings['testvalue']
                    df = self.original_data[[var]].dropna()

                    if df.empty:
                        raise ValueError('No valid data after removing missing values.')

                    t_stat, p_val = stats.ttest_1samp(df[var], test_value)
                    cohens_d = (df[var].mean() - test_value) / df[var].std()

                    out['test_type'] = 'One Sample'
                    out['t_statistic'] = t_stat
                    out['p_value'] = p_val
                    out['df'] = len(df) - 1
                    out['var'] = var
                    out['testvalue'] = test_value
                    out['sample_mean'] = df[var].mean()
                    out['sample_sd'] = df[var].std()
                    out['n'] = len(df)
                    out['cohens_d'] = cohens_d

            elif self.mode == 'CFA' and SEM_AVAILABLE:
                df = self.original_data.select_dtypes(include=[np.number]).dropna()
                
                if df.empty:
                    raise ValueError("No valid numeric data after removing missing values.")
                
                # FIX #4: Sanitize variable names for semopy
                original_cols = df.columns.tolist()
                sanitized_cols = [sanitize_var_name(col) for col in original_cols]
                col_mapping = dict(zip(original_cols, sanitized_cols))
                reverse_mapping = dict(zip(sanitized_cols, original_cols))
                
                # Rename dataframe columns
                df_sanitized = df.rename(columns=col_mapping)
                
                # Sanitize syntax
                syntax_sanitized = self.settings['syntax']
                for orig, san in col_mapping.items():
                    # Replace variable names in syntax (word boundaries)
                    syntax_sanitized = re.sub(r'\b' + re.escape(orig) + r'\b', san, syntax_sanitized)
                
                # Run CFA with sanitized names
                mod = Model(syntax_sanitized)
                n_vars = len(df_sanitized.columns)
                if n_vars > 50:
                    try:
                        mod.fit(df_sanitized, solver="SLSQP", obj="MLW", solver_options={"maxiter": 5000, "ftol": 1e-6})
                    except TypeError:
                        # scipy 1.14+ compatibility
                        mod.fit(df_sanitized, solver="SLSQP", obj="MLW")
                else:
                    mod.fit(df_sanitized)
                # Extract factor loadings
                params_df = mod.inspect(std_est=True)
                loadings = params_df[params_df["op"] == "~"]  # Factor loadings
                covariances = params_df[params_df["op"] == "~~"]  # Variances/covariances
                
                fit_stats = calc_stats(mod).to_dict(orient='records')[0]
                
                out['n'] = len(df)
                out['stats'] = fit_stats
                out['loadings'] = loadings
                out['covariances'] = covariances
                out['reverse_mapping'] = reverse_mapping
                out['syntax'] = self.settings['syntax']
                out['col_mapping'] = col_mapping
                
            self.finished.emit(out)
            
        except Exception as e:
            import traceback
            self.finished.emit({'error': str(e), 'traceback': traceback.format_exc()})

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Psychometric Studio Pro 2026")
        self.setGeometry(100, 100, 1600, 900)
        self.setStyleSheet(STYLESHEET)
        self.data = None
        self.current_report = ""
        self.current_results = {}
        self.analysis_tabs = {}
        
        # Main splitter with SMOOTH RESIZING
        self.main_split = QSplitter(Qt.Orientation.Horizontal)
        self.main_split.setHandleWidth(4)
        self.main_split.setOpaqueResize(True)
        self.main_split.setChildrenCollapsible(False)
        self.setCentralWidget(self.main_split)
        
        # 1. Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setMaximumWidth(300)
        sl = QVBoxLayout(self.sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        title = QLabel("PSYCHOMETRIC STUDIO")
        title.setObjectName("SidebarTitle")
        sl.addWidget(title)
        
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_btns = {}
        nav_items = [
            ("DATA IMPORT", 0),
            ("DESCRIPTIVES", 1),
            ("CORRELATION", 2),
            ("T-TEST", 3),
            ("REGRESSION", 4),
            ("EXPLORATORY (EFA)", 5),
            ("CONFIRMATORY (CFA)", 6),
            ("SEM PATH", 7)
        ]
        
        for name, idx in nav_items:
            btn = QPushButton(name)
            btn.setObjectName("NavBtn")
            btn.setCheckable(True)
            self.nav_btns[name] = btn
            sl.addWidget(btn)
            self.nav_group.addButton(btn, idx)
        
        self.nav_btns["DATA IMPORT"].setChecked(True)
        sl.addStretch()
        self.main_split.addWidget(self.sidebar)

        # 2. Config Panel
        self.config_panel = QStackedWidget()
        self.config_panel.setObjectName("ConfigPanel")
        self.config_panel.setMinimumWidth(300)
        self._init_config_pages()
        self.main_split.addWidget(self.config_panel)

        # 3. Output Pane
        self.output_tabs = QTabWidget()
        self.output_tabs.setTabsClosable(True)
        self.output_tabs.tabCloseRequested.connect(self.close_analysis_tab)
        self.report_area = QTextEdit()
        self.report_area.setObjectName("ReportArea")
        self.report_area.setReadOnly(True)
        self.output_tabs.addTab(self.report_area, "📊 Results")
        

        
        self.main_split.addWidget(self.output_tabs)
        
        # Set initial sizes and stretch factors
        self.main_split.setSizes([220, 350, 1030])
        self.main_split.setStretchFactor(0, 0)
        self.main_split.setStretchFactor(1, 1)
        self.main_split.setStretchFactor(2, 3)
        
        self._connect_nav()
        self._create_menubar()

    def _create_menubar(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        import_action = QAction("Import Data...", self)
        import_action.setShortcut("Ctrl+O")
        import_action.triggered.connect(self.import_data)
        file_menu.addAction(import_action)
        
        save_project = QAction("Save Project...", self)
        save_project.setShortcut("Ctrl+S")
        save_project.triggered.connect(self.save_project)
        file_menu.addAction(save_project)
        
        load_project = QAction("Load Project...", self)
        load_project.setShortcut("Ctrl+L")
        load_project.triggered.connect(self.load_project)
        file_menu.addAction(load_project)
        
        file_menu.addSeparator()
        
        # NEW: Export Results (FIX #3)
        export_results = QAction("Export Results...", self)
        export_results.triggered.connect(self.export_results)
        file_menu.addAction(export_results)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        data_menu = menubar.addMenu("Data")
        
        edit_data = QAction("Edit Data...", self)
        edit_data.triggered.connect(self.edit_data)
        data_menu.addAction(edit_data)
        
        view_data = QAction("View Data Summary", self)
        view_data.triggered.connect(self.view_data_summary)
        data_menu.addAction(view_data)

    def _init_config_pages(self):
        # 0. Data Import
        p1 = QWidget()
        l1 = QVBoxLayout(p1)
        l1.addWidget(QLabel("<h2>Data Source</h2>"))
        self.load_btn = QPushButton("📁 Import Excel (.xlsx) or CSV")
        self.load_btn.setObjectName("ActionBtn")
        self.load_btn.clicked.connect(self.import_data)
        l1.addWidget(self.load_btn)
        self.data_label = QLabel("No active dataset loaded.")
        self.data_label.setWordWrap(True)
        l1.addWidget(self.data_label)
        l1.addStretch()
        self.config_panel.addWidget(p1)

        # 1. Descriptives
        p_desc = QWidget()
        l_desc = QVBoxLayout(p_desc)
        l_desc.addWidget(QLabel("<h2>Descriptive Statistics</h2>"))
        l_desc.addWidget(QLabel("Generates comprehensive descriptive statistics for all numeric variables."))
        self.run_desc = QPushButton("📊 COMPUTE DESCRIPTIVE STATISTICS")
        self.run_desc.setObjectName("ActionBtn")
        self.run_desc.clicked.connect(lambda: self.launch_analysis('DESCRIPTIVES'))
        l_desc.addWidget(self.run_desc)
        l_desc.addStretch()
        self.config_panel.addWidget(p_desc)

        # 2. Correlation Analysis
        p_corr = QWidget()
        l_corr = QVBoxLayout(p_corr)
        l_corr.addWidget(QLabel("<h2>Correlation Analysis</h2>"))
        l_corr.addWidget(QLabel("Compute correlation matrix with significance tests."))

        l_corr.addWidget(QLabel("<b>Correlation Method:</b>"))
        self.corr_method = QComboBox()
        self.corr_method.addItems(["Pearson", "Spearman", "Kendall"])
        l_corr.addWidget(self.corr_method)

        self.corr_select_vars_btn = QPushButton("📋 Select Variables")
        self.corr_select_vars_btn.clicked.connect(self.select_correlation_variables)
        l_corr.addWidget(self.corr_select_vars_btn)

        self.corr_vars_label = QLabel("No variables selected")
        self.corr_vars_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
        l_corr.addWidget(self.corr_vars_label)

        self.corr_selected_vars = []

        self.run_corr = QPushButton("📊 COMPUTE CORRELATIONS")
        self.run_corr.setObjectName("ActionBtn")
        self.run_corr.clicked.connect(lambda: self.launch_analysis('CORRELATION'))
        l_corr.addWidget(self.run_corr)
        l_corr.addStretch()
        self.config_panel.addWidget(p_corr)

        # 3. T-Test Analysis
        p_ttest = QWidget()
        l_ttest = QVBoxLayout(p_ttest)
        l_ttest.addWidget(QLabel("<h2>T-Test Analysis</h2>"))
        l_ttest.addWidget(QLabel("Compare means between groups or paired observations."))

        l_ttest.addWidget(QLabel("<b>Test Type:</b>"))
        self.ttest_type = QComboBox()
        self.ttest_type.addItems(["Independent Samples", "Paired Samples", "One Sample"])
        self.ttest_type.currentIndexChanged.connect(self.on_ttest_type_changed)
        l_ttest.addWidget(self.ttest_type)

        self.ttest_controls = QWidget()
        self.ttest_controls_layout = QVBoxLayout(self.ttest_controls)
        self.ttest_controls_layout.setContentsMargins(0, 0, 0, 0)
        l_ttest.addWidget(self.ttest_controls)

        self.run_ttest = QPushButton("📊 RUN T-TEST")
        self.run_ttest.setObjectName("ActionBtn")
        self.run_ttest.clicked.connect(lambda: self.launch_analysis('TTEST'))
        l_ttest.addWidget(self.run_ttest)
        l_ttest.addStretch()
        self.config_panel.addWidget(p_ttest)

        self.on_ttest_type_changed(0)

        # 4. Regression (ENHANCED)
        p_reg = QWidget()
        l_reg = QVBoxLayout(p_reg)
        l_reg.addWidget(QLabel("<h2>Regression Analysis</h2>"))
        
        # Regression type selector
        l_reg.addWidget(QLabel("<b>Regression Type:</b>"))
        self.reg_type = QComboBox()
        self.reg_type.addItems(["Standard Regression", "Hierarchical Regression", "Moderation Analysis"])
        self.reg_type.currentIndexChanged.connect(self.on_regression_type_changed)
        l_reg.addWidget(self.reg_type)
        
        # Standard regression controls
        self.standard_widget = QWidget()
        std_layout = QVBoxLayout(self.standard_widget)
        std_layout.setContentsMargins(0, 0, 0, 0)
        std_layout.addWidget(QLabel("<b>Dependent Variable:</b>"))
        self.reg_dv_btn = QPushButton("Select DV →")
        self.reg_dv_btn.clicked.connect(self.select_dv)
        std_layout.addWidget(self.reg_dv_btn)
        std_layout.addWidget(QLabel("<b>Independent Variables:</b>"))
        self.reg_iv_btn = QPushButton("Select IVs →")
        self.reg_iv_btn.clicked.connect(self.select_ivs)
        std_layout.addWidget(self.reg_iv_btn)
        # Dummy variable creation button
        self.dummy_var_btn = QPushButton("🔢 Create Dummy Variables")
        self.dummy_var_btn.clicked.connect(self.create_dummy_variables)
        self.dummy_var_btn.setStyleSheet('background-color: #95a5a6; color: white; padding: 8px; font-size: 12px;')
        std_layout.addWidget(self.dummy_var_btn)
        l_reg.addWidget(self.standard_widget)
        
        # Hierarchical regression controls
        self.hierarchical_widget = QWidget()
        hier_layout = QVBoxLayout(self.hierarchical_widget)
        hier_layout.setContentsMargins(0, 0, 0, 0)
        hier_layout.addWidget(QLabel("<b>Dependent Variable:</b>"))
        self.hier_dv_btn = QPushButton("Select DV →")
        self.hier_dv_btn.clicked.connect(self.select_hier_dv)
        hier_layout.addWidget(self.hier_dv_btn)
        hier_layout.addWidget(QLabel("<b>Configure Blocks:</b>"))
        self.hier_blocks_btn = QPushButton("Define Blocks →")
        self.hier_blocks_btn.clicked.connect(self.configure_hierarchical_blocks)
        hier_layout.addWidget(self.hier_blocks_btn)
        self.hierarchical_widget.hide()
        l_reg.addWidget(self.hierarchical_widget)
        
        # Moderation controls
        self.moderation_widget = QWidget()
        mod_layout = QVBoxLayout(self.moderation_widget)
        mod_layout.setContentsMargins(0, 0, 0, 0)
        mod_layout.addWidget(QLabel("<b>Dependent Variable:</b>"))
        self.mod_dv_btn = QPushButton("Select DV →")
        self.mod_dv_btn.clicked.connect(self.select_mod_dv)
        mod_layout.addWidget(self.mod_dv_btn)
        mod_layout.addWidget(QLabel("<b>Focal Predictor:</b>"))
        self.mod_predictor_btn = QPushButton("Select Predictor →")
        self.mod_predictor_btn.clicked.connect(self.select_mod_predictor)
        mod_layout.addWidget(self.mod_predictor_btn)
        mod_layout.addWidget(QLabel("<b>Moderator Variable:</b>"))
        self.mod_moderator_btn = QPushButton("Select Moderator →")
        self.mod_moderator_btn.clicked.connect(self.select_moderator)
        mod_layout.addWidget(self.mod_moderator_btn)
        self.moderation_widget.hide()
        l_reg.addWidget(self.moderation_widget)
        
        self.run_regression = QPushButton("▶️ RUN REGRESSION ANALYSIS")
        self.run_regression.setObjectName("ActionBtn")
        self.run_regression.clicked.connect(lambda: self.launch_analysis('REGRESSION'))
        l_reg.addWidget(self.run_regression)
        l_reg.addStretch()
        self.config_panel.addWidget(p_reg)

        # 3. EFA
        p2 = QWidget()
        l2 = QVBoxLayout(p2)
        l2.addWidget(QLabel("<h2>Exploratory Factor Analysis</h2>"))
        
        l2.addWidget(QLabel("Number of Factors:"))
        self.efa_n_factors = QSpinBox()
        self.efa_n_factors.setMinimum(0)
        self.efa_n_factors.setMaximum(20)
        self.efa_n_factors.setValue(0)
        l2.addWidget(self.efa_n_factors)
        
        l2.addWidget(QLabel("Extraction Method:"))
        self.efa_extr = QComboBox()
        self.efa_extr.addItems(["Principal Axis Factoring", "Minimum Residual (MinRes)", "Maximum Likelihood"])
        l2.addWidget(self.efa_extr)
        
        l2.addWidget(QLabel("Rotation Method:"))
        self.efa_rot = QComboBox()
        self.efa_rot.addItems(["Varimax (Orthogonal)", "Promax (Oblique)", "Oblimin (Oblique)", "None"])
        l2.addWidget(self.efa_rot)
        
        self.run_efa = QPushButton("▶️ RUN EXPLORATORY ANALYSIS")
        self.run_efa.setObjectName("ActionBtn")
        self.run_efa.clicked.connect(lambda: self.launch_analysis('EFA'))
        l2.addWidget(self.run_efa)
        l2.addStretch()
        self.config_panel.addWidget(p2)

        # 4. CFA (ENHANCED WITH HELP)
        p3 = QWidget()
        l3 = QVBoxLayout(p3)
        l3.addWidget(QLabel("<h2>Confirmatory Factor Analysis</h2>"))
        
        # Add syntax help
        help_text = QLabel(
            "<b>📖 Syntax Guide:</b><br>"
            "<b>Basic syntax:</b><br>"
            "FactorName =~ item1 + item2 + item3<br><br>"
            "<b>Example (2-factor model):</b><br>"
            "Anxiety =~ anx1 + anx2 + anx3<br>"
            "Depression =~ dep1 + dep2 + dep3<br><br>"
            "<b>Correlated factors:</b><br>"
            "Anxiety ~~ Depression<br><br>"
            "<b>Fixed loading:</b><br>"
            "Factor =~ 1*item1 + item2<br><br>"
            "<i>Note: Variable names must match your data columns exactly.</i>"
        )
        help_text.setWordWrap(True)
        help_text.setStyleSheet("background-color: #e8f4f8; padding: 10px; border-radius: 5px; font-size: 11px;")
        l3.addWidget(help_text)
        
        self.cfa_syntax = QTextEdit()
        self.cfa_syntax.setPlaceholderText("Enter model syntax here...\n\nExample:\nFactor1 =~ item1 + item2 + item3\nFactor2 =~ item4 + item5 + item6")
        self.cfa_syntax.setMinimumHeight(150)
        l3.addWidget(self.cfa_syntax)
        
        self.run_cfa = QPushButton("▶️ RUN CONFIRMATORY ANALYSIS")
        self.run_cfa.setObjectName("ActionBtn")
        self.run_cfa.clicked.connect(lambda: self.launch_analysis('CFA'))
        l3.addWidget(self.run_cfa)
        l3.addStretch()
        self.config_panel.addWidget(p3)

        # 5. SEM
        p4 = QWidget()
        l4 = QVBoxLayout(p4)
        l4.addWidget(QLabel("<h2>SEM Path Analysis</h2>"))
        l4.addWidget(QLabel("🔧 Coming soon: Mediation and path models"))
        l4.addStretch()
        self.config_panel.addWidget(p4)

    def on_regression_type_changed(self, index):
        self.standard_widget.setVisible(index == 0)
        self.hierarchical_widget.setVisible(index == 1)
        self.moderation_widget.setVisible(index == 2)

    def configure_hierarchical_blocks(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            QMessageBox.warning(self, "No Numeric Variables", "No numeric variables found.")
            return
        
        # Simple dialog for block configuration
        dialog = QDialog(self)
        dialog.setWindowTitle("Configure Hierarchical Blocks")
        dialog.setMinimumSize(500, 400)
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel("<b>Define variable blocks (enter step-by-step):</b>"))
        layout.addWidget(QLabel("Block 1 contains all variables from Block 1.\nBlock 2 contains all from Block 1 + new variables, etc."))
        
        # Block widgets
        self.block_widgets = []
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for i in range(3):  # Allow up to 3 blocks
            block_frame = QGroupBox(f"Block {i+1}")
            block_layout = QVBoxLayout()
            btn = QPushButton(f"Select variables for Block {i+1}")
            block_layout.addWidget(btn)
            label = QLabel("No variables selected")
            label.setWordWrap(True)
            block_layout.addWidget(label)
            block_frame.setLayout(block_layout)
            scroll_layout.addWidget(block_frame)
            
            btn.clicked.connect(lambda checked, idx=i, lbl=label: self.select_block_vars(idx, lbl))
            self.block_widgets.append({'button': btn, 'label': label, 'vars': []})
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)
        
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Store blocks
            self.hierarchical_blocks = [w['vars'] for w in self.block_widgets if w['vars']]
            if self.hierarchical_blocks:
                self.hier_blocks_btn.setText(f"✓ {len(self.hierarchical_blocks)} blocks defined")
                self.hier_blocks_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def select_block_vars(self, block_idx, label):
        if self.data is None:
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        sel = VariableSelector(numeric_cols, title=f"Select Variables for Block {block_idx+1}", parent=self)
        if sel.exec() == QDialog.DialogCode.Accepted:
            vars = sel.get_selected()
            if vars:
                self.block_widgets[block_idx]['vars'] = vars
                label.setText(f"Selected: {', '.join(vars[:3])}{'...' if len(vars) > 3 else ''}")

    def select_dv(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            QMessageBox.warning(self, "No Numeric Variables", "No numeric variables found for regression.")
            return
        
        sel = VariableSelector(numeric_cols, 
                               title="Select Dependent Variable", 
                               parent=self, 
                               multi=False)
        if sel.exec() == QDialog.DialogCode.Accepted:
            vars = sel.get_selected()
            if vars:
                self.selected_dv = vars[0]
                self.reg_dv_btn.setText(f"✓ DV: {self.selected_dv}")
                self.reg_dv_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def select_ivs(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            QMessageBox.warning(self, "No Numeric Variables", "No numeric variables found for regression.")
            return
        
        sel = VariableSelector(numeric_cols, 
                               title="Select Independent Variables", 
                               parent=self)
        if sel.exec() == QDialog.DialogCode.Accepted:
            vars = sel.get_selected()
            if vars:
                self.selected_ivs = vars
                display_text = ', '.join(vars[:2])
                if len(vars) > 2:
                    display_text += f" (+{len(vars)-2} more)"
                self.reg_iv_btn.setText(f"✓ IVs: {display_text}")
                self.reg_iv_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def select_hier_dv(self):
        self.select_dv()
        if hasattr(self, 'selected_dv'):
            self.hier_dv_btn.setText(f"✓ DV: {self.selected_dv}")
            self.hier_dv_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def select_mod_dv(self):
        self.select_dv()
        if hasattr(self, 'selected_dv'):
            self.mod_dv_btn.setText(f"✓ DV: {self.selected_dv}")
            self.mod_dv_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def select_mod_predictor(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        sel = VariableSelector(numeric_cols, title="Select Focal Predictor", parent=self, multi=False)
        if sel.exec() == QDialog.DialogCode.Accepted:
            vars = sel.get_selected()
            if vars:
                self.mod_predictor = vars[0]
                self.mod_predictor_btn.setText(f"✓ Predictor: {self.mod_predictor}")
                self.mod_predictor_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def select_moderator(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        sel = VariableSelector(numeric_cols, title="Select Moderator Variable", parent=self, multi=False)
        if sel.exec() == QDialog.DialogCode.Accepted:
            vars = sel.get_selected()
            if vars:
                self.mod_moderator = vars[0]
                self.mod_moderator_btn.setText(f"✓ Moderator: {self.mod_moderator}")
                self.mod_moderator_btn.setStyleSheet("background-color: #27ae60; color: white; padding: 10px; font-weight: bold;")

    def create_dummy_variables(self):
        """Create dummy variables from categorical columns."""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return

        # Get categorical columns (non-numeric or numeric with few unique values)
        categorical_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                categorical_cols.append(col)
            elif self.data[col].nunique() <= 10:  # Numeric but few categories
                categorical_cols.append(col)

        if not categorical_cols:
            QMessageBox.information(self, "No Categorical Variables",
                                   "No categorical variables found in the dataset.")
            return

        # Let user select which variable to dummy code
        dlg = VariableSelector(categorical_cols, "Select Variable to Dummy Code", self, multi=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected_var = dlg.get_selected()
            if not selected_var:
                return
            
            var = selected_var[0]
            
            # Get unique categories (exclude NaN)
            categories = self.data[var].unique()
            categories = [c for c in categories if pd.notna(c)]
            
            if len(categories) < 2:
                QMessageBox.warning(self, "Insufficient Categories",
                                   "Variable must have at least 2 categories.")
                return
            
            # Ask user to select reference category
            ref_cat, ok = QInputDialog.getItem(
                self, "Reference Category",
                f"Select reference category for '{var}':\n"
                "(This category will be coded as 0 in all dummy variables)",
                [str(c) for c in categories], 0, False)
            
            if not ok:
                return
            
            # Create dummy variables (n-1 dummies, excluding reference)
            created_dummies = []
            for cat in categories:
                if str(cat) != ref_cat:
                    dummy_name = f"{var}_{cat}"
                    self.data[dummy_name] = (self.data[var] == cat).astype(int)
                    created_dummies.append(dummy_name)
            
            # Show success message
            num_dummies = len(created_dummies)
            dummy_list = "\n".join([f"  • {name}" for name in created_dummies])
            QMessageBox.information(
                self, "✓ Dummy Variables Created",
                f"Created {num_dummies} dummy variable(s) for '{var}':\n\n"
                f"{dummy_list}\n\n"
                f"Reference category: {ref_cat} (coded as 0)\n\n"
                f"These new variables can now be used as independent variables in regression.")
            
            # Update data label
            self.data_label.setText(
                f"✓ Dataset: {len(self.data.columns)} variables, N={len(self.data)} cases")

    def import_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Data", "", 
                                             "Data Files (*.csv *.xlsx *.xls)")
        if not path:
            return
        try:
            if path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(path)
            else:
                self.data = pd.read_csv(path)
            
            self.data_label.setText(f"✅ Dataset Loaded\n\n📊 {len(self.data.columns)} variables\n👥 N = {len(self.data)} cases")
            self.data_label.setStyleSheet("color: #27ae60; font-weight: bold; font-size: 13px;")
            QMessageBox.information(self, "Success", 
                                   f"Dataset loaded successfully!\n\nVariables: {len(self.data.columns)}\nCases: {len(self.data)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load file:\n{str(e)}")

    def edit_data(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        editor = DataEditor(self.data, self)
        if editor.exec() == QDialog.DialogCode.Accepted:
            self.data = editor.get_data()
            self.data_label.setText(f"✅ Dataset Loaded\n\n📊 {len(self.data.columns)} variables\n👥 N = {len(self.data)} cases")
            QMessageBox.information(self, "Success", "Data updated successfully!")

    def view_data_summary(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        summary = f"{'='*80}\n"
        summary += f"DATASET SUMMARY\n"
        summary += f"{'='*80}\n\n"
        summary += f"Total Cases: {len(self.data)}\n"
        summary += f"Total Variables: {len(self.data.columns)}\n\n"
        summary += f"Variables:\n{', '.join(self.data.columns.tolist())}\n\n"
        summary += f"{'='*80}\n"
        summary += f"DATA TYPES\n"
        summary += f"{'='*80}\n"
        summary += f"{self.data.dtypes.to_string()}\n\n"
        summary += f"{'='*80}\n"
        summary += f"MISSING VALUES\n"
        summary += f"{'='*80}\n"
        summary += f"{self.data.isnull().sum().to_string()}"
        
        self.report_area.setText(summary)
        # Tab set by get_or_create

    def save_project(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "No data to save.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", 
                                             "Project Files (*.psp)")
        if not path:
            return
        
        try:
            project = {
                'data': self.data,
                'timestamp': datetime.now().isoformat(),
                'report': self.current_report
            }
            with open(path, 'wb') as f:
                pickle.dump(project, f)
            QMessageBox.information(self, "Success", "Project saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not save project:\n{str(e)}")

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", 
                                             "Project Files (*.psp)")
        if not path:
            return
        
        try:
            with open(path, 'rb') as f:
                project = pickle.load(f)
            
            self.data = project['data']
            self.current_report = project.get('report', '')
            self.report_area.setText(self.current_report)
            self.data_label.setText(f"✅ Dataset Loaded\n\n📊 {len(self.data.columns)} variables\n👥 N = {len(self.data)} cases")
            QMessageBox.information(self, "Success", "Project loaded successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not load project:\n{str(e)}")

    # NEW: Export Results (FIX #3)
    def export_results(self):
        if not self.current_report:
            QMessageBox.warning(self, "No Results", "No results to export.")
            return
        
        path, filter_selected = QFileDialog.getSaveFileName(
            self, "Export Results", "", 
            "Excel Files (*.xlsx);;Text Files (*.txt);;CSV Files (*.csv)"
        )
        if not path:
            return
        
        try:
            if path.endswith('.xlsx'):
                # Create Excel file with multiple sheets
                with pd.ExcelWriter(path, engine='openpyxl') as writer:
                    # Write report as text in first sheet
                    report_lines = self.current_report.split('\n')
                    report_df = pd.DataFrame({'Report': report_lines})
                    report_df.to_excel(writer, sheet_name='Report', index=False)
                    
                    # If we have regression results, export coefficients
                    if self.current_results.get('mode') == 'REGRESSION':
                        res = self.current_results
                        if 'params' in res:
                            coef_df = pd.DataFrame({
                                'Parameter': res['params'].index,
                                'Coefficient': res['params'].values,
                                'Std.Error': res['std_err'].values,
                                'T-statistic': res['tvalues'].values,
                                'P-value': res['pvalues'].values
                            })
                            coef_df.to_excel(writer, sheet_name='Coefficients', index=False)
                    
                    # If we have EFA results, export loadings
                    elif self.current_results.get('mode') == 'EFA':
                        if 'loadings' in self.current_results:
                            self.current_results['loadings'].to_excel(writer, sheet_name='Factor_Loadings')
                    
                    # If we have data, add summary
                    if self.data is not None:
                        desc = self.data.describe().T
                        desc.to_excel(writer, sheet_name='Data_Summary')
                
                QMessageBox.information(self, "Success", "Results exported to Excel!")
                
            elif path.endswith('.csv'):
                # Export different content based on analysis type
                if self.current_results.get('mode') == 'EFA' and 'loadings' in self.current_results:
                    self.current_results['loadings'].to_csv(path)
                elif self.data is not None:
                    self.data.describe().T.to_csv(path)
                else:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(self.current_report)
                QMessageBox.information(self, "Success", "Results exported to CSV!")
            else:
                # Text file
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(self.current_report)
                QMessageBox.information(self, "Success", "Results exported successfully!")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not export:\n{str(e)}")

    def on_ttest_type_changed(self, index):
        """Update T-Test controls based on selected test type"""
        for i in reversed(range(self.ttest_controls_layout.count())):
            widget = self.ttest_controls_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        test_type = self.ttest_type.currentText()

        if test_type == "Independent Samples":
            self.ttest_controls_layout.addWidget(QLabel("<b>Dependent Variable:</b>"))
            self.ttest_dv_btn = QPushButton("Select DV")
            self.ttest_dv_btn.clicked.connect(self.select_ttest_dv)
            self.ttest_controls_layout.addWidget(self.ttest_dv_btn)
            self.ttest_dv_label = QLabel("No variable selected")
            self.ttest_dv_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            self.ttest_controls_layout.addWidget(self.ttest_dv_label)

            self.ttest_controls_layout.addWidget(QLabel("<b>Grouping Variable:</b>"))
            self.ttest_group_btn = QPushButton("Select Group Variable")
            self.ttest_group_btn.clicked.connect(self.select_ttest_group)
            self.ttest_controls_layout.addWidget(self.ttest_group_btn)
            self.ttest_group_label = QLabel("No variable selected")
            self.ttest_group_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            self.ttest_controls_layout.addWidget(self.ttest_group_label)

        elif test_type == "Paired Samples":
            self.ttest_controls_layout.addWidget(QLabel("<b>Variable 1 (Time 1):</b>"))
            self.ttest_var1_btn = QPushButton("Select Variable 1")
            self.ttest_var1_btn.clicked.connect(self.select_ttest_var1)
            self.ttest_controls_layout.addWidget(self.ttest_var1_btn)
            self.ttest_var1_label = QLabel("No variable selected")
            self.ttest_var1_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            self.ttest_controls_layout.addWidget(self.ttest_var1_label)

            self.ttest_controls_layout.addWidget(QLabel("<b>Variable 2 (Time 2):</b>"))
            self.ttest_var2_btn = QPushButton("Select Variable 2")
            self.ttest_var2_btn.clicked.connect(self.select_ttest_var2)
            self.ttest_controls_layout.addWidget(self.ttest_var2_btn)
            self.ttest_var2_label = QLabel("No variable selected")
            self.ttest_var2_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            self.ttest_controls_layout.addWidget(self.ttest_var2_label)

        else:
            self.ttest_controls_layout.addWidget(QLabel("<b>Test Variable:</b>"))
            self.ttest_test_var_btn = QPushButton("Select Variable")
            self.ttest_test_var_btn.clicked.connect(self.select_ttest_test_var)
            self.ttest_controls_layout.addWidget(self.ttest_test_var_btn)
            self.ttest_test_var_label = QLabel("No variable selected")
            self.ttest_test_var_label.setStyleSheet("color: #7f8c8d; font-style: italic;")
            self.ttest_controls_layout.addWidget(self.ttest_test_var_label)

            self.ttest_controls_layout.addWidget(QLabel("<b>Test Value (μ₀):</b>"))
            self.ttest_test_value = QLineEdit("0")
            self.ttest_test_value.setPlaceholderText("Enter test value")
            self.ttest_controls_layout.addWidget(self.ttest_test_value)

    def select_ttest_dv(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()
        dlg = VariableSelector(numeric_vars, "Select Dependent Variable", self, multi=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ttest_dv = dlg.get_selected()[0]
            self.ttest_dv_label.setText(f"DV: {self.ttest_dv}")
            self.ttest_dv_label.setStyleSheet("color: #27ae60; font-weight: bold;")

    def select_ttest_group(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        all_vars = self.data.columns.tolist()
        dlg = VariableSelector(all_vars, "Select Grouping Variable", self, multi=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ttest_group = dlg.get_selected()[0]
            self.ttest_group_label.setText(f"Group: {self.ttest_group}")
            self.ttest_group_label.setStyleSheet("color: #27ae60; font-weight: bold;")

    def select_ttest_var1(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()
        dlg = VariableSelector(numeric_vars, "Select Variable 1 (Time 1)", self, multi=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ttest_var1 = dlg.get_selected()[0]
            self.ttest_var1_label.setText(f"Var 1: {self.ttest_var1}")
            self.ttest_var1_label.setStyleSheet("color: #27ae60; font-weight: bold;")

    def select_ttest_var2(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()
        dlg = VariableSelector(numeric_vars, "Select Variable 2 (Time 2)", self, multi=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ttest_var2 = dlg.get_selected()[0]
            self.ttest_var2_label.setText(f"Var 2: {self.ttest_var2}")
            self.ttest_var2_label.setStyleSheet("color: #27ae60; font-weight: bold;")

    def select_ttest_test_var(self):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        numeric_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()
        dlg = VariableSelector(numeric_vars, "Select Test Variable", self, multi=False)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.ttest_test_var = dlg.get_selected()[0]
            self.ttest_test_var_label.setText(f"Variable: {self.ttest_test_var}")
            self.ttest_test_var_label.setStyleSheet("color: #27ae60; font-weight: bold;")

    def select_correlation_variables(self):
        """Select variables for correlation analysis"""
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return

        numeric_vars = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_vars) < 2:
            QMessageBox.warning(self, "Insufficient Variables", 
                              "Correlation analysis requires at least 2 numeric variables.")
            return

        dlg = VariableSelector(numeric_vars, "Select Variables for Correlation", self, multi=True)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.corr_selected_vars = dlg.get_selected()
            if len(self.corr_selected_vars) < 2:
                QMessageBox.warning(self, "Selection Error", 
                                  "Please select at least 2 variables for correlation analysis.")
                self.corr_selected_vars = []
                self.corr_vars_label.setText("No variables selected")
            else:
                self.corr_vars_label.setText(f"{len(self.corr_selected_vars)} variables selected: " + 
                                            ", ".join(self.corr_selected_vars[:3]) + 
                                            ("..." if len(self.corr_selected_vars) > 3 else ""))
                self.corr_vars_label.setStyleSheet("color: #27ae60; font-weight: bold;")

    def launch_analysis(self, mode):
        if self.data is None:
            QMessageBox.warning(self, "No Data", "Please import data first.")
            return
        
        settings = {}
        data_to_use = self.data.copy()
        
        if mode == 'REGRESSION':
            reg_type_idx = self.reg_type.currentIndex()
            
            if reg_type_idx == 0:  # Standard
                if not hasattr(self, 'selected_dv') or not hasattr(self, 'selected_ivs'):
                    QMessageBox.warning(self, "Variables Not Selected", 
                                       "Please select dependent and independent variables.")
                    return
                settings['type'] = 'standard'
                settings['dv'] = self.selected_dv
                settings['ivs'] = self.selected_ivs
                
            elif reg_type_idx == 1:  # Hierarchical
                if not hasattr(self, 'selected_dv') or not hasattr(self, 'hierarchical_blocks'):
                    QMessageBox.warning(self, "Variables Not Selected", 
                                       "Please select DV and configure blocks.")
                    return
                settings['type'] = 'hierarchical'
                settings['dv'] = self.selected_dv
                settings['blocks'] = self.hierarchical_blocks
                # Flatten for column selection
                settings['ivs'] = list(set([var for block in self.hierarchical_blocks for var in block]))
                
            elif reg_type_idx == 2:  # Moderation
                if not hasattr(self, 'selected_dv') or not hasattr(self, 'mod_predictor') or not hasattr(self, 'mod_moderator'):
                    QMessageBox.warning(self, "Variables Not Selected", 
                                       "Please select DV, predictor, and moderator.")
                    return
                settings['type'] = 'moderation'
                settings['dv'] = self.selected_dv
                settings['ivs'] = [self.mod_predictor]
                settings['moderator'] = self.mod_moderator
            
        elif mode == 'DESCRIPTIVES':
            sel = VariableSelector(self.data.select_dtypes(include=[np.number]).columns.tolist(), 
                                  title="Select Variables for Descriptive Statistics",
                                  parent=self)
            if sel.exec() == QDialog.DialogCode.Accepted:
                vars = sel.get_selected()
                if not vars:
                    QMessageBox.warning(self, "No Variables", "Please select at least one variable.")
                    return
                data_to_use = self.data[vars].copy()
            else:
                return
            
        elif mode == 'EFA':
            sel = VariableSelector(self.data.select_dtypes(include=[np.number]).columns.tolist(),
                                  title="Select Variables for EFA",
                                  parent=self)
            if sel.exec() == QDialog.DialogCode.Accepted:
                vars = sel.get_selected()
                if len(vars) < 3:
                    QMessageBox.warning(self, "Insufficient Variables", 
                                      "EFA requires at least 3 variables.")
                    return
                data_to_use = self.data[vars].copy()
            else:
                return
            
            settings['extr'] = map_method_names(self.efa_extr.currentText(), 'extraction')
            settings['rot'] = map_method_names(self.efa_rot.currentText(), 'rotation')
            settings['n_factors'] = self.efa_n_factors.value()
            
        elif mode == 'CFA':
            if not SEM_AVAILABLE:
                QMessageBox.critical(self, "Missing Package", 
                                   "semopy package is required for CFA. Please install it.")
                return
            settings['syntax'] = self.cfa_syntax.toPlainText()
            if not settings['syntax']:
                QMessageBox.warning(self, "No Syntax", "Please enter CFA model syntax.")
                return
        

        elif mode == 'CORRELATION':
            if not hasattr(self, 'corr_selected_vars') or not self.corr_selected_vars:
                QMessageBox.warning(self, "No Variables Selected",
                                   "Please select variables for correlation analysis.")
                return
            settings['variables'] = self.corr_selected_vars
            settings['method'] = self.corr_method.currentText()

        elif mode == 'TTEST':
            test_type = self.ttest_type.currentText()
            if test_type == 'Independent Samples':
                if not hasattr(self, 'ttest_dv') or not hasattr(self, 'ttest_group'):
                    QMessageBox.warning(self, "Variables Not Selected",
                                       "Please select dependent and grouping variables.")
                    return
                settings['test_type'] = 'Independent Samples'
                settings['dv'] = self.ttest_dv
                settings['group'] = self.ttest_group
            elif test_type == 'Paired Samples':
                if not hasattr(self, 'ttest_var1') or not hasattr(self, 'ttest_var2'):
                    QMessageBox.warning(self, "Variables Not Selected",
                                       "Please select both variables.")
                    return
                settings['test_type'] = 'Paired Samples'
                settings['var1'] = self.ttest_var1
                settings['var2'] = self.ttest_var2
            else:  # One Sample
                if not hasattr(self, 'ttesttestvar'):
                    QMessageBox.warning(self, "Variable Not Selected",
                           "Please select test variable.")
                    return
                try:
                    test_value = float(self.ttest_testvalue.text())
                except ValueError:
                    QMessageBox.warning(self, "Invalid Value",
                           "Please enter a valid numeric test value.")
                    return
                settings['test_type'] = 'One Sample'
                settings['var'] = self.ttesttestvar
                settings['testvalue'] = test_value


        # Show progress dialog
        progress = QProgressDialog("Running analysis...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        self.th = AnalysisThread(data_to_use, mode, settings)
        self.th.finished.connect(lambda res: self.display_report(res, progress))
        self.th.start()


    def get_or_create_analysis_tab(self, mode, title):
        if mode not in self.analysis_tabs:
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.setContentsMargins(0, 0, 0, 0)
            splitter = QSplitter(Qt.Orientation.Vertical)
            report_area = QTextEdit()
            report_area.setObjectName("ReportArea")
            report_area.setReadOnly(True)
            splitter.addWidget(report_area)
            plot_widget = QWidget()
            plot_layout = QVBoxLayout(plot_widget)
            splitter.addWidget(plot_widget)
            splitter.setSizes([700, 300])
            tab_layout.addWidget(splitter)
            self.analysis_tabs[mode] = {'widget': tab_widget, 'report': report_area, 'plot_widget': plot_widget, 'plot_layout': plot_layout}
            self.output_tabs.addTab(tab_widget, title)
        else:
            tab_idx = self.output_tabs.indexOf(self.analysis_tabs[mode]['widget'])
            self.output_tabs.setTabText(tab_idx, title)
        tab_idx = self.output_tabs.indexOf(self.analysis_tabs[mode]['widget'])
        self.output_tabs.setCurrentIndex(tab_idx)
        return self.analysis_tabs[mode]

    def close_analysis_tab(self, index):
        widget_at_index = self.output_tabs.widget(index)
        mode_to_remove = None
        for mode, tab_data in self.analysis_tabs.items():
            if tab_data['widget'] == widget_at_index:
                mode_to_remove = mode
                break
        if mode_to_remove:
            del self.analysis_tabs[mode_to_remove]
            if mode_to_remove in self.current_results:
                del self.current_results[mode_to_remove]
        self.output_tabs.removeTab(index)

    def clear_all_results(self):
        reply = QMessageBox.question(self, 'Clear All Results', 'Are you sure?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            while self.output_tabs.count() > 0:
                self.output_tabs.removeTab(0)
            self.analysis_tabs.clear()
            self.current_results.clear()

    def display_report(self, res, progress=None):
        if progress:
            progress.close()
        
        if 'error' in res:
            error_msg = f"Analysis Error:\n\n{res['error']}"
            if 'traceback' in res:
                error_msg += f"\n\nTechnical Details:\n{res['traceback']}"
            QMessageBox.critical(self, "Analysis Error", error_msg)
            return
        
        mode = res['mode']
        if mode == 'DESCRIPTIVES':
            title = f"📊 Descriptives (N={res['n']})"
        elif mode == 'CORRELATION':
            title = f"📊 Correlations ({len(res['variables'])} vars)"
        elif mode == 'TTEST':
            title = f"📊 T-Test: {res['test_type']}"
        elif mode == 'REGRESSION':
            title = f"📈 Regression: {res['dv']}"
        elif mode == 'EFA':
            title = f"🔍 EFA ({res['n_factors']} factors)"
        elif mode == 'CFA':
            title = f"✓ CFA (N={res['n']})"
        elif mode == 'CORRELATION':
            title = f"📊 Correlations ({len(res['variables'])} vars)"
        elif mode == 'TTEST':
            title = f"📊 T-Test: {res['test_type']}"
        else:
            title = f"📊 {mode}"
        tab_data = self.get_or_create_analysis_tab(mode, title)
        report_area = tab_data['report']
        plot_layout = tab_data['plot_layout']
        self.current_results[mode] = res

        
        rep = f"{'='*90}\n"
        rep += f"PSYCHOMETRIC STUDIO PRO 2026 - ANALYSIS REPORT\n"
        rep += f"{'='*90}\n"
        rep += f"Analysis Type: {res['mode']}\n"
        rep += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        rep += f"Sample Size (N): {res.get('n', '?')} (Listwise deletion applied)\n"
        rep += f"{'='*90}\n\n"
        
        if res['mode'] == 'DESCRIPTIVES':
            rep += "DESCRIPTIVE STATISTICS\n"
            rep += f"{'='*90}\n\n"

            stats_data = res['stats']

            var_num = 1
            for var, var_stats in stats_data.items():
                rep += f"[{var_num}] Variable: {var}\n"
                rep += f"    N = {var_stats['N']:>6}  |  Missing = {var_stats['Missing']:>4}  |  Mean = {var_stats['Mean']:>8.3f}  |  SD = {var_stats['SD']:>8.3f}\n"
                rep += f"    Min = {var_stats['Min']:>8.3f}  |  Q1 = {var_stats['Q1']:>8.3f}  |  Median = {var_stats['Median']:>8.3f}  |  Q3 = {var_stats['Q3']:>8.3f}  |  Max = {var_stats['Max']:>8.3f}\n"
                rep += f"    Skewness = {var_stats['Skewness']:>7.3f}  |  Kurtosis = {var_stats['Kurtosis']:>7.3f}\n"
                rep += f"{'-'*90}\n"
                var_num += 1

            rep += f"\n{'='*90}\n"
            rep += "INTERPRETATION GUIDELINES\n"
            rep += f"{'='*90}\n"
            rep += "• Skewness: |value| < 1 (excellent), < 2 (acceptable), < 3 (marginal)\n"
            rep += "• Kurtosis: |value| < 3 (excellent), < 7 (acceptable), < 10 (marginal)\n"
            rep += "• Missing: High missing rates (>5%) may indicate data quality issues\n"
            rep += "• N: Valid sample size after removing missing values\n"

            self.current_report = rep
            report_area.setPlainText(rep)
            return

        elif res['mode'] == 'CORRELATION':
            r = "<h1>📊 CORRELATION ANALYSIS RESULTS</h1>"
            r += f"<p><b>Analysis Type:</b> {res['method'].capitalize()} Correlation</p>"
            r += f"<p><b>Sample Size:</b> N = {res['n']}</p>"
            r += f"<p><b>Variables:</b> {len(res['variables'])}</p>"

            # Descriptive statistics
            r += "<h2>Descriptive Statistics</h2>"
            r += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
            r += "<tr style='background-color: #34495e; color: white;'>"
            r += "<th>Variable</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th></tr>"

            for var in res['variables']:
                desc = res['descriptives'][var]
                r += f"<tr>"
                r += f"<td><b>{var}</b></td>"
                r += f"<td>{desc['mean']:.3f}</td>"
                r += f"<td>{desc['sd']:.3f}</td>"
                r += f"<td>{desc['min']:.3f}</td>"
                r += f"<td>{desc['max']:.3f}</td>"
                r += "</tr>"
            r += "</table>"

            # Correlation matrix with p-values
            r += "<h2>Correlation Matrix</h2>"
            r += "<p><i>Correlation coefficients with significance levels (* p < .05, ** p < .01, *** p < .001)</i></p>"

            corr_matrix = res['corr_matrix']
            p_matrix = res['p_matrix']

            r += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
            r += "<tr style='background-color: #34495e; color: white;'>"
            r += "<th>Variable</th>"
            for var in res['variables']:
                r += f"<th>{var}</th>"
            r += "</tr>"

            for i, var1 in enumerate(res['variables']):
                r += "<tr>"
                r += f"<td style='background-color: #ecf0f1;'><b>{var1}</b></td>"
                for j, var2 in enumerate(res['variables']):
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = p_matrix.iloc[i, j]

                    if i == j:
                        r += "<td style='background-color: #bdc3c7;'>1.000</td>"
                    else:
                        sig_stars = ""
                        if p_val < 0.001:
                            sig_stars = "***"
                        elif p_val < 0.01:
                            sig_stars = "**"
                        elif p_val < 0.05:
                            sig_stars = "*"

                        if abs(corr_val) >= 0.7:
                            color = "#e74c3c" if corr_val > 0 else "#3498db"
                        elif abs(corr_val) >= 0.4:
                            color = "#f39c12" if corr_val > 0 else "#9b59b6"
                        else:
                            color = "inherit"

                        r += f"<td style='color: {color};'>{corr_val:.3f}{sig_stars}</td>"
                r += "</tr>"
            r += "</table>"

            # P-values table
            r += "<h2>P-Values (Two-tailed)</h2>"
            r += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
            r += "<tr style='background-color: #34495e; color: white;'>"
            r += "<th>Variable</th>"
            for var in res['variables']:
                r += f"<th>{var}</th>"
            r += "</tr>"

            for i, var1 in enumerate(res['variables']):
                r += "<tr>"
                r += f"<td style='background-color: #ecf0f1;'><b>{var1}</b></td>"
                for j, var2 in enumerate(res['variables']):
                    if i == j:
                        r += "<td style='background-color: #bdc3c7;'>-</td>"
                    else:
                        p_val = p_matrix.iloc[i, j]
                        r += f"<td>{p_val:.4f}</td>"
                r += "</tr>"
            r += "</table>"

            # Interpretation guide
            r += "<h2>Interpretation Guide</h2>"
            r += "<ul>"
            r += "<li><b>Correlation strength:</b> 0.0-0.3 (weak), 0.3-0.7 (moderate), 0.7-1.0 (strong)</li>"
            r += "<li><b>Significance:</b> * p < .05, ** p < .01, *** p < .001</li>"
            r += "<li><b>Method:</b> " + res['method'].capitalize() + " correlation"
            if res['method'].lower() == 'pearson':
                r += " measures linear relationships (assumes normality)"
            elif res['method'].lower() == 'spearman':
                r += " measures monotonic relationships (rank-based, non-parametric)"
            else:
                r += " measures ordinal associations (rank-based, robust to outliers)"
            r += "</li>"
            r += "</ul>"

            self.current_report = r
            report_area.setHtml(r)
            return

        elif res['mode'] == 'TTEST':
            r = "<h1>📊 T-TEST RESULTS</h1>"
            r += f"<p><b>Test Type:</b> {res['test_type']}</p>"

            if res['test_type'] == 'Independent Samples':
                r += f"<p><b>Dependent Variable:</b> {res['dv']}</p>"
                r += f"<p><b>Grouping Variable:</b> {res['group_var']}</p><hr>"

                r += "<h2>📊 Group Statistics</h2>"
                r += "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%;'>"
                r += "<tr style='background-color:#34495e;color:white;'>"
                r += "<th>Group</th><th>N</th><th>Mean</th><th>SD</th></tr>"
                r += f"<tr><td><b>{res['group1_name']}</b></td><td>{res['group1_n']}</td>"
                r += f"<td>{res['group1_mean']:.3f}</td><td>{res['group1_sd']:.3f}</td></tr>"
                r += f"<tr><td><b>{res['group2_name']}</b></td><td>{res['group2_n']}</td>"
                r += f"<td>{res['group2_mean']:.3f}</td><td>{res['group2_sd']:.3f}</td></tr>"
                r += "</table>"

                r += "<h2>🔍 Levene's Test for Equality of Variances</h2>"
                r += f"<p>F = {res['levene_stat']:.3f}, p = {res['levene_p']:.4f}"
                if res['levene_p'] > 0.05:
                    r += " <span style='color:#27ae60;font-weight:bold;'>(Equal variances assumed ✓)</span></p>"
                else:
                    r += " <span style='color:#e74c3c;font-weight:bold;'>(Equal variances NOT assumed ✗)</span></p>"

                r += "<h2>📈 Independent Samples T-Test</h2>"
                r += "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%;'>"
                r += "<tr style='background-color:#34495e;color:white;'>"
                r += "<th>t</th><th>df</th><th>p-value</th><th>Cohen's d</th><th>Interpretation</th></tr>"
                sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else "ns"
                interpretation = "Significant" if res['p_value'] < 0.05 else "Not significant"
                color = "#27ae60" if res['p_value'] < 0.05 else "#e74c3c"
                r += f"<tr><td>{res['t_statistic']:.3f}</td><td>{res['df']}</td>"
                r += f"<td>{res['p_value']:.4f} {sig}</td><td>{res['cohens_d']:.3f}</td>"
                r += f"<td style='color:{color};font-weight:bold;'>{interpretation}</td></tr></table>"

                mean_diff = res['group1_mean'] - res['group2_mean']
                r += f"<p><b>Mean Difference:</b> {mean_diff:.3f} ({res['group1_name']} - {res['group2_name']})</p>"

            elif res['test_type'] == 'Paired Samples':
                r += f"<p><b>Variable 1:</b> {res['var1']}</p>"
                r += f"<p><b>Variable 2:</b> {res['var2']}</p>"
                r += f"<p><b>N:</b> {res['n']}</p><hr>"

                r += "<h2>📈 Paired Samples T-Test</h2>"
                r += "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%;'>"
                r += "<tr style='background-color:#34495e;color:white;'>"
                r += "<th>Mean Difference</th><th>SD Difference</th><th>t</th><th>df</th><th>p-value</th><th>Interpretation</th></tr>"
                sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else "ns"
                interpretation = "Significant" if res['p_value'] < 0.05 else "Not significant"
                color = "#27ae60" if res['p_value'] < 0.05 else "#e74c3c"
                r += f"<tr><td>{res['mean_diff']:.3f}</td><td>{res['sd_diff']:.3f}</td>"
                r += f"<td>{res['t_statistic']:.3f}</td><td>{res['df']}</td>"
                r += f"<td>{res['p_value']:.4f} {sig}</td>"
                r += f"<td style='color:{color};font-weight:bold;'>{interpretation}</td></tr></table>"

            else:  # One Sample
                r += f"<p><b>Variable:</b> {res['var']}</p>"
                r += f"<p><b>Test Value (μ₀):</b> {res['test_value']}</p>"
                r += f"<p><b>Sample Mean:</b> {res['sample_mean']:.3f}</p>"
                r += f"<p><b>Sample SD:</b> {res['sample_sd']:.3f}</p>"
                r += f"<p><b>N:</b> {res['n']}</p><hr>"

                r += "<h2>📈 One-Sample T-Test</h2>"
                r += "<table border='1' cellpadding='8' style='border-collapse:collapse;width:100%;'>"
                r += "<tr style='background-color:#34495e;color:white;'>"
                r += "<th>t</th><th>df</th><th>p-value</th><th>Interpretation</th></tr>"
                sig = "***" if res['p_value'] < 0.001 else "**" if res['p_value'] < 0.01 else "*" if res['p_value'] < 0.05 else "ns"
                interpretation = "Significant" if res['p_value'] < 0.05 else "Not significant"
                color = "#27ae60" if res['p_value'] < 0.05 else "#e74c3c"
                r += f"<tr><td>{res['t_statistic']:.3f}</td><td>{res['df']}</td>"
                r += f"<td>{res['p_value']:.4f} {sig}</td>"
                r += f"<td style='color:{color};font-weight:bold;'>{interpretation}</td></tr></table>"

                mean_diff = res['sample_mean'] - res['test_value']
                r += f"<p><b>Difference from Test Value:</b> {mean_diff:.3f}</p>"

            r += "<h2>📖 Effect Size Interpretation (Cohen's d)</h2>"
            r += "<table border='1' cellpadding='10' style='border-collapse:collapse;width:100%;'>"
            r += "<tr style='background-color:#34495e;color:white;'><th>Effect Size</th><th>Interpretation</th></tr>"
            r += "<tr><td>|d| < 0.2</td><td>Negligible</td></tr>"
            r += "<tr><td>0.2 ≤ |d| < 0.5</td><td>Small</td></tr>"
            r += "<tr><td>0.5 ≤ |d| < 0.8</td><td>Medium</td></tr>"
            r += "<tr><td>|d| ≥ 0.8</td><td>Large</td></tr>"
            r += "</table>"

            r += "<h2>📋 Assumptions & Notes</h2>"
            r += "<ul>"
            if res['test_type'] == 'Independent Samples':
                r += "<li><b>Independence:</b> Observations are independent between groups.</li>"
                r += "<li><b>Normality:</b> Data in each group should be approximately normally distributed.</li>"
                r += "<li><b>Homogeneity of variance:</b> Checked using Levene's test.</li>"
            elif res['test_type'] == 'Paired Samples':
                r += "<li><b>Paired observations:</b> Each observation in one group is paired with an observation in the other.</li>"
                r += "<li><b>Normality of differences:</b> The differences between pairs should be normally distributed.</li>"
            else:
                r += "<li><b>Independence:</b> Observations are independent.</li>"
                r += "<li><b>Normality:</b> Data should be approximately normally distributed.</li>"
            r += "<li><b>Significance level:</b> α = 0.05 (two-tailed)</li>"
            r += "</ul>"

            self.current_report = r
            report_area.setHtml(r)
            return

        elif res['mode'] == 'REGRESSION':
            reg_type = res.get('reg_type', 'standard')
            
            if reg_type == 'standard':
                rep += f"STANDARD REGRESSION ANALYSIS\n\n"
                rep += f"DEPENDENT VARIABLE: {res['dv']}\n"
                rep += f"INDEPENDENT VARIABLES: {', '.join(res['ivs'])}\n\n"
                
                rep += f"{'='*90}\n"
                rep += "MODEL SUMMARY\n"
                rep += f"{'='*90}\n\n"
                
                rep += f"{'Metric':<30} {'Value':>15} {'Interpretation':>30}\n"
                rep += f"{'-'*90}\n"
                rep += f"{'R-squared':<30} {res['r_squared']:>15.4f} {('Excellent' if res['r_squared'] > 0.7 else 'Good' if res['r_squared'] > 0.5 else 'Moderate' if res['r_squared'] > 0.3 else 'Weak'):>30}\n"
                rep += f"{'Adjusted R-squared':<30} {res['adj_r_squared']:>15.4f}\n"
                rep += f"{'F-statistic':<30} {res['f_stat']:>15.4f}\n"
                rep += f"{'F p-value':<30} {res['f_pvalue']:>15.4f} {('***' if res['f_pvalue'] < 0.001 else '**' if res['f_pvalue'] < 0.01 else '*' if res['f_pvalue'] < 0.05 else 'ns'):>30}\n"
                
                rep += f"\n{'='*90}\n"
                rep += "REGRESSION COEFFICIENTS\n"
                rep += f"{'='*90}\n\n"
                
                # FIX #1: Better regression table formatting with proper alignment
                rep += f"{'Predictor':<50} {'B':>10} {'SE':>10} {'t':>10} {'p':>10} {'Sig':>5}\n"
                rep += f"{'-'*100}\n"
                
                for var in res['params'].index:
                    b = float(res['params'][var])
                    se = float(res['std_err'][var])
                    t = float(res['tvalues'][var])
                    p = float(res['pvalues'][var])
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    
                    # Truncate long names but maintain alignment
                    var_display = var[:47] + '...' if len(var) > 50 else var
                    rep += f"{var_display:<50} {b:>10.4f} {se:>10.4f} {t:>10.3f} {p:>10.4f} {sig:>5}\n"
                
                rep += f"\n{'Significance codes: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant'}\n"
                
            elif reg_type == 'hierarchical':
                rep += f"HIERARCHICAL REGRESSION ANALYSIS\n\n"
                rep += f"DEPENDENT VARIABLE: {res['dv']}\n\n"
                
                rep += f"{'='*90}\n"
                rep += "MODEL COMPARISON\n"
                rep += f"{'='*90}\n\n"
                
                rep += f"{'Block':>6} {'R²':>10} {'Adj R²':>10} {'ΔR²':>10} {'F-change':>12} {'p-value':>12}\n"
                rep += f"{'-'*90}\n"
                
                for block in res['blocks']:
                    block_num = block['block']
                    r2 = block['r_squared']
                    adj_r2 = block['adj_r_squared']
                    
                    if 'r2_change' in block:
                        r2_change = block['r2_change']
                        f_change = block['f_change']
                        p_change = block['p_change']
                        sig = '***' if p_change < 0.001 else '**' if p_change < 0.01 else '*' if p_change < 0.05 else 'ns'
                        rep += f"{block_num:>6} {r2:>10.4f} {adj_r2:>10.4f} {r2_change:>10.4f} {f_change:>12.3f} {p_change:>10.4f} {sig}\n"
                    else:
                        rep += f"{block_num:>6} {r2:>10.4f} {adj_r2:>10.4f} {'--':>10} {'--':>12} {'--':>12}\n"
                
                rep += f"\n\n{'='*90}\n"
                # ALL blocks
                for block in res['blocks']:
                    rep += f"\n\n{'='*110}\n"
                    rep += f"BLOCK {block['block']} (R² = {block['r_squared']:.4f})\n"
                    if 'new_variables' in block:
                        rep += f"New: {', '.join(block['new_variables'])}\n"
                    rep += f"All: {', '.join(block['variables'])}\n"
                    rep += f"{'='*110}\n\n"
                    rep += f"{'Predictor':<40} {'B':>12} {'SE':>10} {'β':>12} {'t':>10} {'p':>10} {'Sig':>5}\n"
                    rep += f"{'-'*110}\n"
                    for var in block['params'].index:
                        b, se = float(block['params'][var]), float(block['std_err'][var])
                        t, p = float(block['tvalues'][var]), float(block['pvalues'][var])
                        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                        beta_str = f"{'-':>12}" if var == 'const' else f"{float(block['standardized_params'][var]):>12.4f}"
                        var_display = var[:37] + '...' if len(var) > 40 else var
                        rep += f"{var_display:<40} {b:>12.4f} {se:>10.4f} {beta_str} {t:>10.3f} {p:>10.4f} {sig:>5}\n"
                
            elif reg_type == 'moderation':
                rep += f"MODERATION ANALYSIS\n\n"
                rep += f"DEPENDENT VARIABLE: {res['dv']}\n"
                rep += f"FOCAL PREDICTOR: {res['focal_predictor']}\n"
                rep += f"MODERATOR: {res['moderator']}\n"
                rep += f"INTERACTION TERM: {res['interaction_term']}\n\n"
                
                rep += f"{'='*90}\n"
                rep += "MODEL COMPARISON\n"
                rep += f"{'='*90}\n\n"
                
                m1 = res['model1']
                m2 = res['model2']
                
                rep += f"{'Model':<20} {'R²':>12} {'Adj R²':>12} {'F-change':>12} {'p-value':>12}\n"
                rep += f"{'-'*90}\n"
                rep += f"{'Model 1 (Main)':<20} {m1['r_squared']:>12.4f} {m1['adj_r_squared']:>12.4f} {'--':>12} {'--':>12}\n"
                rep += f"{'Model 2 (Interact)':<20} {m2['r_squared']:>12.4f} {m2['adj_r_squared']:>12.4f} {res['f_change']:>12.3f} {res['p_change']:>12.4f} {'***' if res['p_change'] < 0.001 else '**' if res['p_change'] < 0.01 else '*' if res['p_change'] < 0.05 else 'ns'}\n"
                
                rep += f"\n\n{'='*90}\n"
                rep += "MODERATION COEFFICIENTS (Model 2)\n"
                rep += f"{'='*90}\n\n"
                
                rep += f"{'Variable':<50} {'B':>10} {'SE':>10} {'t':>10} {'p':>10} {'Sig':>5}\n"
                rep += f"{'-'*100}\n"
                
                for var in m2['params'].index:
                    b = float(m2['params'][var])
                    se = float(m2['std_err'][var])
                    t = float(m2['tvalues'][var])
                    p = float(m2['pvalues'][var])
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    var_display = var[:47] + '...' if len(var) > 50 else var
                    rep += f"{var_display:<50} {b:>10.4f} {se:>10.4f} {t:>10.3f} {p:>10.4f} {sig:>5}\n"
                
                rep += f"\n\n{'='*90}\n"
                rep += "INTERPRETATION\n"
                rep += f"{'='*90}\n"
                if res['p_change'] < 0.05:
                    rep += f"✓ The interaction is SIGNIFICANT (p = {res['p_change']:.4f})\n"
                    rep += f"The moderator ({res['moderator']}) significantly influences the relationship\n"
                    rep += f"between {res['focal_predictor']} and {res['dv']}.\n"
                else:
                    rep += f"✗ The interaction is NOT SIGNIFICANT (p = {res['p_change']:.4f})\n"
                    rep += f"The moderator ({res['moderator']}) does not significantly influence the relationship\n"
                    rep += f"between {res['focal_predictor']} and {res['dv']}.\n"
            
            # Common assumption tests
            rep += f"\n\n{'='*90}\n"
            rep += "REGRESSION ASSUMPTIONS DIAGNOSTICS\n"
            rep += f"{'='*90}\n\n"
            
            # Normality
            rep += "1. NORMALITY OF RESIDUALS\n"
            rep += f"{'-'*90}\n"
            rep += f"Test: {res['normality_test']}\n"
            rep += f"Statistic: {res['normality_stat']:.4f}\n"
            rep += f"p-value: {res['normality_p']:.4f}\n"
            norm_result = "✓ PASSED" if res['normality_p'] > 0.05 else "✗ VIOLATED"
            rep += f"Result: {norm_result} (p > 0.05 indicates normality)\n"
            rep += f"Interpretation: {'Residuals are normally distributed' if res['normality_p'] > 0.05 else 'Residuals deviate from normality - consider transformations'}\n\n"
            
            # Homoscedasticity
            rep += "2. HOMOSCEDASTICITY (Equal Variance)\n"
            rep += f"{'-'*90}\n"
            rep += f"Test: Breusch-Pagan\n"
            rep += f"LM Statistic: {res['bp_stat']:.4f}\n"
            rep += f"p-value: {res['bp_p']:.4f}\n"
            homo_result = "✓ PASSED" if res['bp_p'] > 0.05 else "✗ VIOLATED"
            rep += f"Result: {homo_result} (p > 0.05 indicates homoscedasticity)\n"
            rep += f"Interpretation: {'Equal variance across predictions' if res['bp_p'] > 0.05 else 'Heteroscedasticity detected - consider robust standard errors'}\n\n"
            
            # Multicollinearity
            rep += "3. MULTICOLLINEARITY (VIF)\n"
            rep += f"{'-'*90}\n"
            if res['vif'] is not None:
                rep += f"{'Variable':<30} {'VIF':>15} {'Status':>20}\n"
                rep += f"{'-'*90}\n"
                vif_ok = True
                for _, row in res['vif'].iterrows():
                    vif_val = row['VIF']
                    status = "Excellent" if vif_val < 5 else "Acceptable" if vif_val < 10 else "PROBLEMATIC"
                    vif_ok = False if vif_val > 10 else vif_ok
                    rep += f"{row['Variable']:<30} {vif_val:>15.3f} {status:>20}\n"
                rep += f"\n{'PASSED' if vif_ok else 'VIOLATED'}\n\n"
            else:
                rep += "N/A\n\n"

            rep += "4. INDEPENDENCE (Autocorrelation)\n"
            rep += f"{'-'*90}\n"
            rep += f"Test: Durbin-Watson\n"
            rep += f"DW Statistic: {res['dw_stat']:.4f}\n"
            auto_ok = 1.5 < res['dw_stat'] < 2.5
            auto_result = "✓ PASSED" if auto_ok else "✗ VIOLATED"
            rep += f"Result: {auto_result} (1.5-2.5 acceptable, ~2.0 ideal)\n"
            interpretation = 'No autocorrelation' if auto_ok else ('Positive autocorrelation' if res['dw_stat'] < 1.5 else 'Negative autocorrelation')
            rep += f"Interpretation: {interpretation}\n\n"
            
            rep += "5. LINEARITY\n"
            rep += f"{'-'*90}\n"
            rep += "Assessment: See Residual Plots tab for visual inspection\n"
            rep += "Criterion: Random scatter around zero line indicates linearity\n\n"
            
            rep += f"{'='*90}\n"
            rep += "REFERENCES\n"
            rep += f"{'='*90}\n"
            rep += "• Hair, J. F., et al. (2010). Multivariate Data Analysis (7th ed.).\n"
            rep += "• Durbin, J., & Watson, G. S. (1950, 1951). Testing for serial correlation.\n"
            rep += "• Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity.\n"
            rep += "• O'Brien, R. M. (2007). A caution regarding rules of thumb for VIF.\n"
            rep += "• Aiken, L. S., & West, S. G. (1991). Multiple regression (moderation).\n"
            
            self.create_regression_plots(res['residuals'], res['fitted'], plot_layout)
            
        elif mode == 'CORRELATION':
            r = "<h1>📊 CORRELATION ANALYSIS RESULTS</h1>"
            r += f"<p><b>Analysis Type:</b> {results['method'].capitalize()} Correlation</p>"
            r += f"<p><b>Sample Size:</b> N = {results['n']}</p>"
            r += f"<p><b>Variables:</b> {len(results['variables'])}</p>"

            # Descriptive statistics
            r += "<h2>Descriptive Statistics</h2>"
            r += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
            r += "<tr style='background-color: #34495e; color: white;'>"
            r += "<th>Variable</th><th>Mean</th><th>SD</th><th>Min</th><th>Max</th></tr>"

            for var in results['variables']:
                desc = results['descriptives'][var]
                r += f"<tr>"
                r += f"<td><b>{var}</b></td>"
                r += f"<td>{desc['mean']:.3f}</td>"
                r += f"<td>{desc['sd']:.3f}</td>"
                r += f"<td>{desc['min']:.3f}</td>"
                r += f"<td>{desc['max']:.3f}</td>"
                r += "</tr>"
            r += "</table>"

            # Correlation matrix with p-values
            r += "<h2>Correlation Matrix</h2>"
            r += "<p><i>Correlation coefficients with significance levels (* p < .05, ** p < .01, *** p < .001)</i></p>"

            corr_matrix = results['corr_matrix']
            p_matrix = results['p_matrix']

            r += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
            r += "<tr style='background-color: #34495e; color: white;'>"
            r += "<th>Variable</th>"
            for var in results['variables']:
                r += f"<th>{var}</th>"
            r += "</tr>"

            for i, var1 in enumerate(results['variables']):
                r += "<tr>"
                r += f"<td style='background-color: #ecf0f1;'><b>{var1}</b></td>"
                for j, var2 in enumerate(results['variables']):
                    corr_val = corr_matrix.iloc[i, j]
                    p_val = p_matrix.iloc[i, j]

                    if i == j:
                        r += "<td style='background-color: #bdc3c7;'>1.000</td>"
                    else:
                        # Format with significance stars
                        sig_stars = ""
                        if p_val < 0.001:
                            sig_stars = "***"
                        elif p_val < 0.01:
                            sig_stars = "**"
                        elif p_val < 0.05:
                            sig_stars = "*"

                        # Color code by strength
                        if abs(corr_val) >= 0.7:
                            color = "#e74c3c" if corr_val > 0 else "#3498db"
                        elif abs(corr_val) >= 0.4:
                            color = "#f39c12" if corr_val > 0 else "#9b59b6"
                        else:
                            color = "inherit"

                        r += f"<td style='color: {color};'>{corr_val:.3f}{sig_stars}</td>"
                r += "</tr>"
            r += "</table>"

            # P-values table
            r += "<h2>P-Values (Two-tailed)</h2>"
            r += "<table border='1' cellpadding='8' style='border-collapse: collapse; width: 100%;'>"
            r += "<tr style='background-color: #34495e; color: white;'>"
            r += "<th>Variable</th>"
            for var in results['variables']:
                r += f"<th>{var}</th>"
            r += "</tr>"

            for i, var1 in enumerate(results['variables']):
                r += "<tr>"
                r += f"<td style='background-color: #ecf0f1;'><b>{var1}</b></td>"
                for j, var2 in enumerate(results['variables']):
                    if i == j:
                        r += "<td style='background-color: #bdc3c7;'>-</td>"
                    else:
                        p_val = p_matrix.iloc[i, j]
                        r += f"<td>{p_val:.4f}</td>"
                r += "</tr>"
            r += "</table>"

            # Interpretation guide
            r += "<h2>Interpretation Guide</h2>"
            r += "<ul>"
            r += "<li><b>Correlation strength:</b> 0.0-0.3 (weak), 0.3-0.7 (moderate), 0.7-1.0 (strong)</li>"
            r += "<li><b>Significance:</b> * p < .05, ** p < .01, *** p < .001</li>"
            r += "<li><b>Method:</b> " + results['method'].capitalize() + " correlation measures "
            if results['method'] == 'pearson':
                r += "linear relationships (assumes normality)"
            elif results['method'] == 'spearman':
                r += "monotonic relationships (rank-based, non-parametric)"
            else:  # kendall
                r += "ordinal associations (rank-based, robust to outliers)"
            r += "</li>"
            r += "</ul>"

            self.current_report = r
            reportarea.setHtml(r)
            return

        elif mode == 'EFA':
            rep += f"EXTRACTION METHOD: {res['extraction'].upper()}\n"
            rep += f"ROTATION METHOD: {res['rotation'].upper()}\n"
            rep += f"NUMBER OF FACTORS REQUESTED: {res['n_factors']}\n"
            rep += f"NUMBER OF FACTORS EXTRACTED: {res['actual_factors']}\n\n"
            
            rep += f"{'='*90}\n"
            rep += "1. SAMPLING ADEQUACY\n"
            rep += f"{'='*90}\n\n"
            rep += f"Kaiser-Meyer-Olkin (KMO) Measure: {res['kmo']:.3f}\n"
            kmo_interp = 'Marvelous' if res['kmo'] > 0.9 else 'Meritorious' if res['kmo'] > 0.8 else 'Middling' if res['kmo'] > 0.7 else 'Mediocre' if res['kmo'] > 0.6 else 'Unacceptable'
            rep += f"Interpretation: {kmo_interp} (Kaiser, 1974)\n\n"
            rep += f"Bartlett's Test of Sphericity p-value: {res['bartlett_p']:.6f}\n"
            rep += f"Interpretation: {'Data is suitable for factor analysis' if res['bartlett_p'] < 0.05 else 'Data may not be suitable for factor analysis'}\n\n"
            
            rep += f"{'='*90}\n"
            rep += "2. TOTAL VARIANCE EXPLAINED\n"
            rep += f"{'='*90}\n\n"
            rep += f"{'Factor':>8} {'Eigenvalue':>15} {'% Variance':>15} {'Cumulative %':>15}\n"
            rep += f"{'-'*90}\n"
            
            num_to_display = min(10, len(res['eigenvalues']), res['actual_factors'])
            for i in range(num_to_display):
                ev = res['eigenvalues'][i]
                pct = res['pct_var'][i]
                cum = res['cum_var'][i]
                rep += f"{i+1:>8} {ev:>15.3f} {pct:>14.2f}% {cum:>14.2f}%\n"
            
            rep += f"\nKaiser Criterion: Retain factors with eigenvalue > 1.0\n\n"
            
            # FIX #1: Better EFA loadings display with item numbers
            rep += f"{'='*90}\n"
            rep += "3. FACTOR LOADINGS (|loading| > 0.30 displayed)\n"
            rep += f"{'='*90}\n\n"
            
            loadings = res['loadings'].copy()
            variable_names = res['variable_names']
            
            # Create item number mapping
            rep += "ITEM KEY:\n"
            rep += f"{'-'*90}\n"
            for idx, var_name in enumerate(variable_names, 1):
                # Truncate long names for display
                display_name = var_name[:70] + '...' if len(var_name) > 70 else var_name
                rep += f"Item {idx:2d}: {display_name}\n"
            rep += f"\n{'-'*90}\n"
            rep += "FACTOR LOADING MATRIX:\n"
            rep += f"{'-'*90}\n\n"
            
            # Create compact table with item numbers
            rep += f"{'Item':>6}"
            for col in loadings.columns:
                rep += f"{col:>12}"
            rep += "\n"
            rep += f"{'-'*90}\n"
            
            for idx, var_name in enumerate(variable_names, 1):
                rep += f"{idx:>6}"
                for col in loadings.columns:
                    val = loadings.loc[var_name, col]
                    if abs(val) >= 0.3:
                        rep += f"{val:>12.3f}"
                    else:
                        rep += f"{'':>12}"
                rep += "\n"
            
            rep += f"\n{'='*90}\n"
            rep += "4. COMMUNALITIES (h²)\n"
            rep += f"{'='*90}\n\n"
            
            comm = res['communalities']
            rep += f"{'Item':>6} {'h²':>10} {'Interpretation':>20} {'Variable Name (truncated)':>40}\n"
            rep += f"{'-'*90}\n"
            for idx, (var, val) in enumerate(comm.items(), 1):
                interp = 'Excellent' if val > 0.7 else 'Good' if val > 0.5 else 'Fair' if val > 0.3 else 'Poor'
                var_truncated = var[:37] + '...' if len(var) > 40 else var
                rep += f"{idx:>6} {val:>10.3f} {interp:>20} {var_truncated:>40}\n"
            
            # FIX #2: Add Reliability Analysis to EFA Report
            rep += f"\n{'='*90}\n"
            rep += f"\n{'='*90}\n"
            rep += "5. RELIABILITY (Per-Factor)\n"
            rep += f"{'='*90}\n\n"
            if 'original_df' in res:
                rep += f"{'Factor':<20} {'α':>15} {'CR':>12} {'Items':>8} {'Interpretation':>25}\n"
                rep += f"{'-'*90}\n"
                for idx in range(res['n_factors']):
                    fname = loadings.columns[idx]
                    items = loadings.index[loadings.iloc[:, idx].abs() > 0.3].tolist()
                    if len(items) >= 2:
                        try:
                            alpha, _ = calculate_cronbach_alpha(res['original_df'][items])
                            fl = loadings.iloc[:, idx].abs()[loadings.iloc[:, idx].abs() > 0.3]
                            cr = calculate_composite_reliability(fl)
                            interp = "Excellent" if alpha >= 0.9 else "Good" if alpha >= 0.8 else "Acceptable" if alpha >= 0.7 else "Poor"
                            rep += f"{fname:<20} {alpha:>15.3f} {cr:>12.3f} {len(items):>8} {interp:>25}\n"
                        except:
                            rep += f"{fname:<20} {'Error':>15} {'--':>12} {len(items):>8} {'N/A':>25}\n"
                    else:
                        rep += f"{fname:<20} {'N/A':>15} {'--':>12} {len(items):>8} {'< 2':>25}\n"

            rep += "INTERPRETATION GUIDELINES\n"
            rep += f"{'='*90}\n"
            rep += "Factor Loadings (Hair et al., 2010):\n"
            rep += "  • |λ| > 0.70: Excellent\n"
            rep += "  • |λ| > 0.60: Very good\n"
            rep += "  • |λ| > 0.50: Good\n"
            rep += "  • |λ| > 0.40: Fair\n"
            rep += "  • |λ| > 0.30: Minimal acceptable\n\n"
            rep += "Communalities:\n"
            rep += "  • h² > 0.50: Adequate variance extracted\n"
            rep += "  • h² < 0.50: Consider removal or refinement\n"
            rep += "\nReliability:\n"
            rep += "  • Cronbach's α > 0.70: Acceptable internal consistency\n"
            rep += "  • CR > 0.70: Acceptable composite reliability\n"
            

            # Scree
            if 'eigenvalues' in res:
                for i in reversed(range(plot_layout.count())):
                    w = plot_layout.itemAt(i).widget()
                    if w: w.setParent(None)
                fig, ax = plt.subplots(figsize=(10, 6))
                ev = res['eigenvalues']
                ax.plot(range(1, len(ev)+1), ev, 'bo-', linewidth=2, markersize=8)
                ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Kaiser')
                ax.set_xlabel('Factor', fontsize=12)
                ax.set_ylabel('Eigenvalue', fontsize=12)
                ax.set_title('Scree Plot', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.legend()
                plt.tight_layout()
                canvas = FigureCanvas(fig)
                plot_layout.addWidget(canvas)
            
        elif res['mode'] == 'CFA':
            s = res['stats']
            rep += f"{'='*90}\n"
            rep += "CONFIRMATORY FACTOR ANALYSIS\n"
            rep += f"{'='*90}\n\n"
                
            # Model Fit Indices
            rep += "MODEL FIT INDICES\n"
            rep += f"{'-'*90}\n"
            rep += f"{'Fit Index':<40} {'Value':>15} {'Cutoff':>20}\n"
            rep += f"{'-'*90}\n"
            rep += f"{'Comparative Fit Index (CFI)':<40} {s.get('CFI', 0):>15.3f} {'≥ 0.95 (good)':>20}\n"
            rep += f"{'Tucker-Lewis Index (TLI)':<40} {s.get('TLI', 0):>15.3f} {'≥ 0.95 (good)':>20}\n"
            rep += f"{'RMSEA':<40} {s.get('RMSEA', 0):>15.3f} {'≤ 0.06 (good)':>20}\n"
            rep += f"{'SRMR':<40} {s.get('SRMR', 0):>15.3f} {'≤ 0.08 (good)':>20}\n"
            rep += f"{'Chi-square':<40} {s.get('chi2', 0):>15.3f}\n"
            rep += f"{'Degrees of Freedom':<40} {s.get('DoF', 0):>15.0f}\n"
            rep += f"{'Chi-square/df':<40} {s.get('chi2/DoF', 0):>15.3f} {'< 3 (good)':>20}\n"
            rep += "\n"
                
            # Factor Loadings (lavaan-style)
            if 'loadings' in res:
                rep += f"{'='*90}\n"
                rep += "FACTOR LOADINGS (Standardized)\n"
                rep += f"{'='*90}\n\n"
                    
                loadings = res['loadings']
                rev_map = res.get('reverse_mapping', {})
                    
                # Group by latent variable
                for latent in loadings['rval'].unique():
                    rep += f"{latent}:\n"
                    rep += f"{'-'*70}\n"
                    rep += f"{'Indicator':<30} {'Estimate':>12} {'Std.Err':>12} {'z-value':>10} {'p-value':>10}\n"
                    rep += f"{'-'*70}\n"
                        
                    factor_items = loadings[loadings['rval'] == latent]
                    for _, row in factor_items.iterrows():
                        indicator = rev_map.get(row['lval'], row['lval'])
                        est = safe_float(row.get('Estimate', row.get('Est.', 0)), 0.0)
                        std_est = safe_float(row.get('Est. Std', 0), 0.0)
                        se = safe_float(row.get('Std. Err', 0), 0.001)
                        z = est / se if se > 0.001 else 0.0
                        p = 2 * (1 - stats.norm.cdf(abs(z)))
                        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                        rep += f"{indicator:<30} {std_est:>12.3f} {se:>12.3f} {z:>10.3f} {p:>10.4f} {sig}\n"
                    rep += "\n"
                
            # Latent Variable Covariances
            if 'covariances' in res:
                covariances = res['covariances']
                latent_covs = covariances[(covariances['lval'].isin(loadings['rval'].unique())) & 
                                            (covariances['rval'].isin(loadings['rval'].unique()))]
                    
                if not latent_covs.empty:
                    rep += f"{'='*90}\n"
                    rep += "LATENT VARIABLE COVARIANCES/CORRELATIONS\n"
                    rep += f"{'='*90}\n\n"
                    rep += f"{'Variable 1':<20} {'Variable 2':<20} {'Covariance':>15} {'Correlation':>15}\n"
                    rep += f"{'-'*75}\n"
                        
                    for _, row in latent_covs.iterrows():
                        if row['lval'] != row['rval']:  # Only covariances, not variances
                            cov = safe_float(row.get('Estimate', row.get('Est.', 0)), 0.0)
                            corr = safe_float(row.get('Est. Std', 0), 0.0)
                            rep += f"{row['lval']:<20} {row['rval']:<20} {cov:>15.3f} {corr:>15.3f}\n"
                    rep += "\n"
                
            # Reliability Metrics
            rep += f"{'='*90}\n"
            rep += "RELIABILITY METRICS (Per Factor)\n"
            rep += f"{'='*90}\n\n"
            rep += "Note: Calculate Composite Reliability (CR) and AVE from standardized loadings\n"
            rep += "      CR = (Σλ)² / [(Σλ)² + Σ(1-λ²)]\n"
            rep += "      AVE = Σλ² / [Σλ² + Σ(1-λ²)]\n\n"
                
            if 'loadings' in res:
                rep += f"{'Factor':<20} {'CR':>15} {'AVE':>15} {'Items':>10} {'Status':>20}\n"
                rep += f"{'-'*85}\n"
                    
                for latent in loadings['rval'].unique():
                    factor_items = loadings[loadings['rval'] == latent]
                    std_loadings = factor_items['Est. Std'].values
                        
                    # Calculate CR
                    sum_loadings = std_loadings.sum()
                    sum_error = (1 - std_loadings**2).sum()
                    cr = sum_loadings**2 / (sum_loadings**2 + sum_error)
                        
                    # Calculate AVE
                    sum_squared = (std_loadings**2).sum()
                    ave = sum_squared / (sum_squared + sum_error)
                        
                    status = 'Good' if cr >= 0.7 and ave >= 0.5 else 'Acceptable' if cr >= 0.6 else 'Poor'
                    rep += f"{latent:<20} {cr:>15.3f} {ave:>15.3f} {len(std_loadings):>10} {status:>20}\n"
                rep += "\n"
                
            # Model Syntax
            rep += f"{'='*90}\n"
            rep += "MODEL SYNTAX\n"
            rep += f"{'='*90}\n"
            rep += f"{res['syntax']}\n\n"
                
            # References
            rep += f"{'='*90}\n"
            rep += "INTERPRETATION GUIDELINES\n"
            rep += f"{'='*90}\n"
            rep += "Model Fit (Hu & Bentler, 1999):\n"
            rep += "  • CFI/TLI ≥ 0.95: Excellent fit\n"
            rep += "  • RMSEA ≤ 0.06: Good fit\n"
            rep += "  • SRMR ≤ 0.08: Good fit\n\n"
            rep += "Factor Loadings (Hair et al., 2019):\n"
            rep += "  • λ ≥ 0.70: Excellent\n"
            rep += "  • λ ≥ 0.50: Acceptable\n"
            rep += "  • λ < 0.50: Consider removal\n\n"
            rep += "Reliability:\n"
            rep += "  • CR ≥ 0.70: Acceptable\n"
            rep += "  • AVE ≥ 0.50: Acceptable convergent validity\n"
            
        self.current_report = rep
        report_area.setText(rep)
        # Tab set by get_or_create
        
        QMessageBox.information(self, "✓ Analysis Complete", 
                               f"{res['mode']} analysis completed successfully!\n\nN = {res['n']}")

    def create_regression_plots(self, residuals, fitted, plot_layout):
        for i in reversed(range(plot_layout.count())): 
            widget = plot_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Regression Diagnostic Plots', fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Fitted
        axes[0, 0].scatter(fitted, residuals, alpha=0.6, edgecolors='k', s=50)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Fitted Values', fontsize=12)
        axes[0, 0].set_ylabel('Residuals', fontsize=12)
        axes[0, 0].set_title('Residuals vs Fitted\n(Linearity & Homoscedasticity)', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Normal Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Normal Q-Q Plot\n(Normality Check)', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scale-Location
        standardized_residuals = residuals / np.std(residuals)
        axes[1, 0].scatter(fitted, np.sqrt(np.abs(standardized_residuals)), alpha=0.6, edgecolors='k', s=50)
        axes[1, 0].set_xlabel('Fitted Values', fontsize=12)
        axes[1, 0].set_ylabel('√|Standardized Residuals|', fontsize=12)
        axes[1, 0].set_title('Scale-Location Plot\n(Homoscedasticity Check)', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Histogram
        axes[1, 1].hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1, 1].set_xlabel('Residuals', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Histogram of Residuals\n(Normality Check)', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        canvas = FigureCanvas(fig)
        plot_layout.addWidget(canvas)

    def _connect_nav(self):
        for name, btn in self.nav_btns.items():
            btn.clicked.connect(self.navigate)

    def navigate(self):
        sender = self.sender()
        for name, btn in self.nav_btns.items():
            if btn == sender:
                idx_map = {
                    "DATA IMPORT": 0,
                    "DESCRIPTIVES": 1,
                    "CORRELATION": 2,
                    "T-TEST": 3,
                    "REGRESSION": 4,
                    "EXPLORATORY (EFA)": 5,
                    "CONFIRMATORY (CFA)": 6,
                    "SEM PATH": 7
                }
                if name in idx_map:
                    self.config_panel.setCurrentIndex(idx_map[name])
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())