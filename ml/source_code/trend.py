import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
import shap
import datetime
import warnings
warnings.filterwarnings('ignore')

class SalesTrendPredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Sales Forcasting Model")
        self.root.geometry("1200x750")
        self.root.configure(bg="#f0f0f0")       
        self.data = None

        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        }
        
        self.anomaly_models = {
            "None": None,
            "Isolation Forest": IsolationForest(contamination=0.05, random_state=42),
            "Local Outlier Factor": LocalOutlierFactor(contamination=0.05, novelty=True)
        }        
        self.setup_ui()
    
    def setup_ui(self):
        # Create notebook for tabbed interface
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Main tab
        main_tab = ttk.Frame(self.notebook)
        self.notebook.add(main_tab, text="Dashboard")
        
        # Anomaly Detection tab
        anomaly_tab = ttk.Frame(self.notebook)
        self.notebook.add(anomaly_tab, text="Anomaly Detection")
        
        # Explainable AI tab
        xai_tab = ttk.Frame(self.notebook)
        self.notebook.add(xai_tab, text="Explainable AI")
        
        # Setup main dashboard tab
        self.setup_main_tab(main_tab)
        
        # Setup anomaly detection tab
        self.setup_anomaly_tab(anomaly_tab)
        
        # Setup XAI tab
        self.setup_xai_tab(xai_tab)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready. Import data to begin.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_main_tab(self, parent):
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        control_outer_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_outer_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
    
        control_canvas = tk.Canvas(control_outer_frame, width=200)
        scrollbar = ttk.Scrollbar(control_outer_frame, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas)
    
        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        canvas_frame = control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        
        def configure_canvas(event):
            control_canvas.configure(scrollregion=control_canvas.bbox("all"))
            control_canvas.itemconfig(canvas_frame, width=control_canvas.winfo_width())
            
        control_frame.bind("<Configure>", configure_canvas)
        control_canvas.bind("<Configure>", lambda e: control_canvas.itemconfig(canvas_frame, width=control_canvas.winfo_width()))
        
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        ttk.Label(control_frame, text="Data Source:").pack(anchor=tk.W, padx=5, pady=5)
        self.btn_import = ttk.Button(control_frame, text="Import CSV", command=self.import_data)
        self.btn_import.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(control_frame, text="Target Feature:").pack(anchor=tk.W, padx=5, pady=5)
        self.target_var = tk.StringVar()
        self.target_select = ttk.Combobox(control_frame, textvariable=self.target_var, state="readonly")
        self.target_select.pack(fill=tk.X, padx=5, pady=5)
        self.target_select.bind("<<ComboboxSelected>>", self.update_feature_list)
        
        ttk.Label(control_frame, text="Date Column:").pack(anchor=tk.W, padx=5, pady=5)
        self.date_var = tk.StringVar()
        self.date_select = ttk.Combobox(control_frame, textvariable=self.date_var, state="readonly")
        self.date_select.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Predictor Features:").pack(anchor=tk.W, padx=5, pady=5)
        self.feature_frame = ttk.Frame(control_frame)
        self.feature_frame.pack(fill=tk.X, padx=5, pady=5)
        self.feature_vars = []
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(control_frame, text="Prediction Model:").pack(anchor=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar(value="Linear Regression")
        for model_name in self.models.keys():
            ttk.Radiobutton(control_frame, text=model_name, value=model_name, 
                        variable=self.model_var).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(control_frame, text="Anomaly Detection:").pack(anchor=tk.W, padx=5, pady=5)
        self.anomaly_var = tk.StringVar(value="None")
        for model_name in self.anomaly_models.keys():
            ttk.Radiobutton(control_frame, text=model_name, value=model_name, 
                        variable=self.anomaly_var).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Label(control_frame, text="Forecast Period (days):").pack(anchor=tk.W, padx=5, pady=5)
        self.forecast_days = tk.IntVar(value=30)
        forecast_spin = ttk.Spinbox(control_frame, from_=1, to=365, textvariable=self.forecast_days, width=10)
        forecast_spin.pack(anchor=tk.W, padx=5, pady=5)
        
        self.use_automl = tk.BooleanVar(value=False)
        ttk.Checkbutton(control_frame, text="Use AutoML to tune parameters", 
                    variable=self.use_automl).pack(anchor=tk.W, padx=5, pady=5)
        
        self.btn_predict = ttk.Button(control_frame, text="Run Prediction", command=self.run_prediction)
        self.btn_predict.pack(fill=tk.X, padx=5, pady=20)
        self.btn_predict.config(state=tk.DISABLED)
        
        viz_frame = ttk.Frame(main_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.summary_frame = ttk.LabelFrame(viz_frame, text="Data Summary")
        self.summary_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.summary_text = tk.Text(self.summary_frame, height=5, width=50)
        self.summary_text.pack(fill=tk.X, padx=5, pady=5)
        self.summary_text.config(state=tk.DISABLED)
        
        self.fig_frame = ttk.LabelFrame(viz_frame, text="Trend Visualization")
        self.fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.plot_canvas = FigureCanvasTkAgg(self.fig, self.fig_frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        def _on_tab_close():
            control_canvas.unbind_all("<MouseWheel>")
            
        parent.bind("<Destroy>", lambda e: _on_tab_close())
    
    def setup_anomaly_tab(self, parent):
        anomaly_frame = ttk.Frame(parent)
        anomaly_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(anomaly_frame, text="Anomaly Detection Settings")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Anomaly Threshold:").pack(anchor=tk.W, padx=5, pady=5)
        self.threshold_var = tk.DoubleVar(value=0.05)
        threshold_spin = ttk.Spinbox(control_frame, from_=0.01, to=0.5, increment=0.01, 
                                     textvariable=self.threshold_var, width=10)
        threshold_spin.pack(anchor=tk.W, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Detection Methods:").pack(anchor=tk.W, padx=5, pady=5)
        self.iso_forest_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Isolation Forest", 
                       variable=self.iso_forest_var).pack(anchor=tk.W, padx=20, pady=2)
        
        self.lof_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Local Outlier Factor", 
                       variable=self.lof_var).pack(anchor=tk.W, padx=20, pady=2)
        
        self.zscore_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Z-Score Method", 
                       variable=self.zscore_var).pack(anchor=tk.W, padx=20, pady=2)
        
        self.btn_detect_anomalies = ttk.Button(control_frame, text="Detect Anomalies", 
                                              command=self.detect_anomalies)
        self.btn_detect_anomalies.pack(fill=tk.X, padx=5, pady=20)
        self.btn_detect_anomalies.config(state=tk.DISABLED)
        
        viz_frame = ttk.Frame(anomaly_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.anomaly_fig_frame = ttk.LabelFrame(viz_frame, text="Anomaly Visualization")
        self.anomaly_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.anomaly_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.anomaly_canvas = FigureCanvasTkAgg(self.anomaly_fig, self.anomaly_fig_frame)
        self.anomaly_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_xai_tab(self, parent):
        xai_frame = ttk.Frame(parent)
        xai_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        control_frame = ttk.LabelFrame(xai_frame, text="Explainable AI Settings")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(control_frame, text="Explainability Method:").pack(anchor=tk.W, padx=5, pady=5)
        self.xai_method_var = tk.StringVar(value="SHAP")
        ttk.Radiobutton(control_frame, text="SHAP Values", value="SHAP", 
                       variable=self.xai_method_var).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(control_frame, text="Feature Importance", value="Importance", 
                       variable=self.xai_method_var).pack(anchor=tk.W, padx=20, pady=2)
        
        ttk.Label(control_frame, text="Sample to Explain:").pack(anchor=tk.W, padx=5, pady=5)
        self.explain_var = tk.StringVar(value="Recent")
        ttk.Radiobutton(control_frame, text="Most Recent Period", value="Recent", 
                       variable=self.explain_var).pack(anchor=tk.W, padx=20, pady=2)
        ttk.Radiobutton(control_frame, text="Average Prediction", value="Average", 
                       variable=self.explain_var).pack(anchor=tk.W, padx=20, pady=2)
        
        self.btn_explain = ttk.Button(control_frame, text="Generate Explanation", 
                                     command=self.generate_explanation)
        self.btn_explain.pack(fill=tk.X, padx=5, pady=20)
        self.btn_explain.config(state=tk.DISABLED)
        
        viz_frame = ttk.Frame(xai_frame)
        viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.xai_fig_frame = ttk.LabelFrame(viz_frame, text="Feature Impact Visualization")
        self.xai_fig_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.xai_fig = plt.Figure(figsize=(6, 4), dpi=100)
        self.xai_canvas = FigureCanvasTkAgg(self.xai_fig, self.xai_fig_frame)
        self.xai_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def import_data(self):
        file_path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.data = pd.read_csv(file_path)
                
                self.date_select['values'] = self.data.columns.tolist()
                for col in self.data.columns:
                    if 'date' in col.lower() or 'time' in col.lower():
                        self.date_var.set(col)
                        break
                
                self.target_select['values'] = [c for c in self.data.columns if self.data[c].dtype in ['int64', 'float64']]
                if len(self.target_select['values']) > 0:
                    self.target_select.current(0)
                    self.update_feature_list()
                    self.btn_predict.config(state=tk.NORMAL)
                    self.btn_detect_anomalies.config(state=tk.NORMAL)
                    self.btn_explain.config(state=tk.NORMAL)
                
                if self.date_var.get():
                    try:
                        self.data[self.date_var.get()] = pd.to_datetime(self.data[self.date_var.get()])
                        self.data.sort_values(by=self.date_var.get(), inplace=True)
                    except:
                        messagebox.showwarning("Warning", "Could not convert the selected column to datetime format.")
                
                self.update_summary()
                
                self.status_var.set(f"Data imported: {len(self.data)} rows, {len(self.data.columns)} columns")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import data: {str(e)}")
                self.status_var.set("Import failed.")
    
    def update_feature_list(self, event=None):
        for widget in self.feature_frame.winfo_children():
            widget.destroy()
        self.feature_vars = []
        
        current_target = self.target_var.get()
        for col in self.data.columns:
            if col != current_target and col != self.date_var.get() and self.data[col].dtype in ['int64', 'float64']:
                var = tk.BooleanVar(value=True)
                self.feature_vars.append((col, var))
                ttk.Checkbutton(self.feature_frame, text=col, variable=var).pack(anchor=tk.W)
    
    def update_summary(self):
        if self.data is not None:
            summary_text = f"Dataset Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns\n\n"
            
            date_col = self.date_var.get()
            if date_col and pd.api.types.is_datetime64_any_dtype(self.data[date_col]):
                start_date = self.data[date_col].min().strftime('%Y-%m-%d')
                end_date = self.data[date_col].max().strftime('%Y-%m-%d')
                summary_text += f"Date Range: {start_date} to {end_date}\n\n"
            
            num_cols = self.data.select_dtypes(include=['int64', 'float64']).columns[:3]
            if len(num_cols) > 0:
                summary_text += "Key Metrics:\n"
                for col in num_cols:
                    summary_text += f"{col}: avg={self.data[col].mean():.2f}, min={self.data[col].min():.2f}, max={self.data[col].max():.2f}\n"
            
            self.summary_text.config(state=tk.NORMAL)
            self.summary_text.delete(1.0, tk.END)
            self.summary_text.insert(tk.END, summary_text)
            self.summary_text.config(state=tk.DISABLED)
    
    def prepare_data_for_models(self):
        if self.data is None:
            messagebox.showwarning("Warning", "Please import data first.")
            return None
        
        target = self.target_var.get()
        selected_features = [col for col, var in self.feature_vars if var.get()]
        
        if not selected_features:
            messagebox.showwarning("Warning", "Please select at least one predictor feature.")
            return None
        
        data_copy = self.data.copy()
        
        date_col = self.date_var.get()
        if date_col and date_col in data_copy.columns:
            data_copy[date_col] = pd.to_datetime(data_copy[date_col])
            data_copy.sort_values(by=date_col, inplace=True)
            
            if 'day_of_week' not in data_copy.columns:
                data_copy['day_of_week'] = data_copy[date_col].dt.dayofweek
                selected_features.append('day_of_week')
            
            if 'month' not in data_copy.columns:
                data_copy['month'] = data_copy[date_col].dt.month
                selected_features.append('month')
            
            dates = data_copy[date_col]
            future_dates = pd.date_range(
                start=dates.iloc[-1] + pd.Timedelta(days=1),
                periods=self.forecast_days.get(),
                freq='D'
            )
        else:
            dates = pd.RangeIndex(len(data_copy))
            future_dates = pd.RangeIndex(len(data_copy), len(data_copy) + self.forecast_days.get())
        
        #lag and rolling features
        data_copy['lag_1'] = data_copy[target].shift(1)
        data_copy['lag_7'] = data_copy[target].shift(7)
        data_copy['lag_30'] = data_copy[target].shift(30)
        
        data_copy['rolling_mean_7'] = data_copy[target].rolling(7).mean()
        data_copy['rolling_std_7'] = data_copy[target].rolling(7).std()
        data_copy['rolling_mean_30'] = data_copy[target].rolling(30).mean()
        data_copy['rolling_std_30'] = data_copy[target].rolling(30).std()
        
        data_copy.dropna(inplace=True)
        
        time_series_features = ['lag_1', 'lag_7', 'lag_30', 'rolling_mean_7', 'rolling_std_7', 'rolling_mean_30', 'rolling_std_30']
        selected_features.extend(time_series_features)
        
        if isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Series):
            dates = dates[data_copy.index]
        else:
            dates = pd.RangeIndex(len(data_copy))
        
        X = data_copy[selected_features]
        y = data_copy[target]
        
        return {
            'X': X,
            'y': y,
            'dates': dates,
            'future_dates': future_dates,
            'target': target,
            'features': selected_features,
            'data_copy': data_copy
        }
    
    def run_prediction(self):
        data_dict = self.prepare_data_for_models()
        if data_dict is None:
            return
        
        X, y = data_dict['X'], data_dict['y']
        dates = data_dict['dates']
        future_dates = data_dict['future_dates']
        target = data_dict['target']
        selected_features = data_dict['features']
        data_copy = data_dict['data_copy']
        
        try:
            model_name = self.model_var.get()
            use_automl = self.use_automl.get()
            
            future_pred = None
            model_info = {}
            
            #Time-aware train-test split
            split = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split], X.iloc[split:]
            y_train, y_test = y.iloc[:split], y.iloc[split:]
            
            self.status_var.set(f"Training {model_name} model with time-series features...")
            self.root.update_idletasks()
            
            if use_automl and model_name == "Random Forest":
                model = self.models[model_name]

                param_space = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
                
                tscv = TimeSeriesSplit(n_splits=3)
                random_search = RandomizedSearchCV(
                    model, param_distributions=param_space,
                    n_iter=10, cv=tscv, verbose=0, random_state=42, n_jobs=-1
                )
                
                random_search.fit(X_train, y_train)
                model = random_search.best_estimator_
                model_info['name'] = f"{model_name} with AutoML"
                model_info['best_params'] = random_search.best_params_
            
            else:
                model = self.models[model_name]
                model.fit(X_train, y_train)
                model_info['name'] = model_name
            
            #Evaluate on test set
            test_predictions = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, test_predictions)
            mse = mean_squared_error(y_test, test_predictions)
            rmse = np.sqrt(mse)

            try:
                mape = mean_absolute_percentage_error(y_test, test_predictions)
                mape_str = f", MAPE: {mape:.4f}"
            except:
                mape_str = ""
            
            model_info['mae'] = mae
            model_info['rmse'] = rmse
            
            print(f"\n=== Model Evaluation on Test Set ===")
            print(f"Model: {model_info['name']}")
            print(f"MAE: {mae:.4f}")
            print(f"RMSE: {rmse:.4f}{mape_str}")
            print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
            print(f"Features used: {len(selected_features)}")
            print(f"Time-series features: lag_1, lag_7, lag_30, rolling_mean_7, rolling_std_7, rolling_mean_30, rolling_std_30")
            
            model.fit(X, y)
            
            #Recursive forecasting with lag/rolling updates
            future_predictions = []
            recent_values = list(y.iloc[-30:].values)
            
            for i in range(self.forecast_days.get()):
                next_features = []
                
                for feat in selected_features:
                    if feat in ['day_of_week', 'month']:
                        if isinstance(future_dates, pd.DatetimeIndex):
                            if feat == 'day_of_week':
                                next_features.append(future_dates[i].dayofweek)
                            elif feat == 'month':
                                next_features.append(future_dates[i].month)
                        else:
                            next_features.append(X[feat].iloc[-1])
                    elif feat == 'lag_1':
                        next_features.append(recent_values[-1])
                    elif feat == 'lag_7':
                        next_features.append(recent_values[-7] if len(recent_values) >= 7 else recent_values[0])
                    elif feat == 'lag_30':
                        next_features.append(recent_values[-30] if len(recent_values) >= 30 else recent_values[0])
                    elif feat == 'rolling_mean_7':
                        next_features.append(np.mean(recent_values[-7:]))
                    elif feat == 'rolling_std_7':
                        next_features.append(np.std(recent_values[-7:]))
                    elif feat == 'rolling_mean_30':
                        next_features.append(np.mean(recent_values[-30:]))
                    elif feat == 'rolling_std_30':
                        next_features.append(np.std(recent_values[-30:]))
                    else:
                        next_features.append(X[feat].iloc[-1])
                
                next_pred = model.predict([next_features])[0]
                future_predictions.append(next_pred)
                recent_values.append(next_pred)
                if len(recent_values) > 30:
                    recent_values.pop(0)
            
            future_pred = np.array(future_predictions)
            
            self.forecast_result = future_pred
            self.model_info = model_info
            self.forecast_dates = future_dates
            
            self.plot_forecast(dates, y, future_dates, future_pred, target)
            
            self.status_var.set(f"Prediction complete using {model_info['name']} | Test MAE: {mae:.2f}, RMSE: {rmse:.2f}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during prediction: {str(e)}")
            self.status_var.set("Prediction failed.")
    
    def plot_forecast(self, dates, historical_values, future_dates, forecast_values, target_name):
        self.fig.clear()
        
        ax = self.fig.add_subplot(111)
        
        if isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Series) and pd.api.types.is_datetime64_any_dtype(dates):
            if len(historical_values) > 30:
                start_idx = len(historical_values) - 30
                display_hist_dates = dates[start_idx:]
                display_hist_values = historical_values[start_idx:]
            else:
                display_hist_dates = dates
                display_hist_values = historical_values
                
            ax.plot(display_hist_dates, display_hist_values, 'b-', label='Historical Data')
            ax.plot(future_dates, forecast_values, 'r--', label='Forecast')
            
            from matplotlib.dates import DateFormatter
            date_format = DateFormatter('%m-%d')
            ax.xaxis.set_major_formatter(date_format)
            
            plt.xticks(rotation=45)
            ax.set_xlabel('Date (MM-DD)')
        else:
            #If dates are just indices
            if len(historical_values) > 30:
                start_idx = len(historical_values) - 30
                display_range = range(start_idx, len(historical_values))
                display_values = historical_values[start_idx:]
            else:
                display_range = range(len(historical_values))
                display_values = historical_values
                
            ax.plot(display_range, display_values, 'b-', label='Historical Data')
            ax.plot(range(len(historical_values), len(historical_values) + len(forecast_values)), 
                forecast_values, 'r--', label='Forecast')
            ax.set_xlabel('Time Period (Days)')
        
        ax.set_ylabel(target_name)
        ax.set_title(f'{target_name} Forecast (Last Month + {len(forecast_values)} Days)')
        ax.legend()
        
        #Add confidence interval
        if isinstance(forecast_values, np.ndarray) and len(forecast_values) > 0:
            std_dev = historical_values.std() * 0.5  #Simplified assumption
            if isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Series) and pd.api.types.is_datetime64_any_dtype(dates):
                ax.fill_between(
                    future_dates,
                    forecast_values - std_dev,
                    forecast_values + std_dev,
                    color='r', alpha=0.2,
                    label='Confidence Interval'
                )
            else:
                ax.fill_between(
                    range(len(historical_values), len(historical_values) + len(forecast_values)),
                    forecast_values - std_dev,
                    forecast_values + std_dev,
                    color='r', alpha=0.2,
                    label='Confidence Interval'
                )
        
        #Adjust layout to make room for dates
        self.fig.tight_layout()
        
        #Redraw canvas
        self.plot_canvas.draw()
    
    def detect_anomalies(self):
        data_dict = self.prepare_data_for_models()
        if data_dict is None:
            return
        
        X, y = data_dict['X'], data_dict['y']
        dates = data_dict['dates']
        target = data_dict['target']
        
        try:
            self.status_var.set("Detecting anomalies...")
            self.root.update_idletasks()
            
            # Clear previous plot
            self.anomaly_fig.clear()
            ax = self.anomaly_fig.add_subplot(111)
            
            # Focus on recent data (last month/30 points)
            if len(y) > 30:
                start_idx = len(y) - 30
                display_slice = slice(start_idx, None)
            else:
                display_slice = slice(None)
            
            # Plot original data
            if isinstance(dates, pd.DatetimeIndex) or isinstance(dates, pd.Series) and pd.api.types.is_datetime64_any_dtype(dates):
                display_dates = dates[display_slice]
                display_y = y[display_slice]
                ax.plot(display_dates, display_y, 'b-', label='Original Data')
                x_values = display_dates
                
                # Format x-axis to show days
                from matplotlib.dates import DateFormatter
                date_format = DateFormatter('%m-%d')
                ax.xaxis.set_major_formatter(date_format)
                plt.xticks(rotation=45)
                ax.set_xlabel('Date (MM-DD)')
            else:
                display_range = range(len(y))[display_slice]
                display_y = y[display_slice]
                ax.plot(display_range, display_y, 'b-', label='Original Data')
                x_values = display_range
                ax.set_xlabel('Time Period (Days)')
            
            # Initialize anomaly detection results for all data
            anomalies = np.zeros(len(y), dtype=bool)
            
            # Apply selected methods
            methods_used = []
            
            if self.iso_forest_var.get():
                # Isolation Forest
                iso_forest = IsolationForest(contamination=self.threshold_var.get(), random_state=42)
                iso_anomalies = iso_forest.fit_predict(X) == -1
                anomalies = anomalies | iso_anomalies
                methods_used.append("Isolation Forest")
            
            if self.lof_var.get():
                # Local Outlier Factor
                lof = LocalOutlierFactor(contamination=self.threshold_var.get(), novelty=False)
                lof_anomalies = lof.fit_predict(X) == -1
                anomalies = anomalies | lof_anomalies
                methods_used.append("LOF")
            
            if self.zscore_var.get():
                # Z-score method
                z_scores = np.abs((y - y.mean()) / y.std())
                z_threshold = 3.0  # Standard 3-sigma rule
                zscore_anomalies = z_scores > z_threshold
                anomalies = anomalies | zscore_anomalies
                methods_used.append("Z-score")
            
            # Get the anomalies for the display range
            display_anomalies = anomalies[display_slice]
            
            # Plot anomalies
            if isinstance(x_values, pd.DatetimeIndex) or isinstance(x_values, pd.Series) and pd.api.types.is_datetime64_any_dtype(x_values):
                ax.scatter(x_values[display_anomalies], display_y[display_anomalies], color='red', label='Anomalies', s=50, zorder=5)
            else:
                ax.scatter(np.array(x_values)[display_anomalies], display_y[display_anomalies], color='red', label='Anomalies', s=50, zorder=5)
            
            ax.set_ylabel(target)
            ax.set_title(f'Anomaly Detection for {target}')
            ax.legend()
            
            # Adjust layout
            self.anomaly_fig.tight_layout()
            
            # Redraw canvas
            self.anomaly_canvas.draw()
            
            # Update status
            anomaly_count = np.sum(anomalies)
            display_anomaly_count = np.sum(display_anomalies)
            self.status_var.set(f"Found {display_anomaly_count} anomalies in view, {anomaly_count} total using {', '.join(methods_used)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during anomaly detection: {str(e)}")
            self.status_var.set("Anomaly detection failed.")
    
    def generate_explanation(self):
        data_dict = self.prepare_data_for_models()
        if data_dict is None:
            return
        
        X, y = data_dict['X'], data_dict['y']
        selected_features = data_dict['features']
        
        try:
            self.status_var.set("Generating explanations...")
            self.root.update_idletasks()
            
            # Train a random forest model for explanation
            # (Using Random Forest even if another model was selected for prediction)
            explainer_model = RandomForestRegressor(n_estimators=100, random_state=42)
            explainer_model.fit(X, y)
            
            # Clear previous plot
            self.xai_fig.clear()
            
            # Choose method
            method = self.xai_method_var.get()
            
            if method == "SHAP":
                # SHAP values explanation
                try:
                    # Create a small subset for SHAP explanation to speed up calculation
                    X_sample = X.iloc[-20:] if len(X) > 20 else X
                    
                    # Create explainer
                    explainer = shap.TreeExplainer(explainer_model)
                    shap_values = explainer.shap_values(X_sample)
                    
                    # Plot
                    ax = self.xai_fig.add_subplot(111)
                    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                    plt.tight_layout()
                    self.xai_canvas.draw()
                    
                    self.status_var.set("SHAP explanation generated successfully")
                except Exception as shap_error:
                    # Fallback to feature importance if SHAP fails
                    self.status_var.set(f"SHAP analysis failed, falling back to feature importance. Error: {str(shap_error)}")
                    method = "Importance"
            
            if method == "Importance":
                # Feature importance
                ax = self.xai_fig.add_subplot(111)
                
                # Get importance
                importances = explainer_model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                # Plot feature importances
                ax.barh(range(len(indices)), importances[indices], align='center')
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([selected_features[i] for i in indices])
                ax.set_xlabel('Feature Importance')
                ax.set_title('Feature Importance for Prediction')
                
                self.xai_fig.tight_layout()
                self.xai_canvas.draw()
                
                self.status_var.set("Feature importance explanation generated successfully")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during explanation: {str(e)}")
            self.status_var.set("Explanation generation failed.")
    
if __name__ == "__main__":
    root = tk.Tk()
    app = SalesTrendPredictor(root)
    root.mainloop()