import streamlit as st
import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects, PooledOLS
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from streamlit_option_menu import option_menu

# Fungsi untuk mengupload file
def upload_file():
    uploaded_file = st.file_uploader("Upload file Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error: {e}")
            return None
    else:
        return None

# Fungsi untuk menampilkan eksplorasi data
def explore_data(df):
    st.header("Data Exploration")

    # Pilih jenis grafik
    chart_types = st.multiselect("Pilih chart", ["Scatterplot", "Histogram", "Line Chart", "Boxplot"])
    
    # Pilih kolom untuk visualisasi
    columns = df.columns.tolist()
    selected_columns = st.multiselect("Pilih kolom", columns, default=columns[:2])
    
    if "Scatterplot" in chart_types:
        st.subheader("Scatterplot")
        if len(selected_columns) >= 2:
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=selected_columns[0], y=selected_columns[1], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Pilih minimal dua variabel untuk scatterplot.")
    
    if "Histogram" in chart_types:
        st.subheader("Histogram")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)
    
    if "Line Chart" in chart_types:
        st.subheader("Line Chart")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=df.index, y=col, ax=ax)
            st.pyplot(fig)
    
    if "Boxplot" in chart_types:
        st.subheader("Boxplot")
        for col in selected_columns:
            fig, ax = plt.subplots()
            sns.boxplot(data=df[col], ax=ax)
            st.pyplot(fig)

# Fungsi untuk memeriksa multikolinearitas
def check_multicollinearity(df, predictors):
    st.subheader("Asumsi Multikolinearitas")
    X = df[predictors]
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.write(vif_data)

# Fungsi untuk uji Hausman
def hausman_test(fixed, random):
    st.subheader("Uji Hausman")
    b = fixed.params
    B = random.params
    v_b = fixed.cov
    v_B = random.cov
    df = b.shape[0]
    chi2 = np.dot((b - B).T, np.linalg.inv(v_b - v_B).dot(b - B))
    p_value = stats.chi2.sf(chi2, df)
    st.write(f"Chi-Square: {chi2}, p-value: {p_value}")

# Fungsi untuk uji Lagrange Multiplier (LM)
def lm_test(pooled_model, random_model):
    st.subheader("Uji Lagrange Multiplier (LM)")
    test_stat = (pooled_model.rsquared - random_model.rsquared) / (1 - random_model.rsquared) * (random_model.df_resid / random_model.df_model)
    p_value = 1 - stats.chi2.cdf(test_stat, df=1)
    st.write(f"Test Statistic: {test_stat}, p-value: {p_value}")

# Fungsi untuk model regresi data panel
def panel_regression(df, predictors, response, entity_col, time_col, model_type):
    st.subheader("Data Panel Regression")
    df.set_index([entity_col, time_col], inplace=True)
    y = df[response]
    X = df[predictors]
    X = sm.add_constant(X)
    
    if model_type == "FEM":
        model = PanelOLS(y, X, entity_effects=True)
        results = model.fit()
        st.write(results.summary)
    elif model_type == "REM":
        model = RandomEffects(y, X)
        results = model.fit()
        st.write(results.summary)
    elif model_type == "CEM":
        model = PanelOLS(y, X)
        results = model.fit()
        st.write(results.summary)
    else:
        st.write("Model not found.")

    return results

# Fungsi untuk memeriksa kolom dengan tepat dua grup yang berbeda
def get_columns_with_two_groups(df):
    return [col for col in df.columns if df[col].nunique() == 2]

# Fungsi untuk halaman eksplorasi data
def page_eksplorasi(df):
    st.title("Data Exploration")
    if df is not None:
        st.write("Dataframe:")
        st.write(df)
        
        explore_data(df)
        
        predictors = st.multiselect("Pilih variabel prediktor", df.columns.tolist())
        if len(predictors) > 0:
            check_multicollinearity(df, predictors)

# Fungsi untuk halaman analisis data panel
def page_analisis_panel(df):
    st.title("Data Panel Regression Analysis")
    if df is not None:
        st.write("Dataframe:")
        st.write(df)
        
        predictors = st.multiselect("Pilih variabel prediktor", df.columns.tolist())
        response = st.selectbox("Pilih variabel respon", df.columns.tolist())
        entity_col = st.selectbox("Pilih kolom entitas (example: Province)", df.columns.tolist())
        time_col = st.selectbox("Pilih kolom waktu (example: Year)", df.columns.tolist())
        
        if len(predictors) > 0 and response:
            model_type = st.selectbox("Pilih model regresi", ["FEM", "REM", "CEM"])
            results = panel_regression(df, predictors, response, entity_col, time_col, model_type)
            
            st.subheader("Pemilihan Model Regresi")
            additional_tests = st.multiselect("Pilih uji", ["Uji Hausman", "Uji LM"])
            
            if "Uji Hausman" in additional_tests:
                fixed_model = PanelOLS(df[response], df[predictors], entity_effects=True).fit()
                random_model = RandomEffects(df[response], df[predictors]).fit()
                hausman_test(fixed_model, random_model)
                
            if "Uji LM" in additional_tests:
                pooled_model = PooledOLS(df[response], sm.add_constant(df[predictors])).fit()
                random_model = RandomEffects(df[response], df[predictors]).fit()
                lm_test(pooled_model, random_model)

# Fungsi untuk halaman pemeriksaan asumsi IIDN
def page_pemeriksaan_asumsi(df):
    st.title("IIDN Assumption")
    if df is not None:
        st.write("Dataframe:")
        st.write(df)
        
        # Memilih variabel untuk analisis
        predictors = st.multiselect("Pilih variabel prediktor", df.columns.tolist())
        response = st.selectbox("Pilih variabel respon", df.columns.tolist())
        
        if len(predictors) > 0 and response:
            X = df[predictors]
            X = sm.add_constant(X)
            y = df[response]
            
            # Fit a regression model
            model = sm.OLS(y, X).fit()
            residuals = model.resid
            
            # Asumsi Heteroskedastisitas (Identik)
            st.subheader("Asumsi Heteroskedastisitas")
            st.write("Hasil pengujian asumsi identik.")
            het_bp = het_breuschpagan(residuals, X)
            st.write(f"LM Statistic: {het_bp[0]}")
            st.write(f"LM-Test P-value: {het_bp[1]}")
            st.write(f"F-Statistic: {het_bp[2]}")
            st.write(f"F-Test P-value: {het_bp[3]}")
            
            # Asumsi Autokorelasi (Independen)
            st.subheader("Asumsi Autokorelasi")
            st.write("Hasil pengujian asumsi independen.")
            acorr_lb = acorr_ljungbox(residuals, lags=[10], boxpierce=True)
            st.write(f"Ljung-Box Statistic: {acorr_lb['lb_stat'].iloc[0]}")
            st.write(f"Ljung-Box P-value: {acorr_lb['lb_pvalue'].iloc[0]}")
            st.write(f"Box-Pierce Statistic: {acorr_lb['bp_stat'].iloc[0]}")
            st.write(f"Box-Pierce P-value: {acorr_lb['bp_pvalue'].iloc[0]}")
            
            # Asumsi Distribusi Normal
            st.subheader("Asumsi Distribusi Normal")
            st.write("Hasil pengujian asumsi distribusi normal.")
            _, pval = stats.normaltest(residuals)
            st.write(f"Normality Test P-value: {pval}")

# Fungsi utama untuk navigasi sidebar
def main():
    df = upload_file()
    with st.sidebar:
        selected = option_menu(
            menu_title='Regression',
            options=[
                'Data Exploration',
                'Data Panel Regression',
                'IIDN Assumption'
            ],
            icons=['image', 'house', 'chat'],
            menu_icon='clock',  # Icon for the main menu
            default_index=0
        )

    if selected == 'Data Exploration':
        page_eksplorasi(df)
    elif selected == 'Data Panel Regression':
        page_analisis_panel(df)
    elif selected == 'IIDN Assumption':
        page_pemeriksaan_asumsi(df)

if __name__ == "__main__":
    main()
