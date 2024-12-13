import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import google.generativeai as genai
import io
import os
from typing import Dict

# Enhanced Error Handling for API Configuration
def configure_gemini_api():
    """
    Safely configure Gemini API with comprehensive error handling
    """
    GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
    
    if not GEMINI_API_KEY:
        st.error("🚨 Gemini API key is missing. Please configure it in Streamlit secrets.")
        st.warning("""
        To set up your API key:
        1. Go to Google AI Studio (https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Add it to your Streamlit app's secrets:
           - Open .streamlit/secrets.toml
           - Add: GEMINI_API_KEY = "your_api_key_here"
        """)
        st.stop()
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        return GEMINI_API_KEY
    except Exception as e:
        st.error(f"🔒 API Configuration Failed: {e}")
        st.stop()

class BudgITAnalyzer:
    def __init__(self):
        """
        Initialize the BudgIT Analyzer with robust error handling
        """
        self.supported_file_types = ['.csv', '.xlsx', '.xls']
        
        try:
            # Use latest available model with fallback
            self.generative_model = genai.GenerativeModel('gemini-1.5-flash')
        except Exception as e:
            st.error(f"Model Initialization Error: {e}")
            self.generative_model = None

        # Predefined templates with more comprehensive data
        self.templates = {
            "Personal Budget": self.create_personal_budget_template(),
            "Business Expenses": self.create_business_expenses_template(),
            "Investment Portfolio": self.create_investment_portfolio_template()
        }

    def create_personal_budget_template(self):
        """Enhanced personal budget template with more columns"""
        data = {
            'Month': ['January', 'February', 'March', 'April', 'May', 'June'],
            'Category': ['Housing', 'Transportation', 'Food', 'Utilities', 'Entertainment', 'Savings'],
            'Planned Budget': [1500, 300, 500, 200, 150, 500],
            'Actual Expense': [1450, 280, 520, 210, 170, 450],
            'Variance': [-50, -20, 20, 10, 20, -50]
        }
        return pd.DataFrame(data)

    def create_business_expenses_template(self):
        """Enhanced business expenses template"""
        data = {
            'Quarter': ['Q1', 'Q1', 'Q1', 'Q1', 'Q1'],
            'Department': ['Sales', 'Marketing', 'IT', 'HR', 'Operations'],
            'Monthly Budget': [5000, 3000, 4000, 2500, 3500],
            'Actual Spending': [4800, 3200, 3900, 2400, 3600],
            'Budget Utilization %': [96, 106, 97.5, 96, 102.8]
        }
        return pd.DataFrame(data)

    def create_investment_portfolio_template(self):
        """Enhanced investment portfolio template"""
        data = {
            'Asset Class': ['Stocks', 'Bonds', 'Real Estate', 'Crypto', 'Commodities'],
            'Allocation Percentage': [50, 30, 10, 5, 5],
            'Current Value': [50000, 30000, 10000, 5000, 5000],
            'Year-to-Date Return %': [12.5, 4.2, 7.8, -15.3, 6.1]
        }
        return pd.DataFrame(data)

    def read_file(self, uploaded_file):
        """
        Read different file types with enhanced error handling
        """
        try:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_extension == '.csv':
                df = pd.read_csv(uploaded_file, encoding='utf-8')
            elif file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("❌ Unsupported file type. Only CSV and Excel files are supported.")
                return None

            # Basic data validation
            if df.empty:
                st.warning("⚠️ The uploaded file is empty.")
                return None

            return df

        except UnicodeDecodeError:
            st.error("❗ File encoding error. Try saving the file with UTF-8 encoding.")
        except Exception as e:
            st.error(f"📛 File Reading Error: {e}")
        
        return None

    def generate_ai_insights(self, data):
        """
        Generate AI-powered insights with robust error handling
        """
        if not self.generative_model:
            return "❌ AI model not initialized."

        try:
            prompt = f"""Analyze the following financial data professionally and provide:
            1. Top 3 key financial observations
            2. Potential trends and patterns
            3. Strategic recommendations

            Context: Detailed financial data analysis
            Data Preview: {data.head().to_string()}
            Total Rows: {len(data)}
            Columns: {', '.join(data.columns)}
            """

            response = self.generative_model.generate_content(prompt)
            return response.text
        
        except Exception as e:
            st.error(f"🤖 AI Insight Generation Error: {e}")
            return "Unable to generate AI insights at this moment."

    def create_visualizations(self, df):
        """
        Create interactive visualizations with fallback mechanisms
        """
        visualizations = {}

        try:
            # Detect numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            # Comparison Chart
            if len(numeric_cols) >= 2:
                # Use first two numeric columns for comparison
                col1, col2 = numeric_cols[:2]
                
                # Bar Chart
                fig_compare = go.Figure(data=[
                    go.Bar(name=col1, x=df.index, y=df[col1]),
                    go.Bar(name=col2, x=df.index, y=df[col2])
                ])
                fig_compare.update_layout(
                    title=f'{col1} vs {col2} Comparison',
                    xaxis_title='Index',
                    yaxis_title='Value',
                    barmode='group'
                )
                visualizations['comparison_chart'] = fig_compare

            # Pie Chart (if categorical column exists)
            categorical_cols = df.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                pie_col = categorical_cols[0]
                value_col = numeric_cols[0]
                
                pie_data = df.groupby(pie_col)[value_col].sum()
                fig_pie = px.pie(
                    values=pie_data.values, 
                    names=pie_data.index, 
                    title=f'Distribution of {value_col} by {pie_col}'
                )
                visualizations['pie_chart'] = fig_pie

        except Exception as e:
            st.error(f"📊 Visualization Error: {e}")

        return visualizations

def main():
    # Configure page
    st.set_page_config(page_title="BudgIT", page_icon="💰", layout="wide")

    # Configure Gemini API
    configure_gemini_api()

    # Try to load logo with fallback
    try:
        st.sidebar.image("media/logo.png", use_container_width=True)
    except Exception:
        st.sidebar.markdown("## 💰 BudgIT")

    # Main title
    st.title("🚀 BudgIT - Financial Document Analyzer")
    st.subheader("Unlock Insights from Your Financial Documents")

    # Initialize analyzer
    analyzer = BudgITAnalyzer()

    # Sidebar template selection
    st.sidebar.header("📋 Template Selection")
    template_selection = st.sidebar.selectbox(
        "Choose a Template", 
        list(analyzer.templates.keys()) + ["Upload Custom File"]
    )

    # Template handling
    if template_selection != "Upload Custom File":
        template_df = analyzer.templates[template_selection]
        csv_buffer = io.StringIO()
        template_df.to_csv(csv_buffer, index=False)

        st.sidebar.download_button(
            label=f"Download {template_selection} Template",
            data=csv_buffer.getvalue(),
            file_name=f"{template_selection.lower().replace(' ', '_')}_template.csv",
            mime="text/csv",
            help="Download a pre-formatted template to simplify your data entry"
        )

        st.sidebar.markdown("### 📝 Template Usage Guide")
        st.sidebar.markdown("""
        1. 📥 Download template
        2. 📊 Fill with your data
        3. 📤 Upload completed file
        4. 🔍 Get powerful insights
        """)

    # File uploader
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file", 
        type=['csv', 'xlsx', 'xls']
    )

    # Analysis section
    if uploaded_file is not None:
        data = analyzer.read_file(uploaded_file)

        if data is not None:
            # Create tabs
            tab1, tab2, tab3 = st.tabs(
                ["📊 Data Overview", "📈 Visualizations", "🤖 AI Insights"]
            )

            with tab1:
                st.subheader("Data Overview")
                st.dataframe(data)
                st.write(f"📏 Rows: {data.shape[0]}, Columns: {data.shape[1]}")

            with tab2:
                st.subheader("Interactive Visualizations")
                visualizations = analyzer.create_visualizations(data)

                if visualizations:
                    for name, fig in visualizations.items():
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No visualizations could be generated.")

            with tab3:
                st.subheader("AI-Powered Insights")
                ai_insights = analyzer.generate_ai_insights(data)
                st.markdown(ai_insights)

if __name__ == "__main__":
    main()
