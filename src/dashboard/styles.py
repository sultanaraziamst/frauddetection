# Theme Configurations

DARK_THEME_CSS = """
<style>
    /* Global Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.2);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert Cards */
    .alert-card-high {
        background-color: rgba(239, 68, 68, 0.2);
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .alert-card-med {
        background-color: rgba(245, 158, 11, 0.2);
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    .alert-card-low {
        background-color: rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #ffffff;
    }
    
    /* Headers */
    /* Headers */
    h1, h2, h3, h4, h5, p, span {
        color: #ffffff !important;
    }
    
    /* Ensure Dataframe text is visible (black text on white/light/dark background handled by streamlit usually)
       But if we force white text globally, it breaks.
       We removed 'div' from the above rule to prevent conflict with dataframes. 
    */
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: #d1d5db !important;
    }
    
    /* Inputs */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #1f2937 !important;
        color: white !important;
        border-color: #374151 !important;
    }
    .stFileUploader {
        background-color: rgba(255,255,255,0.05);
        padding: 15px;
        border-radius: 10px;
    }
</style>
"""

LIGHT_THEME_CSS = """
<style>
    /* Global Background */
    .stApp {
        background-color: #f8fafc;
        color: #1f2937;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #4f46e5;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 14px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert Cards */
    .alert-card-high {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #991b1b;
    }
    .alert-card-med {
        background-color: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #92400e;
    }
    .alert-card-low {
        background-color: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 4px;
        margin-bottom: 10px;
        color: #065f46;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5 {
        color: #111827 !important;
    }
    p, span, div {
        color: #374151;
    }
    
    /* Table Styling */
    div[data-testid="stDataFrame"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        border: 1px solid #e5e7eb;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: #374151 !important;
    }
    
    /* Inputs */
    .stSelectbox div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #1f2937 !important;
        border-color: #d1d5db !important;
    }
</style>
"""

def get_css(theme="Dark"):
    if theme == "Dark":
        return DARK_THEME_CSS
    return LIGHT_THEME_CSS
