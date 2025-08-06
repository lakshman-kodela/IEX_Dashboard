import streamlit as st
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import re
import os
import numpy as np

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import json

# Page configuration
st.set_page_config(page_title="AI Analytics Chat", layout="wide")

# Load API Key from .env
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'available_files' not in st.session_state:
    st.session_state.available_files = []

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .dashboard-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e0e0e0;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border: none;
        border-radius: 4px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .success-message {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #2E7D32;
        margin: 10px 0;
    }
    .info-box {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0; 
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 5px solid #1E88E5;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #F5F5F5;
        border-left: 5px solid #9E9E9E;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        background-color: #FAFAFA;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .message-time {
        font-size: 0.8rem;
        color: #757575;
        margin-top: 0.3rem;
        align-self: flex-end;
    }
    .file-info {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .error-message {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #C62828;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def load_excel_files():
    """Load multiple Excel and CSV files from the current directory"""
    files = [
        "Weather_Data/Weather_Hyderabad.xlsx",
        "Weather_Data/Weather_Bengaluru.xlsx",
        "Weather_Data/Weather_Chennai.xlsx",
        "Weather_Data/Weather_Delhi.xlsx",
        "Weather_Data/Weather_Mumbai.xlsx",
        "Weather_Data/Weather_Ahmedabad.xlsx",
        "Weather_Data/Weather_Kolkatta.xlsx",
        "DAM_output_cleaned.xlsx",
        "GDAM_output_cleaned.xlsx",
        "RTM_output_cleaned_converted.xlsx",
        "monthly_coal_full_data.csv"
    ]
    loaded_files = {}

    for file_name in files:
        try:
            if os.path.exists(file_name):
                # Check file extension and load accordingly
                if file_name.endswith('.csv'):
                    df = pd.read_csv(file_name)
                elif file_name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(file_name)
                else:
                    st.warning(f"⚠️ Unsupported file format: {file_name}")
                    continue

                # Detect potential date columns that weren't automatically parsed
                for col in df.select_dtypes(include=['object']).columns:
                    try:
                        # First check if column name suggests a date
                        if any(date_hint in col.lower() for date_hint in
                               ['date', 'time', 'day', 'year', 'month', 'dt_']):
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                            continue

                        # Sample the first few non-null values to check if they look like dates
                        sample = df[col].dropna().head(5)
                        if len(sample) > 0:
                            # Check if the column might contain dates
                            if all(isinstance(val, str) for val in sample):
                                # Look for date-like patterns in the sample
                                date_patterns = [
                                    r'\d{1,4}[-/]\d{1,2}[-/]\d{1,4}',  # yyyy-mm-dd or dd/mm/yyyy formats
                                    r'\d{1,2}[-/]\w{3}[-/]\d{2,4}',  # dd-mmm-yyyy formats
                                    r'\w{3,9} \d{1,2},? \d{2,4}'  # Month dd, yyyy formats
                                ]

                                is_date_like = any(
                                    any(re.search(pattern, str(val)) for pattern in date_patterns)
                                    for val in sample
                                )

                                if is_date_like:
                                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        # If conversion fails, keep as is
                        pass

                # Basic data cleaning
                # Fill missing numeric values with mean
                for col in df.select_dtypes(include=['number']).columns:
                    df[col] = df[col].fillna(df[col].mean())

                # Fill missing categorical values with mode
                for col in df.select_dtypes(include=['object']).columns:
                    if df[col].notna().any():
                        df[col] = df[col].fillna(df[col].mode()[0])
                    else:
                        df[col] = df[col].fillna("Unknown")

                # Fill missing datetime values with the median date
                for col in df.select_dtypes(include=['datetime']).columns:
                    if df[col].notna().any():
                        median_date = df[col].dropna().median()
                        df[col] = df[col].fillna(median_date)

                loaded_files[file_name] = df
                st.success(f"✅ Successfully loaded {file_name}")
            else:
                st.warning(f"⚠️ File {file_name} not found in the current directory")
        except Exception as e:
            st.error(f"❌ Error loading {file_name}: {str(e)}")

    return loaded_files


def extract_file_keywords(df, file_name):
    """
    Dynamically extract keywords from a file based on its content
    """
    keywords = set()

    # Extract from file name (remove extension and split by common separators)
    base_name = file_name.replace('.xlsx', '').replace('.csv', '')
    name_parts = base_name.replace('_', ' ').replace('-', ' ').split()
    keywords.update([part.lower() for part in name_parts if len(part) > 2])

    # Extract from column names
    for col in df.columns:
        col_words = str(col).lower().replace('_', ' ').replace('-', ' ').split()
        keywords.update([word for word in col_words if len(word) > 2])

    # Extract from categorical data values (top frequent values)
    for col in df.select_dtypes(include=['object']).columns:
        try:
            top_values = df[col].value_counts().head(10).index.tolist()
            for value in top_values:
                if isinstance(value, str) and len(value) > 2:
                    value_words = value.lower().replace('_', ' ').replace('-', ' ').split()
                    keywords.update([word for word in value_words if len(word) > 2])
        except:
            continue

    # Extract from numeric column statistics (if they suggest specific domains)
    for col in df.select_dtypes(include=['number']).columns:
        col_name = str(col).lower()
        # Add domain-specific keywords based on column patterns
        if any(term in col_name for term in ['price', 'cost', 'amount', 'value']):
            keywords.add('financial')
        elif any(term in col_name for term in ['date', 'time', 'year', 'month']):
            keywords.add('temporal')
        elif any(term in col_name for term in ['quantity', 'volume', 'count']):
            keywords.add('quantity')

    return list(keywords)


def detect_relevant_files(question, available_files, threshold=0.1):
    """
    Dynamically detect relevant files based on user question
    """
    if not available_files:
        return []

    file_scores = {}
    file_keywords_cache = {}

    # Calculate relevance scores for each file
    for file_name, df in available_files.items():
        # Extract keywords for this file
        keywords = extract_file_keywords(df, file_name)
        file_keywords_cache[file_name] = keywords

        # Calculate relevance score
        score = calculate_relevance_score(question, keywords, df, file_name)
        file_scores[file_name] = score

    # Sort files by relevance score
    sorted_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine relevant files
    relevant_files = []

    if sorted_files:
        # If top score is above threshold, include it
        if sorted_files[0][1] > threshold:
            relevant_files.append(sorted_files[0][0])

            # Include additional files if they're close to the top score
            top_score = sorted_files[0][1]
            for file_name, score in sorted_files[1:]:
                if score > threshold and score >= top_score * 0.7:  # Within 70% of top score
                    relevant_files.append(file_name)
        else:
            # If no files meet threshold, include all (general analysis)
            relevant_files = list(available_files.keys())

    return relevant_files


def generate_multi_file_context(relevant_files, available_files):
    """
    Generate comprehensive context from multiple relevant files
    """
    if not relevant_files:
        return "No relevant files found."

    multi_file_context = f"Analyzing {len(relevant_files)} relevant file(s):\n\n"

    for file_name in relevant_files:
        df = available_files[file_name]

        # File header
        multi_file_context += f"## File: {file_name}\n"
        multi_file_context += f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n\n"

        # Dynamic column analysis
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

        # Column information
        multi_file_context += "### Columns:\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            null_percentage = round((1 - non_null_count / len(df)) * 100, 2)

            if df[col].dtype.kind in 'ifc':  # numeric
                try:
                    sample = f"Range: {df[col].min()} to {df[col].max()}, Mean: {round(df[col].mean(), 2)}"
                except:
                    sample = "Statistical summary unavailable"
            elif df[col].dtype.kind == 'O':  # object/string
                unique_count = df[col].nunique()
                try:
                    top_values = df[col].value_counts().head(3).index.tolist()
                    sample = f"{unique_count} unique values. Top: {top_values}"
                except:
                    sample = f"{unique_count} unique values"
            elif df[col].dtype.kind == 'M':  # datetime
                try:
                    sample = f"Range: {df[col].min()} to {df[col].max()}"
                except:
                    sample = "Date range unavailable"
            else:
                sample = "Analysis unavailable"

            multi_file_context += f"- {col} ({dtype}): {null_percentage}% missing, {sample}\n"

        # Data preview
        multi_file_context += "\n### Data Preview:\n"
        try:
            if len(df) <= 3:
                preview = df.to_string(index=False)
            else:
                preview = df.head(3).to_string(index=False)
            multi_file_context += f"{preview}\n"
        except:
            multi_file_context += "Preview unavailable\n"

        # Statistical summary for numeric columns
        if numeric_cols:
            multi_file_context += "\n### Numeric Summary:\n"
            try:
                numeric_summary = df[numeric_cols].describe()
                multi_file_context += f"{numeric_summary.to_string()}\n"
            except:
                multi_file_context += "Numeric summary unavailable\n"

        # Categorical summary
        if categorical_cols:
            multi_file_context += "\n### Categorical Summary:\n"
            for col in categorical_cols[:3]:  # Limit to first 3
                try:
                    unique_count = df[col].nunique()
                    multi_file_context += f"{col}: {unique_count} unique values\n"
                    if unique_count <= 10:
                        distribution = df[col].value_counts().head(5)
                        multi_file_context += f"  Distribution: {distribution.to_dict()}\n"
                except:
                    multi_file_context += f"{col}: Analysis unavailable\n"

        multi_file_context += "\n" + "=" * 50 + "\n\n"

    return multi_file_context

def calculate_relevance_score(question, file_keywords, df, file_name):
    """
    Calculate relevance score between user question and file content
    """
    question_lower = question.lower()

    # Direct keyword matching score
    keyword_matches = sum(1 for keyword in file_keywords if keyword in question_lower)
    keyword_score = keyword_matches / len(file_keywords) if file_keywords else 0

    # Column name matching score
    column_matches = sum(1 for col in df.columns if str(col).lower() in question_lower)
    column_score = column_matches / len(df.columns) if len(df.columns) > 0 else 0

    # File name matching score
    file_name_words = file_name.replace('.xlsx', '').replace('_', ' ').lower().split()
    file_name_matches = sum(1 for word in file_name_words if word in question_lower)
    file_name_score = file_name_matches / len(file_name_words) if file_name_words else 0

    # Semantic similarity using TF-IDF (if available)
    try:
        # Create document from file content
        file_content = ' '.join(file_keywords + [str(col) for col in df.columns])

        # Use TF-IDF for semantic similarity
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
        tfidf_matrix = vectorizer.fit_transform([question_lower, file_content])

        if tfidf_matrix.shape[0] > 1:
            semantic_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        else:
            semantic_score = 0
    except:
        semantic_score = 0

    # Weighted combination of scores
    total_score = (
            keyword_score * 0.3 +
            column_score * 0.3 +
            file_name_score * 0.2 +
            semantic_score * 0.2
    )

    return total_score


def create_vector_store(df):
    """Create FAISS vector store from dataframe for semantic search"""
    try:
        # Convert dataframe to text documents
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = []

        # Create documents from each row
        for i, row in df.iterrows():
            content = " ".join([f"{col}: {val}" for col, val in row.items()])
            documents.append(Document(page_content=content, metadata={"row_index": i}))

        # Split documents
        docs = text_splitter.split_documents(documents)

        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Create vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.warning(f"Could not create vector store: {str(e)}")
        return None

def get_greeting():
    """Generate appropriate greeting based on time of day"""
    current_hour = datetime.datetime.now().hour

    if current_hour < 12:
        return "Good morning"
    elif current_hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


def extract_code_blocks(text):
    # Simple extraction of Python code blocks from markdown
    code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
    return [code.strip() for code in code_blocks]


def execute_chart_code(code, available_files, relevant_files):
    """Execute chart code and return the Plotly figure as JSON"""
    try:
        # Create a safe execution environment with access to data
        exec_globals = {
            "px": px,
            "go": go,
            "make_subplots": make_subplots,
            "pd": pd,
            "numpy": np,
            "np": np,
            "json": json
        }

        # Add relevant dataframes to the execution context
        for file_name in relevant_files:
            if file_name in available_files:
                df = available_files[file_name]
                # Create a clean variable name from filename
                var_name = file_name.replace('.xlsx', '').replace('.csv', '').replace(' ', '_').replace('-',
                                                                                                        '_').replace(
                    '/', '_')
                exec_globals[var_name] = df
                exec_globals['df'] = df  # Also provide as 'df' for convenience

        exec_locals = {}

        # Add debugging info before execution
        print(f"Available columns in df: {list(exec_globals['df'].columns)}")

        exec(code, exec_globals, exec_locals)

        # Get the figure from locals (assumes the code creates a variable called 'fig')
        if 'fig' in exec_locals:
            fig = exec_locals['fig']
            # Convert to JSON for storage
            fig_json = fig.to_json()
            return fig_json, True
        else:
            return "⚠️ Error: No figure variable 'fig' found in the code", False

    except Exception as e:
        # Get the actual dataframe to show available columns
        try:
            df = exec_globals.get('df', None)
            if df is not None:
                available_cols = list(df.columns)
                error_msg = f"⚠️ Error creating chart: {str(e)}\n\nAvailable columns: {available_cols}"
                return error_msg, False
            else:
                return f"⚠️ Error creating chart: {str(e)}", False
        except:
            return f"⚠️ Error creating chart: {str(e)}", False


# Update the chart generation prompt in generate_ai_response function
def generate_ai_response(question, available_files, chat_history):
    """Generate AI response - either text analysis or chart"""
    try:
        if not GEMINI_API_KEY:
            return "To enable AI analysis, please add your Google Gemini API key in the .env file.", False

        relevant_files = detect_relevant_files(question, available_files)
        if not relevant_files:
            return "I couldn't find any relevant files to analyze for your question. Please try rephrasing your question.", False

        multi_file_context = generate_multi_file_context(relevant_files, available_files)
        recent_history = chat_history[-6:] if chat_history else []
        chat_context = "\n".join([f"{'User' if i % 2 == 0 else 'AI'}: {msg}" for i, msg in enumerate(recent_history)])

        # Check if user is asking for a chart/visualization
        chart_keywords = ["chart", "plot", "visual", "graph", "visualize", "show me", "create a", "make a", "draw",
                          "histogram", "bar chart", "line chart", "scatter plot", "pie chart"]
        is_chart_request = any(kw in question.lower() for kw in chart_keywords)

        if is_chart_request:
            # Brief chart description (2-3 lines)
            description_prompt = f"""Based on the user's request, provide a brief 2-3 line description of what chart will be created.

            User Question: {question}
            Available Data: {multi_file_context}

            Format: **[Chart Type]**: Brief description of what data will be visualized and why.
            Keep it concise and use bold formatting for the chart type.
            """

            model = genai.GenerativeModel('gemini-2.0-flash')
            generation_config = {"temperature": 0.1, "top_p": 0.95, "top_k": 40}

            description_response = model.generate_content(description_prompt, generation_config=generation_config)
            chart_description = description_response.text if hasattr(description_response, 'text') else description_response.candidates[0].content.parts[0].text

            # Enhanced prompt for Plotly chart generation
            # Enhanced prompt for Plotly chart generation
            code_prompt = f"""You are a data visualization expert. Generate Python Plotly code to create an interactive chart based on the user's request.

            # Available Data Files and Context:
            {multi_file_context}

            # User Question:
            {question}

            # CRITICAL INSTRUCTIONS:
            1. **ALWAYS check column names exactly as they appear in the data context above**
            2. **Use df.columns.tolist() to verify column names if needed**
            3. **Column names are case-sensitive and may contain spaces or special characters**
            4. **Wrap column names with spaces or special characters in quotes**
            5. Generate ONLY Python code that creates a chart using Plotly (px or go)
            6. Use 'df' as the variable name for the main dataframe
            7. The code should be complete and executable
            8. Create a variable named 'fig' that contains the Plotly figure
            9. Include proper labels, titles, and formatting
            10. Use appropriate chart types based on data types
            11. Make the chart interactive with hover information
            12. Handle missing data appropriately
            13. Use colors and styling to make the chart visually appealing
            14. Do NOT include fig.show() - just create the figure

            # Example of handling column names with spaces:
            # Correct: df['Column Name'] or df["Column Name"]
            # Wrong: df[Column Name] or df.Column Name

            # Before using any column, you can add this check:
            # print("Available columns:", df.columns.tolist())

            Return only the Python code without any markdown formatting or explanations:
            """

            code_response = model.generate_content(code_prompt, generation_config=generation_config)
            response_text = code_response.text if hasattr(code_response, 'text') else code_response.candidates[0].content.parts[0].text

            # Clean the response to get pure Python code
            clean_code = response_text.replace("```python", "").replace("```", "").strip()

            # Execute the code and get the chart
            chart_result, success = execute_chart_code(clean_code, available_files, relevant_files)

            if success:
                # Generate chart explanation (6-10 lines)
                explanation_prompt = f"""Provide a 6-10 line explanation of what this chart shows and the data trends.

                User Question: {question}
                Available Data Context: {multi_file_context}
                Chart Code: {clean_code}

                Focus on:
                - Key values and trends visible in the chart
                - Notable patterns or insights
                - What the data tells us
                - Any significant observations

                Keep it concise and informative.
                """

                explanation_response = model.generate_content(explanation_prompt, generation_config=generation_config)
                chart_explanation = explanation_response.text if hasattr(explanation_response, 'text') else explanation_response.candidates[0].content.parts[0].text

                # Return chart with description and explanation
                return (chart_result, chart_description, chart_explanation), True
            else:
                return f"I encountered an issue creating the chart: {chart_result}", False
        else:
            # Regular analysis prompt - NO CODE GENERATION
            prompt = f"""You are an advanced data analysis AI assistant designed for non-technical users. You provide clear, easy-to-understand insights without any code.

                        # Available Data Files and Context:
                        {multi_file_context}

                        # Recent Conversation:
                        {chat_context}

                        # User Question:
                        {question}

                        # Response Guidelines:
                        1. **User-Friendly Language**: Use simple, clear language that non-technical users can understand
                        2. **Direct Answers**: Start with a direct answer to the question
                        3. **Data Insights**: Provide specific insights, trends, and patterns from the data
                        4. **No Code**: NEVER include any code, programming syntax, or technical jargon
                        5. **Clear Explanations**: Explain what the data shows in plain English
                        6. **Actionable Insights**: Provide practical recommendations based on the analysis
                        7. **Visual Descriptions**: If describing data patterns, use clear descriptive language
                        8. **Multi-File Analysis**: When relevant, compare insights across different files

                        # Response Requirements:
                        - Use conversational, friendly tone
                        - Include specific numbers and statistics in easy-to-read format
                        - Highlight key findings and trends
                        - Provide context for what the numbers mean
                        - Use bullet points or clear formatting for readability
                        - NO programming code or technical implementation details
                        - Focus on business insights and practical implications

                        Provide a comprehensive, user-friendly analysis without any code or technical details.
                        """

            model = genai.GenerativeModel('gemini-2.0-flash')
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
            }

            response = model.generate_content(prompt, generation_config=generation_config)
            response_text = response.text if hasattr(response, 'text') else response.candidates[0].content.parts[0].text

            return response_text + f"\n\n---\n📊 **Files Analyzed**: {', '.join(relevant_files)}", False

    except Exception as e:
        return f"❌ Error analyzing your data: {str(e)}", False


def display_file_info(df, file_name):
    """Display information about the current file"""
    st.markdown(f"""
    <div class="file-info">
        <h4>📊 Current Dataset: {file_name}</h4>
        <p><strong>Rows:</strong> {df.shape[0]} | <strong>Columns:</strong> {df.shape[1]}</p>
        <p><strong>Numeric Columns:</strong> {len(df.select_dtypes(include=['number']).columns)} | 
        <strong>Categorical Columns:</strong> {len(df.select_dtypes(include=['object']).columns)}</p>
    </div>
    """, unsafe_allow_html=True)


def display_chat_message(message, is_user=False):
    """Display a chat message with styling - handles both text and Plotly charts"""
    message_class = "user-message" if is_user else "bot-message"
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # Create a unique key using timestamp and random number
    import random
    unique_key = f"chart_{timestamp.replace(':', '')}_{random.randint(1000, 9999)}"

    # Handle chart messages with description and explanation
    if isinstance(message, tuple) and len(message) == 4 and message[3] == "CHART":
        chart_json, description, explanation = message[0], message[1], message[2]

        # Display brief chart description (2-3 lines) - NO timestamp here
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div>{description}</div>
        </div>
        """, unsafe_allow_html=True)

        # Display the Plotly chart
        try:
            # Convert JSON back to Plotly figure
            import plotly.graph_objects as go
            fig = go.Figure(json.loads(chart_json))

            # Display the interactive chart with unique key
            st.plotly_chart(fig, use_container_width=True, key=unique_key)

        except Exception as e:
            st.error(f"Error displaying chart: {str(e)}")

        # Display chart explanation (6-10 lines) with timestamp - ONLY timestamp here
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div>{explanation}</div>
            <div class="message-time">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

    # Handle old chart format for backward compatibility
    elif isinstance(message, tuple) and len(message) == 2 and message[1] == "CHART":
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div>📊 Here's your interactive chart:</div>
            <div class="message-time">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            # For old format, assume it's JSON
            import plotly.graph_objects as go
            fig = go.Figure(json.loads(message[0]))
            st.plotly_chart(fig, use_container_width=True, key=f"chart_old_{unique_key}")
        except:
            st.error("Error displaying chart")
    else:
        # Regular text message
        st.markdown(f"""
        <div class="chat-message {message_class}">
            <div>{message}</div>
            <div class="message-time">{timestamp}</div>
        </div>
        """, unsafe_allow_html=True)

def ai_chatbot_page():
    """
    Updated chatbot page with proper chart handling
    """
    st.markdown('<div class="sub-header">🤖 AI Data Analysis Assistant</div>', unsafe_allow_html=True)

    # Load Excel files automatically
    if not st.session_state.available_files:
        st.info("Loading Excel files...")
        loaded_files = load_excel_files()
        if loaded_files:
            st.session_state.available_files = loaded_files
            st.success(f"✅ Loaded {len(loaded_files)} files: {', '.join(loaded_files.keys())}")
        else:
            st.error("No Excel files could be loaded. Please check that Excel files exist in the current directory.")
            return

    # Display available files info
    if st.session_state.available_files:
        st.markdown("### 📁 Available Datasets")
        cols = st.columns(min(len(st.session_state.available_files), 4))  # Limit to 4 columns
        for i, (file_name, df) in enumerate(st.session_state.available_files.items()):
            with cols[i % 4]:
                st.metric(
                    label=file_name,
                    value=f"{df.shape[0]} rows",
                    delta=f"{df.shape[1]} columns"
                )

    # Initialize chat
    if len(st.session_state.chat_history) == 0:
        greeting = get_greeting()
        file_list = ", ".join(st.session_state.available_files.keys())
        welcome_message = f"""{greeting}! I'm your AI data analysis assistant. 

I have access to your datasets: **{file_list}**

I can automatically detect which files are relevant to your questions and provide comprehensive analysis across multiple files when needed. You can ask me to:
- Analyze data trends and patterns
- Create charts and visualizations
- Compare data across different files
- Provide statistical insights

What would you like to analyze?"""
        st.session_state.chat_history.append(welcome_message)

    # Display chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for i, message in enumerate(st.session_state.chat_history):
        is_user = i % 2 != 0
        display_chat_message(message, is_user)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    user_question = st.chat_input("Ask about your data or request a chart...", key="user_question")

    # In the ai_chatbot_page function, modify the response handling:
    if user_question:
        st.session_state.chat_history.append(user_question)

        with st.spinner("Analyzing data across relevant files..."):
            response, is_chart = generate_ai_response(
                user_question,
                st.session_state.available_files,
                st.session_state.chat_history
            )


            # Handle chart vs text response
            if is_chart:
                # Store chart data with description and explanation
                chart_data, description, explanation = response
                st.session_state.chat_history.append((chart_data, description, explanation, "CHART"))
            else:
                st.session_state.chat_history.append(response)

        st.rerun()

    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    with col2:
        if st.button("Show All Data Preview", key="show_preview"):
            for file_name, df in st.session_state.available_files.items():
                with st.expander(f"Preview: {file_name}", expanded=False):
                    st.dataframe(df.head(5))


def main():
    st.markdown('<div class="main-header">🤖 AI Analytics Chat Interface</div>', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.freepik.com/premium-vector/modern-data-analytic-accounting-logo-design_273648-1180.jpg",
                 use_container_width=True)

        st.title("AI Chat Interface")

        # Show current file info in sidebar
        if st.session_state.current_file:
            st.markdown("### Current Dataset")
            st.info(f"📄 {st.session_state.current_file}")
            if st.session_state.data is not None:
                st.metric("Rows", st.session_state.data.shape[0])
                st.metric("Columns", st.session_state.data.shape[1])

        # Show available files
        if st.session_state.available_files:
            st.markdown("### Available Files")
            for file_name in st.session_state.available_files.keys():
                if file_name == st.session_state.current_file:
                    st.success(f"📄 {file_name} (Current)")
                else:
                    st.info(f"📄 {file_name}")

        st.markdown("---")
        st.markdown("### About")
        with st.expander("About This Chat"):
            st.write("""
            This AI-powered chat interface allows you to:
            - Analyze multiple Excel files
            - Ask questions about your data in natural language
            - Get statistical insights and analysis
            - Explore correlations and patterns
            - Receive data-driven recommendations
            """)

        st.markdown("AI Chat Interface")
        st.markdown("Created by Lakshman Kodela")

    # Main content area
    ai_chatbot_page()


# Run the app
if __name__ == "__main__":
    main()
