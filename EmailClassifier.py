import os
import pickle
import base64
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Ensure nltk resources are available
import nltk
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

# GMAIL API FUNCTIONS
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly', 'https://www.googleapis.com/auth/gmail.modify']

def authenticate_gmail():
    """Authenticate to Gmail API and return service object"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_secret_file = 'client_secret_668159476613-c0mj5k89ivn1rm2hmqjrpso8vh57bilm.apps.googleusercontent.com.json'
            
            if not os.path.exists(client_secret_file):
                st.error(f"Error: {client_secret_file} not found!")
                return None
                
            flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
            creds = flow.run_local_server(port=0)
        
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    try:
        service = build('gmail', 'v1', credentials=creds)
        return service
    except Exception as e:
        st.error(f"Error building Gmail service: {e}")
        return None

def get_email(service):
    """Fetch and process unread emails from Gmail inbox"""
    if service is None:
        return "Gmail service not available."
        
    try:
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], q="is:unread").execute()
        messages = results.get('messages', [])
    except Exception as e:
        return f"Error fetching emails: {e}"

    if not messages:
        return "No new messages."

    try:
        msg = service.users().messages().get(userId='me', id=messages[0]['id']).execute()
        message_id = messages[0]['id']
        email_data = msg.get('payload', {}).get('headers', [])
        from_name = ""
        subject = ""
        for values in email_data:
            name = values.get('name', '')
            if name == 'From':
                from_name = values.get('value', '')
            elif name == 'Subject':
                subject = values.get('value', '')

        email_body = None
        if 'payload' in msg and 'body' in msg['payload']:
            email_body = msg['payload']['body'].get('data', None)

        if not email_body and 'payload' in msg and 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                if part.get('mimeType') == 'text/plain':
                    email_body = part.get('body', {}).get('data', None)
                    if email_body:
                        break
                elif part.get('mimeType') == 'text/html':
                    email_body = part.get('body', {}).get('data', None)
                    if email_body:
                        break

        if email_body:
            try:
                byte_code = base64.urlsafe_b64decode(email_body)
                text = byte_code.decode("utf-8")
                
                # Mark as read
                service.users().messages().modify(
                    userId='me',
                    id=message_id,
                    body={'removeLabelIds': ['UNREAD']}
                ).execute()
                
                return {"id": message_id, "from": from_name, "subject": subject, "content": text}
            except Exception as e:
                return f"Error decoding email content: {e}"
        else:
            return "No readable email content found."
    except Exception as e:
        return f"Error processing email: {e}"

# DATA PREPROCESSING FUNCTIONS
def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize, remove stopwords and apply stemming
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    
    tokens = text.split()
    tokens = [ps.stem(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# MODEL TRAINING FUNCTIONS
def train_models(X_train, X_test, y_train, y_test):
    """Train all three models"""
    # Create vectorizer
    vectorizer = CountVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train models
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        
    return models, vectorizer, X_test_vec, y_test

def classify_email_text(email_text, model, vectorizer):
    """Classify a single email text"""
    # Preprocess the text
    preprocessed_text = preprocess_text(email_text)
    
    # Vectorize
    text_vec = vectorizer.transform([preprocessed_text])
    
    # Predict probability
    proba = model.predict_proba(text_vec)[0]
    
    # Get probability for each class
    if hasattr(model, 'classes_'):
        ham_idx = np.where(model.classes_ == 'ham')[0][0] if 'ham' in model.classes_ else -1
        spam_idx = np.where(model.classes_ == 'spam')[0][0] if 'spam' in model.classes_ else -1
        
        ham_prob = proba[ham_idx] if ham_idx >= 0 else 0
        spam_prob = proba[spam_idx] if spam_idx >= 0 else 0
    else:
        # Fallback if classes_ attribute doesn't exist
        prediction = model.predict(text_vec)[0]
        ham_prob = 1.0 if prediction == 'ham' else 0.0
        spam_prob = 1.0 if prediction == 'spam' else 0.0
    
    return {
        'ham': ham_prob,
        'spam': spam_prob
    }

# STREAMLIT APP
def main():
    st.set_page_config(page_title="Email Spam Classifier", page_icon="📧", layout="wide")
    
    # Title
    st.markdown("<h1 style='text-align: center; color: #4a4a4a;'>Email Spam Classifier</h1>", unsafe_allow_html=True)
    
    # Check if models exist or need to be trained
    model_file = 'email_models.pkl'
    vectorizer_file = 'email_vectorizer.pkl'
    
    # Initialize session state
    if 'models' not in st.session_state:
        st.session_state.models = None
    if 'vectorizer' not in st.session_state:
        st.session_state.vectorizer = None
    if 'email_content' not in st.session_state:
        st.session_state.email_content = ""
    if 'authentication_done' not in st.session_state:
        st.session_state.authentication_done = False
    if 'service' not in st.session_state:
        st.session_state.service = None
    if 'email_metadata' not in st.session_state:
        st.session_state.email_metadata = {"from": "", "subject": ""}
        
    # Load or train models
    if st.session_state.models is None:
        try:
            # Check if models are already saved
            if os.path.exists(model_file) and os.path.exists(vectorizer_file):
                with st.spinner('Loading models...'):
                    with open(model_file, 'rb') as f:
                        st.session_state.models = pickle.load(f)
                    with open(vectorizer_file, 'rb') as f:
                        st.session_state.vectorizer = pickle.load(f)
            else:
                with st.spinner('Training models...'):
                    # Load dataset
                    try:
                        df = pd.read_csv('Dataset.csv')
                        
                        # Check if required columns exist
                        if 'email_text' not in df.columns or 'label' not in df.columns:
                            st.error("The dataset must contain 'email_text' and 'label' columns.")
                            return
                        
                        # Preprocess texts
                        df['processed_text'] = df['email_text'].apply(preprocess_text)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            df['processed_text'], df['label'], test_size=0.2, random_state=42
                        )
                        
                        # Train models
                        models, vectorizer, X_test_vec, y_test = train_models(X_train, X_test, y_train, y_test)
                        
                        # Save models for future use
                        with open(model_file, 'wb') as f:
                            pickle.dump(models, f)
                        with open(vectorizer_file, 'wb') as f:
                            pickle.dump(vectorizer, f)
                        
                        st.session_state.models = models
                        st.session_state.vectorizer = vectorizer
                        
                    except Exception as e:
                        st.error(f"Error loading or processing dataset: {e}")
                        return
        except Exception as e:
            st.error(f"Error loading or training models: {e}")
            return

    # Create layout with columns
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Authentication button and email fetching (only show if not authenticated)
        if not st.session_state.authentication_done:
            if st.button('Connect to Gmail'):
                with st.spinner('Authenticating with Gmail...'):
                    service = authenticate_gmail()
                    if service:
                        st.session_state.service = service
                        st.session_state.authentication_done = True
                        st.success('Successfully connected to Gmail')
                        st.experimental_rerun()
                    else:
                        st.error('Failed to authenticate with Gmail')
        
        # Email fetching (only show if authenticated)
        if st.session_state.authentication_done:
            if st.button('Get Unread Email'):
                with st.spinner('Fetching email...'):
                    email_data = get_email(st.session_state.service)
                    if isinstance(email_data, dict):
                        st.session_state.email_content = email_data.get('content', '')
                        st.session_state.email_metadata = {
                            "from": email_data.get('from', ''),
                            "subject": email_data.get('subject', '')
                        }
                        st.success('Email fetched successfully')
                    else:
                        st.info(email_data)  # Show error or info message
        
        # Model selection with radio button (corrected to prevent multiple selections)
        st.write('Select classification model:')
        model_options = ['Multinomial Naive Bayes', 'Decision Tree Classifier', 'Random Forest Classifier']
        selected_model = st.radio("", model_options, key='model_selection')
        
        # Classify button
        if st.button('Classify Email'):
            email_text = st.session_state.email_content
            if selected_model and st.session_state.models and email_text:
                with st.spinner('Classifying email...'):
                    model = st.session_state.models[selected_model]
                    vectorizer = st.session_state.vectorizer
                    
                    # Classify the email
                    result = classify_email_text(email_text, model, vectorizer)
                    
                    # Store results in session state for display
                    st.session_state.result = result
            elif not email_text:
                st.warning('Please fetch an email or enter text first')
                st.session_state.result = None

    with col2:
        # Display email metadata if available
        if st.session_state.email_metadata["from"] or st.session_state.email_metadata["subject"]:
            st.subheader("Email Details")
            st.write(f"**From:** {st.session_state.email_metadata['from']}")
            st.write(f"**Subject:** {st.session_state.email_metadata['subject']}")
        
        # Email content text area (editable)
        st.subheader("Email Content")
        email_text = st.text_area("", 
                                  value=st.session_state.email_content, 
                                  height=300, 
                                  key="email_text_input")
        
        # Update session state with any changes made by the user
        st.session_state.email_content = email_text
        
        # Display classification results if available
        if 'result' in st.session_state and st.session_state.result:
            st.subheader('Classification Results:')
            
            # Format percentages
            ham_percentage = st.session_state.result['ham'] * 100
            spam_percentage = st.session_state.result['spam'] * 100
            
            # Display horizontal bars with improved styling
            st.markdown(
                f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 5px; margin-top: 10px;">
                    <div style="margin-bottom: 15px;">
                        <span style="display: inline-block; width: 60px; font-weight: bold;">Ham:</span>
                        <div style="display: inline-block; width: {ham_percentage}%; height: 20px; background-color: #28a745; border-radius: 5px;"></div>
                        <span style="margin-left: 10px; font-weight: bold;">{ham_percentage:.1f}%</span>
                    </div>
                    <div>
                        <span style="display: inline-block; width: 60px; font-weight: bold;">Spam:</span>
                        <div style="display: inline-block; width: {spam_percentage}%; height: 20px; background-color: #dc3545; border-radius: 5px;"></div>
                        <span style="margin-left: 10px; font-weight: bold;">{spam_percentage:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Display final verdict
            if ham_percentage >= spam_percentage:
                st.markdown(f"<p style='font-size: 18px; font-weight: bold; color: #28a745;'>Verdict: This email is likely <span style='color: #28a745;'>HAM</span> (legitimate)</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='font-size: 18px; font-weight: bold; color: #dc3545;'>Verdict: This email is likely <span style='color: #dc3545;'>SPAM</span></p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()