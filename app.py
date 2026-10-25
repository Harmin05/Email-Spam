import streamlit as st
import pandas as pd
import tempfile
import os
import time
from src.pipeline.prediction_pipeline import PredictionPipeline

# Page configuration
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

# Initialize pipeline
@st.cache_resource
def get_pipeline():
    return PredictionPipeline(load_models=True)

try:
    pipeline = get_pipeline()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

st.title("📧 Spam Email Classifier")
st.markdown("Classify emails as **Spam** or **Ham** (Clean) using Machine Learning.")

# Tabs for different modes
tab1, tab2 = st.tabs(["Single Email", "Batch MBOX Processing"])

with tab1:
    st.header("Check a Single Email")
    email_text = st.text_area("Paste the email content here:", height=200, placeholder="Dear friend, I have a business proposal...")
    
    if st.button("Classify Email", type="primary"):
        if email_text.strip():
            with st.spinner("Analyzing..."):
                try:
                    # Get prediction
                    result = pipeline.predict_single_email(email_text)
                    prediction = result['prediction']
                    confidence = result.get('confidence', 0)
                    
                    # Display result
                    if prediction == "Spam":
                        st.error(f"🚨 This email is **SPAM**")
                    else:
                        st.success(f"✅ This email is **HAM** (Safe)")
                    
                    if confidence:
                        st.info(f"Confidence Score: {confidence:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error analyzing email: {str(e)}")
        else:
            st.warning("Please enter some text to classify.")

with tab2:
    st.header("Process MBOX File")
    uploaded_file = st.file_uploader("Upload an MBOX file", type=['mbox', 'txt'])
    
    if uploaded_file is not None:
        file_key = f"mbox_{uploaded_file.name}_{uploaded_file.size}"
        
        # If new file is uploaded, clear old results
        if st.session_state.get("current_mbox_key") != file_key:
            st.session_state.pop("mbox_df", None)
            
        if st.button("Process File", type="primary"):
            with st.spinner("Processing file... this may take a moment"):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mbox') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        df = pipeline.predict_mbox_file(tmp_path)
                        st.session_state["mbox_df"] = df
                        st.session_state["current_mbox_key"] = file_key
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

        # Render preview when results exist in session state
        if "mbox_df" in st.session_state:
            df = st.session_state["mbox_df"]
            
            # Metrics
            total_count = len(df)
            spam_count = len(df[df['Prediction'] == 'Spam'])
            ham_count = len(df[df['Prediction'] == 'Ham'])
            spam_pct = (spam_count / total_count * 100) if total_count > 0 else 0
            
            st.markdown("### 📊 Summary Metrics")
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Emails", total_count)
            col2.metric("🚨 Spam Detected", f"{spam_count} ({spam_pct:.1f}%)")
            col3.metric("✅ Clean (Ham)", f"{ham_count} ({100 - spam_pct:.1f}%)")
            
            st.divider()
            st.markdown("### 📋 Results Preview Table")
            
            # Filter and Search
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                filter_option = st.selectbox(
                    "Filter by Classification:",
                    ["All Emails", "🚨 Spam Only", "✅ Ham Only"],
                    key="filter_option"
                )
            with f_col2:
                search_query = st.text_input(
                    "🔍 Search Subject or Body:",
                    placeholder="Type keyword to filter...",
                    key="search_query"
                )
            
            filtered_df = df.copy()
            if filter_option == "🚨 Spam Only":
                filtered_df = filtered_df[filtered_df['Prediction'] == 'Spam']
            elif filter_option == "✅ Ham Only":
                filtered_df = filtered_df[filtered_df['Prediction'] == 'Ham']
            
            if search_query.strip():
                q = search_query.strip().lower()
                filtered_df = filtered_df[
                    filtered_df['Subject'].astype(str).str.lower().str.contains(q) |
                    filtered_df['Body'].astype(str).str.lower().str.contains(q)
                ]
            
            # Formatted status column
            filtered_df['Status'] = filtered_df['Prediction'].apply(
                lambda p: "🚨 Spam" if p == "Spam" else "✅ Ham"
            )
            
            # Clean dataframe for display
            display_df = filtered_df[['Status', 'Subject', 'Recipients', 'Time', 'Body']].copy()
            display_df = display_df.rename(columns={
                'Time': 'Date / Time',
                'Body': 'Content Preview'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Classification Result",
                        width="small"
                    ),
                    "Subject": st.column_config.TextColumn(
                        "Subject",
                        width="medium"
                    ),
                    "Recipients": st.column_config.TextColumn(
                        "Recipients / Sender",
                        width="medium"
                    ),
                    "Date / Time": st.column_config.TextColumn(
                        "Date / Time",
                        width="medium"
                    ),
                    "Content Preview": st.column_config.TextColumn(
                        "Content Preview",
                        width="large"
                    ),
                }
            )
            
            st.caption(f"Showing **{len(display_df)}** of **{total_count}** emails")
            
            # Email Inspector
            if len(filtered_df) > 0:
                with st.expander("🔍 Inspect Full Email Details", expanded=False):
                    email_options = [
                        f"{row['Status']} | {str(row['Subject'])[:50]}... ({row['Time']})"
                        for _, row in filtered_df.iterrows()
                    ]
                    selected_idx = st.selectbox(
                        "Select email to inspect:",
                        options=range(len(email_options)),
                        format_func=lambda i: email_options[i]
                    )
                    selected_email = filtered_df.iloc[selected_idx]
                    
                    st.markdown(f"**Classification:** {selected_email['Status']}")
                    st.markdown(f"**Subject:** {selected_email['Subject']}")
                    st.markdown(f"**Date:** {selected_email['Time']}")
                    st.markdown(f"**Recipients/Sender:** `{selected_email['Recipients']}`")
                    st.text_area("Full Body Text:", value=selected_email['Body'], height=160, disabled=True)
            
            st.divider()
            
            # Download options
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                csv_all = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Results (CSV)",
                    data=csv_all,
                    file_name=f"all_predictions_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with d_col2:
                spam_df = df[df['Prediction'] == 'Spam']
                csv_spam = spam_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="🚨 Download Spam Only (CSV)",
                    data=csv_spam,
                    file_name=f"spam_only_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
