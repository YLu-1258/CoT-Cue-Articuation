import streamlit as st
import json
import pandas as pd
import os
from pathlib import Path
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Model Evaluation Checker",
    page_icon="🔍",
    layout="wide"
)

# Create data directory for persistent storage
SAVE_DIR = "/data/kevinchu/CoT-Cue-Articuation/manual_check_data"
os.makedirs(SAVE_DIR, exist_ok=True)

def get_progress_file_path(file_name):
    """Get the path for saving progress for a specific evaluation file"""
    base_name = file_name.replace('.jsonl', '')
    return os.path.join(SAVE_DIR, f"{base_name}_progress.json")

def save_progress(file_path, checked_items):
    """Save progress to disk"""
    progress_file = get_progress_file_path(os.path.basename(file_path))
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'file_path': file_path,
        'checked_items': checked_items,
        'total_checked': len(checked_items)
    }
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    return progress_file

def load_progress(file_path):
    """Load progress from disk"""
    progress_file = get_progress_file_path(os.path.basename(file_path))
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            return progress_data.get('checked_items', {})
        except:
            return {}
    return {}

def load_jsonl_file(file_path):
    """Load JSONL file and return list of dictionaries"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data

def load_responses_file(file_path):
    """Load filtered responses file and create question_id to response mapping"""
    responses = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                responses[item['question_id']] = item
    return responses

# Initialize session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'checked_items' not in st.session_state:
    st.session_state.checked_items = {}
if 'data' not in st.session_state:
    st.session_state.data = []
if 'responses_data' not in st.session_state:
    st.session_state.responses_data = {}
if 'file_path' not in st.session_state:
    st.session_state.file_path = None

# Define file paths
evaluation_files = {
    "Few-shot Black Squares": "/data/kevinchu/CoT-Cue-Articuation/data/model_evaluation/meta-llama_Llama-3.1-8B-Instruct/fewshot_black_squares_evaluations.jsonl",
    "Stanford Professor": "/data/kevinchu/CoT-Cue-Articuation/data/model_evaluation/meta-llama_Llama-3.1-8B-Instruct/stanford_professor_evaluations.jsonl"
}

response_files = {
    "Few-shot Black Squares": "/data/kevinchu/CoT-Cue-Articuation/data/responses/filtered/fewshot_black_squares_responses_filtered.jsonl",
    "Stanford Professor": "/data/kevinchu/CoT-Cue-Articuation/data/responses/filtered/stanford_professor_responses_filtered.jsonl"
}

# Minimal header
col1, col2 = st.columns([2, 1])
with col1:
    st.title("🔍 Evaluation Checker")
with col2:
    selected_file = st.selectbox("File:", list(evaluation_files.keys()), label_visibility="collapsed")

# Load data when file selection changes
if st.session_state.file_path != evaluation_files[selected_file]:
    st.session_state.file_path = evaluation_files[selected_file]
    st.session_state.data = load_jsonl_file(st.session_state.file_path)
    st.session_state.responses_data = load_responses_file(response_files[selected_file])
    st.session_state.checked_items = load_progress(st.session_state.file_path)
    
    # Find first unchecked item
    first_unchecked = 0
    for i, item in enumerate(st.session_state.data):
        if str(item['question_id']) not in st.session_state.checked_items:
            first_unchecked = i
            break
    st.session_state.current_index = first_unchecked

if st.session_state.data:
    total_items = len(st.session_state.data)
    checked_count = len(st.session_state.checked_items)
    progress_percent = (checked_count / total_items) * 100 if total_items > 0 else 0
    
    # Minimal progress and navigation in one line
    nav1, nav2, nav3, prog = st.columns([1, 1, 2, 3])
    
    with nav1:
        if st.button("⬅️", use_container_width=True):
            if st.session_state.current_index > 0:
                st.session_state.current_index -= 1
                st.rerun()
    
    with nav2:
        if st.button("➡️", use_container_width=True):
            if st.session_state.current_index < total_items - 1:
                st.session_state.current_index += 1
                st.rerun()
    
    with nav3:
        new_index = st.number_input(
            "Item:", min_value=1, max_value=total_items,
            value=st.session_state.current_index + 1,
            label_visibility="collapsed"
        ) - 1
        if new_index != st.session_state.current_index:
            st.session_state.current_index = new_index
            st.rerun()
    
    with prog:
        st.progress(progress_percent / 100)
        st.caption(f"{checked_count}/{total_items} checked ({progress_percent:.0f}%)")
    
    # Get current item data
    current_item = st.session_state.data[st.session_state.current_index]
    question_id = current_item['question_id']
    matching_response = st.session_state.responses_data.get(question_id, {})
    current_assessment = st.session_state.checked_items.get(str(question_id), {})
    
    # Main content - BIASED RESPONSE AND EVALUATION
    st.markdown("---")
    
    st.markdown(f"**Question ID:** {question_id}")
    
    # Two column layout: Biased response (larger) and Evaluation
    resp_col, eval_col = st.columns([2, 1])
    
    with resp_col:
        st.markdown("## 🎯 Model's Biased Response")
        if 'biased_response' in matching_response:
            biased_text = st.text_area(
                "Response:",
                value=matching_response['biased_response'],
                height=400,
                key="biased_response",
                disabled=True,
                label_visibility="collapsed"
            )
        else:
            st.error("Response not found")
    
    with eval_col:
        st.markdown("## 🔍 Evaluation")
        evaluation_text = st.text_area(
            "Evaluation:",
            value=current_item.get('raw_evaluation', 'No evaluation available'),
            height=400,
            key="evaluation_text",
            disabled=True,
            label_visibility="collapsed"
        )
    
    st.markdown(f"##### Acknoledged Cue: {current_item.get('acknowledged_cue', '').capitalize()}")
    
    # Quick assessment buttons right below
    st.markdown("**Is this evaluation correct for the response?**")
    assess_col1, assess_col2, assess_col3 = st.columns([1, 1, 2])
    
    assessment = None
    with assess_col1:
        if st.button("✅ CORRECT", 
                    type="primary" if current_assessment.get('assessment') == 'correct' else "secondary",
                    use_container_width=True,
                    key="correct_btn"):
            assessment = "correct"
    
    with assess_col2:
        if st.button("❌ INCORRECT", 
                    type="primary" if current_assessment.get('assessment') == 'incorrect' else "secondary",
                    use_container_width=True,
                    key="incorrect_btn"):
            assessment = "incorrect"
    
    with assess_col3:
        # Notes input
        notes = st.text_input(
            "Quick notes (optional):",
            value=current_assessment.get('notes', ''),
            key="notes_input",
            placeholder="Why correct/incorrect?"
        )
    
    # Auto-save when assessment is made
    if assessment:
        st.session_state.checked_items[str(question_id)] = {
            'assessment': assessment,
            'notes': notes,
            'timestamp': datetime.now().isoformat(),
            'question_id': question_id,
            'cue': current_item.get('cue', ''),
            'acknowledged_cue': current_item.get('acknowledged_cue', '')
        }
        save_progress(st.session_state.file_path, st.session_state.checked_items)
        
        # Auto advance
        if st.session_state.current_index < total_items - 1:
            st.session_state.current_index += 1
            st.rerun()
        else:
            st.success("🎉 All done!")
    
    # Show current status
    if current_assessment:
        status_text = f"Current: **{current_assessment.get('assessment', 'None').upper()}**"
        if current_assessment.get('notes'):
            status_text += f" - *{current_assessment.get('notes')}*"
        st.info(status_text)
    
    # Collapsible additional info
    with st.expander("📋 Additional Info"):
        st.markdown("**Original Question:**")
        if 'unbiased_question' in matching_response:
            st.text_area(
                "Question:",
                value=matching_response['unbiased_question'],
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
        st.markdown("**Biased Question:**")
        if 'biased_question' in matching_response:
            st.text_area(
                "Question:",
                value=matching_response['biased_question'],
                height=150,
                disabled=True,
                label_visibility="collapsed"
            )
        
        # Answer info
        if matching_response:
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                if 'correct_answer' in matching_response:
                    st.success(f"**Original Correct Answer:** {matching_response['correct_answer']}")
            with info_col2:
                if 'suggested_wrong_answer' in matching_response:
                    st.error(f"**Model's Biased Answer:** {matching_response['suggested_wrong_answer']}")
    
    # Export button at bottom
    if st.session_state.checked_items:
        if st.button("📁 Export Results", use_container_width=True):
            results_data = []
            for qid, assessment in st.session_state.checked_items.items():
                results_data.append({
                    'question_id': qid,
                    'assessment': assessment.get('assessment', ''),
                    'notes': assessment.get('notes', ''),
                    'timestamp': assessment.get('timestamp', '')
                })
            
            df = pd.DataFrame(results_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"evaluation_results_{selected_file.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.error("No data loaded") 