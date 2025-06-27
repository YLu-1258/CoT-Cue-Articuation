#!/usr/bin/env python3
"""
Simple JSON viewer for JSONL files.

Displays all JSON data in a clean, readable format with basic navigation.
"""

import streamlit as st
import json
from pathlib import Path
from typing import Dict, List


@st.cache_data
def load_json_data(file_path: str) -> List[Dict]:
    """Load JSON data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    entry = json.loads(line.strip())
                    data.append(entry)
                except json.JSONDecodeError as e:
                    st.error(f"JSON decode error on line {line_num}: {e}")
                    continue
    return data


def display_json_entry(entry: Dict, index: int):
    """Display a single JSON entry."""
    with st.expander(f"📄 Entry {index + 1}", expanded=False):
        st.json(entry)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Simple JSON Viewer",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Simple JSON Viewer")
    st.markdown("View all JSON data from JSONL files")
    
    # File input
    uploaded_file = st.file_uploader(
        "Choose a JSONL file",
        type=['jsonl', 'json'],
        help="Upload a JSONL file to view its contents"
    )
    
    # Default file path input
    if not uploaded_file:
        default_path = st.text_input(
            "Or enter file path:",
            value="data/bias_analysis/stanford_professor_responses_biased.jsonl",
            help="Enter the path to a JSONL file on the server"
        )
        
        if default_path and Path(default_path).exists():
            try:
                data = load_json_data(default_path)
                st.success(f"✅ Loaded {len(data)} entries from {default_path}")
            except Exception as e:
                st.error(f"❌ Failed to load file: {e}")
                return
        else:
            st.info("Please upload a file or enter a valid file path to begin")
            return
    else:
        # Handle uploaded file
        try:
            content = uploaded_file.read().decode('utf-8')
            data = []
            for line_num, line in enumerate(content.split('\n'), 1):
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        data.append(entry)
                    except json.JSONDecodeError as e:
                        st.error(f"JSON decode error on line {line_num}: {e}")
                        continue
            
            st.success(f"✅ Loaded {len(data)} entries from uploaded file")
        except Exception as e:
            st.error(f"❌ Failed to load uploaded file: {e}")
            return
    
    if not data:
        st.warning("No data found in the file")
        return
    
    # Display options
    st.subheader("📊 Display Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        entries_per_page = st.selectbox(
            "Entries per page",
            [5, 10, 20, 50, 100],
            index=2
        )
    
    with col2:
        view_mode = st.selectbox(
            "View mode",
            ["Expandable cards", "Full display", "Raw JSON"],
            index=0
        )
    
    # Pagination
    total_pages = (len(data) - 1) // entries_per_page + 1
    
    if total_pages > 1:
        page = st.selectbox(
            "Page",
            range(1, total_pages + 1),
            index=0
        )
    else:
        page = 1
    
    start_idx = (page - 1) * entries_per_page
    end_idx = min(start_idx + entries_per_page, len(data))
    
    # Display data
    st.subheader(f"📝 JSON Data ({len(data)} total entries)")
    
    if view_mode == "Expandable cards":
        # Display as expandable cards
        for i, entry in enumerate(data[start_idx:end_idx]):
            display_json_entry(entry, start_idx + i)
    
    elif view_mode == "Full display":
        # Display all entries fully expanded
        for i, entry in enumerate(data[start_idx:end_idx]):
            st.subheader(f"📄 Entry {start_idx + i + 1}")
            st.json(entry)
            st.divider()
    
    else:  # Raw JSON
        # Display as raw JSON text
        st.subheader("Raw JSON")
        raw_json = ""
        for entry in data[start_idx:end_idx]:
            raw_json += json.dumps(entry, indent=2) + "\n\n"
        
        st.code(raw_json, language='json')
    
    # Page navigation info
    if total_pages > 1:
        st.info(f"Showing entries {start_idx + 1}-{end_idx} of {len(data)} (Page {page} of {total_pages})")
    
    # Download option
    st.subheader("💾 Download")
    
    # Convert to pretty JSON for download
    pretty_json = json.dumps(data, indent=2)
    
    st.download_button(
        label="📥 Download as JSON",
        data=pretty_json,
        file_name="data_export.json",
        mime="application/json"
    )


if __name__ == "__main__":
    main() 