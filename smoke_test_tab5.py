import sys
import os
import pandas as pd

# Set PYTHONPATH
sys.path.append(os.getcwd())

try:
    from dashboard.tab_contexts import _build_tab5_context

    # 1. Load CSV
    data_path = 'data/youtube_vn_music_cleaned.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"CSV not found: {data_path}")
    
    df = pd.read_csv(data_path)

    # 2. Build Tab 5 context
    payload = _build_tab5_context(df)

    # 3. Check glossary
    glossary = payload.get('glossary', {})
    if not isinstance(glossary, dict) or not glossary:
        raise ValueError("Glossary dict missing or empty in payload")
    print(f"✓ Glossary entries found: {len(glossary)}")

    # 4. Check how_to_read
    how_to_read = payload.get('how_to_read', {})
    if not isinstance(how_to_read, dict) or not how_to_read:
        raise ValueError("How to read sections missing or empty in payload")
    print(f"✓ How to read sections found: {len(how_to_read)}")

    print("✓ Tab 5 context built successfully")

except Exception as e:
    print(f"ERROR: {str(e)}")
    sys.exit(1)
