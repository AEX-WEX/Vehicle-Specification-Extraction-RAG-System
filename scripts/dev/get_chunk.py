#!/usr/bin/env python3
"""
Quick utility to map chunk_id to chunk text
"""

import pickle
import sys
from pathlib import Path

def get_chunk_text(chunk_id: str, index_dir: str = "index") -> str:
   
    
    # Load pickled chunks
    pkl_path = Path(index_dir) / "vehicle_specs.pkl"
    
    if not pkl_path.exists():
        return f"❌ Index not found: {pkl_path}"
    
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            chunks = data['chunks']
        
        # Find chunk by ID
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return f"""✅ Found: {chunk_id}
📄 Page: {chunk.page_number}
📍 Position: chars {chunk.start_char}-{chunk.end_char}
📝 Text:
─────────────────────────────────────
{chunk.text}
─────────────────────────────────────"""
        
        return f"❌ Chunk not found: {chunk_id}"
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python get_chunk.py <chunk_id> [index_dir]")
        print("Example: python get_chunk.py chunk_00854")
        sys.exit(1)
    
    chunk_id = sys.argv[1]
    index_dir = sys.argv[2] if len(sys.argv) > 2 else "index"
    
    result = get_chunk_text(chunk_id, index_dir)
    print(result)
