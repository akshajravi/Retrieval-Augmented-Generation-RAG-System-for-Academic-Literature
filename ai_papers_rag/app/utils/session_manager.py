import streamlit as st
from typing import Dict, Any, Optional
import json
from pathlib import Path
from datetime import datetime

class SessionManager:
    def __init__(self):
        self.session_file = Path("data") / "session_data.json"
        self._initialize_session()
    
    def _initialize_session(self):
        # Initialize session state variables if they don't exist
        default_values = {
            'user_id': None,
            'session_start': datetime.now().isoformat(),
            'chat_history': [],
            'search_history': [],
            'uploaded_documents': [],
            'user_preferences': {
                'theme': 'light',
                'results_per_page': 10,
                'default_model': 'gpt-3.5-turbo',
                'temperature': 0.1
            }
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def save_session(self) -> bool:
        try:
            session_data = {
                'user_id': st.session_state.get('user_id'),
                'session_start': st.session_state.get('session_start'),
                'chat_history': st.session_state.get('chat_history', []),
                'search_history': st.session_state.get('search_history', []),
                'uploaded_documents': st.session_state.get('uploaded_documents', []),
                'user_preferences': st.session_state.get('user_preferences', {}),
                'last_updated': datetime.now().isoformat()
            }
            
            # Create directory if it doesn't exist
            self.session_file.parent.mkdir(exist_ok=True)
            
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
        
        except Exception as e:
            st.error(f"Failed to save session: {e}")
            return False
    
    def load_session(self, session_id: Optional[str] = None) -> bool:
        try:
            if not self.session_file.exists():
                return False
            
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            # Update session state with loaded data
            for key, value in session_data.items():
                if key != 'last_updated':  # Don't overwrite system keys
                    st.session_state[key] = value
            
            return True
        
        except Exception as e:
            st.error(f"Failed to load session: {e}")
            return False
    
    def clear_session(self):
        # Clear specific session data but keep user preferences
        keys_to_clear = [
            'chat_history',
            'search_history',
            'search_results'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                st.session_state[key] = []
    
    def update_preferences(self, preferences: Dict[str, Any]):
        if 'user_preferences' not in st.session_state:
            st.session_state.user_preferences = {}
        
        st.session_state.user_preferences.update(preferences)
        self.save_session()
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        return st.session_state.get('user_preferences', {}).get(key, default)
    
    def add_to_chat_history(self, message: Dict[str, Any]):
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({
            **message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 50 messages to prevent memory issues
        if len(st.session_state.chat_history) > 50:
            st.session_state.chat_history = st.session_state.chat_history[-50:]
    
    def add_to_search_history(self, search_query: str, results_count: int):
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        st.session_state.search_history.append({
            'query': search_query,
            'results_count': results_count,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 20 searches
        if len(st.session_state.search_history) > 20:
            st.session_state.search_history = st.session_state.search_history[-20:]
    
    def get_recent_searches(self, limit: int = 5) -> list:
        history = st.session_state.get('search_history', [])
        return history[-limit:] if history else []
    
    def export_session_data(self) -> Dict[str, Any]:
        return {
            'chat_history': st.session_state.get('chat_history', []),
            'search_history': st.session_state.get('search_history', []),
            'user_preferences': st.session_state.get('user_preferences', {}),
            'export_timestamp': datetime.now().isoformat()
        }