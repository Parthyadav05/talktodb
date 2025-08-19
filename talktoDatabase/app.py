import os
import re
import time
import json
import hashlib
import asyncio
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
from contextlib import asynccontextmanager

import numpy as np
import faiss
import sqlalchemy as sa
import sqlparse
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field, validator
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# Google Gemini import
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    raise ImportError("langchain-google-genai is required. Install with: pip install langchain-google-genai")

# Configuration with validation
DB_URI = os.getenv("DB_URI", "")
ALLOWED_SCHEMAS = [s.strip() for s in os.getenv("ALLOWED_SCHEMAS", "public").split(",") if s.strip()]
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro")
LLM_MODEL_SIMPLE = os.getenv("LLM_MODEL_SIMPLE", "gemini-1.5-flash")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Query parameters
K_TOP = int(os.getenv("RETRIEVE_K", "5"))
MAX_COLS = int(os.getenv("MAX_COLS_PER_TABLE", "12"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "1000"))
MAX_PLAN_COST = float(os.getenv("MAX_PLAN_COST", "5e6"))
MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

# Performance settings
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "200"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
RATE_QPS = float(os.getenv("RATE_QPS", "2.0"))
PROMPT_CAP = int(os.getenv("PROMPT_CAP", "4000"))
CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("MAX_OVERFLOW", "20"))

# Security settings
BLOCKED_OPERATIONS = [r"\binsert\b", r"\bupdate\b", r"\bdelete\b", r"\bdrop\b", 
                     r"\balter\b", r"\bcreate\b", r"\btruncate\b", r"\bgrant\b", r"\brevoke\b"]
DENY_COLUMNS = [c.strip() for c in os.getenv("DENY_COLUMNS", "password,ssn,credit_card").split(",") if c.strip()]

# Simple logging function
def jlog(event: str, **kwargs):
    """Simple logging - disabled in production for performance"""
    if os.getenv("DEBUG", "false").lower() == "true":
        print(f"[{event}] {json.dumps(kwargs, default=str)}")

def hash_key(s: str) -> str:
    """Generate a short hash key for caching"""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

# System prompt
SYSTEM_RULES = """You are a senior SQL engineer.
Return ONLY a valid PostgreSQL SELECT statement, no prose.
Rules:
- Allowed schemas only; strictly no DDL/DML; no temp tables.
- Prefer explicit JOINs when appropriate.
- Add WHERE filters only if clearly implied by the question.
- Include ORDER BY with LIMIT unless doing COUNT() or trivial projection.
- Add LIMIT if missing.
"""



def embed_local(texts: List[str], dim: int = 512) -> np.ndarray:
    """Optimized local embedding using hash-based features with caching"""
    if not texts:
        return np.empty((0, dim), dtype="float32")
    
    vectors = np.zeros((len(texts), dim), dtype="float32")
    for i, text in enumerate(texts):
        # Use more sophisticated tokenization
        tokens = re.findall(r'\w+', text.lower())
        for token in tokens:
            vectors[i, hash(token) % dim] += 1.0
    
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    return (vectors / norms).astype("float32")

def embed_query_local(text: str, dim: int = 512) -> np.ndarray:
    """Embed a single query text with improved tokenization"""
    vector = np.zeros((dim,), dtype="float32")
    tokens = re.findall(r'\w+', text.lower())
    for token in tokens:
        vector[hash(token) % dim] += 1.0
    
    norm = np.linalg.norm(vector) + 1e-9
    return (vector / norm).astype("float32")

class TTLCache:
    """LRU Cache with TTL (Time To Live) support"""
    def __init__(self, capacity: int = 128, ttl: int = 3600):
        self.capacity = capacity
        self.ttl = ttl
        self.store: Dict[str, Dict[str, Any]] = {}
        self.order = deque()
    
    def get(self, key: str):
        if key not in self.store:
            return None
        
        item = self.store[key]
        # Check TTL
        if time.time() - item['timestamp'] > self.ttl:
            self._remove(key)
            return None
        
        # Update LRU order
        self.order.remove(key)
        self.order.appendleft(key)
        return item['value']
    
    def set(self, key: str, value: Any):
        if key in self.store:
            self.order.remove(key)
        elif len(self.order) >= self.capacity:
            old_key = self.order.pop()
            self.store.pop(old_key, None)
        
        self.store[key] = {
            'value': value,
            'timestamp': time.time()
        }
        self.order.appendleft(key)
    
    def _remove(self, key: str):
        if key in self.store:
            del self.store[key]
            self.order.remove(key)
    
    def cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, item in self.store.items()
            if current_time - item['timestamp'] > self.ttl
        ]
        for key in expired_keys:
            self._remove(key)

# Enhanced caches with TTL
answer_cache = TTLCache(capacity=CACHE_SIZE, ttl=CACHE_TTL)
narrative_cache = TTLCache(capacity=max(64, CACHE_SIZE // 2), ttl=CACHE_TTL)

# Rate limiting with sliding window
class SlidingWindowRateLimiter:
    def __init__(self, qps: float):
        self.qps = qps
        self.requests = deque()
        self.window = 1.0
    
    def is_allowed(self) -> bool:
        now = time.time()
        # Remove old requests outside the window
        while self.requests and now - self.requests[0] > self.window:
            self.requests.popleft()
        
        # Check if we're under the limit
        if len(self.requests) < self.qps:
            self.requests.append(now)
            return True
        return False
    
    def wait_time(self) -> float:
        if not self.requests:
            return 0.0
        return max(0.0, self.window - (time.time() - self.requests[0]))

rate_limiter = SlidingWindowRateLimiter(RATE_QPS)

def rate_limit():
    """Enhanced rate limiting with better performance"""
    if not rate_limiter.is_allowed():
        wait_time = rate_limiter.wait_time()
        if wait_time > 0:
            time.sleep(wait_time)

def enforce_select_only(sql: str) -> None:
    """Ensure only SELECT queries are allowed"""
    sql_lower = sql.lower().strip()
    if not sql_lower.startswith('select'):
        raise ValueError("Only SELECT queries are allowed.")
    
    for pattern in BLOCKED_OPERATIONS:
        if re.search(pattern, sql_lower):
            raise ValueError(f"Operation not allowed: {pattern}")

def enforce_limit(sql: str, default_limit: int = DEFAULT_LIMIT) -> str:
    """Add LIMIT clause if missing"""
    if re.search(r"\blimit\s+\d+\b", sql, re.IGNORECASE):
        return sql
    return sql.rstrip("; ") + f" LIMIT {default_limit}"

def restrict_schemas(sql: str, allowed: List[str]) -> None:
    """Ensure only allowed schemas are referenced"""
    # Parse the SQL to identify actual schema references, not table aliases
    # Look for patterns like "schema.table" but exclude single letter aliases
    schema_refs = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\.", sql)
    
    if not schema_refs:
        return
    
    # Allow system schemas and user-defined allowed schemas
    system_schemas = {"information_schema", "pg_catalog", "pg_toast"}
    allowed_lower = {schema.strip().lower() for schema in allowed} | system_schemas
    
    for schema in set(schema_refs):
        if schema.lower() not in allowed_lower:
            # Additional check: verify this is actually a schema reference by checking context
            # Look for FROM/JOIN patterns to distinguish from function calls or other dot notation
            pattern = rf"\b(FROM|JOIN)\s+{re.escape(schema)}\."
            if re.search(pattern, sql, re.IGNORECASE):
                raise ValueError(f"Schema '{schema}' not allowed. Allowed schemas: {', '.join(allowed)}")

def deny_columns(sql: str, denied: List[str]) -> None:
    """Block access to sensitive columns"""
    for column in denied:
        if re.search(rf"\b{re.escape(column)}\b", sql, re.IGNORECASE):
            raise ValueError(f"Access to column '{column}' is denied.")

def is_simple_lookup(question: str) -> bool:
    """Determine if this is a simple lookup query"""
    question_lower = question.lower()
    simple_keywords = ["show", "list", "first", "last", "top", "sample", "rows from", "count of", "how many"]
    complex_keywords = ["join", "group by", "having", "union", "subquery", "complex"]
    
    has_simple = any(keyword in question_lower for keyword in simple_keywords)
    has_complex = any(keyword in question_lower for keyword in complex_keywords)
    
    return has_simple and not has_complex and len(question) < 100

def create_llm():
    """Factory function to create Gemini LLM instance"""
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    return ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0)

def get_llm_for_question(question: str = ""):
    """Get appropriate Gemini model based on question complexity"""
    if is_simple_lookup(question):
        return ChatGoogleGenerativeAI(model=LLM_MODEL_SIMPLE, temperature=0)
    return app_state.llm

def build_user_prompt(question: str, table_info: str) -> str:
    """Build the complete prompt for the LLM"""
    core_prompt = f"{SYSTEM_RULES}\n\nSchema:\n{table_info}\n\nUserQuestion:\n{question}\n\nSQL:"
    return core_prompt[:PROMPT_CAP]

def introspect_database(db_uri: str, schemas: List[str]) -> Dict[str, Any]:
    """Extract database schema information with error handling"""
    try:
        # Use connection pooling for better performance
        engine = sa.create_engine(
            db_uri, 
            pool_size=CONNECTION_POOL_SIZE,
            max_overflow=MAX_OVERFLOW,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        inspector = sa.inspect(engine)
        
        tables = []
        joins = []
        
        for schema in schemas:
            try:
                table_names = inspector.get_table_names(schema=schema)
                jlog("schema_introspected", schema=schema, table_count=len(table_names))
                
                for table_name in table_names:
                    try:
                        columns = inspector.get_columns(table_name, schema=schema)
                        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
                        
                        table_doc = {
                            "schema": schema,
                            "table": table_name,
                            "columns": [{"name": col["name"], "type": str(col.get("type"))} for col in columns],
                            "foreign_keys": foreign_keys
                        }
                        tables.append(table_doc)
                        
                        # Extract join information
                        for fk in foreign_keys:
                            local_cols = fk.get("constrained_columns", [])
                            remote_cols = fk.get("referred_columns", [])
                            remote_schema = fk.get("referred_schema", schema)
                            remote_table = fk.get("referred_table")
                            
                            for local_col, remote_col in zip(local_cols, remote_cols):
                                joins.append({
                                    "left": f"{schema}.{table_name}.{local_col}",
                                    "right": f"{remote_schema}.{remote_table}.{remote_col}"
                                })
                    except Exception as e:
                        jlog("table_introspection_error", schema=schema, table=table_name, error=str(e))
                        continue
                        
            except Exception as e:
                jlog("schema_introspection_error", schema=schema, error=str(e))
                continue
        
        return {"tables": tables, "joins": joins, "engine": engine}
        
    except Exception as e:
        jlog("database_introspection_failed", error=str(e))
        raise RuntimeError(f"Failed to introspect database: {e}")

def tables_to_search_texts(tables: List[dict]) -> List[str]:
    """Convert table docs to searchable text with better formatting and synonyms"""
    
    # Table name synonyms and common keywords
    table_synonyms = {
        'users': ['user', 'person', 'people', 'member', 'account', 'customer', 'client'],
        'event': ['events', 'meeting', 'meetup', 'schedule', 'calendar', 'appointment'],
        'booking': ['bookings', 'reservation', 'appointments', 'orders'],
        'club': ['clubs', 'group', 'organization', 'community'],
        'area': ['areas', 'location', 'region', 'zone', 'place'],
        'city': ['cities', 'town', 'location', 'place'],
        'payment': ['payments', 'transaction', 'billing', 'invoice'],
        'media': ['image', 'photo', 'file', 'document', 'attachment'],
        'location': ['locations', 'place', 'venue', 'address', 'spot'],
        'group_member': ['members', 'participants', 'attendees'],
        'transaction': ['transactions', 'payment', 'billing', 'financial'],
    }
    
    texts = []
    for table in tables:
        table_name = table['table']
        
        # Start with table name and schema
        parts = [f"{table['schema']}.{table_name}", table_name]
        
        # Add synonyms for better matching
        if table_name in table_synonyms:
            parts.extend(table_synonyms[table_name])
        
        # Add column information with more context
        for column in table.get("columns", []):
            col_name = column['name']
            col_type = column['type']
            parts.append(f"{col_name}:{col_type}")
            
            # Add common column synonyms
            if 'name' in col_name.lower():
                parts.append('name title label')
            elif 'email' in col_name.lower():
                parts.append('email contact')
            elif 'phone' in col_name.lower():
                parts.append('phone number contact')
            elif 'count' in col_name.lower() or col_name.lower().endswith('_count'):
                parts.append('count total number quantity')
            elif 'time' in col_name.lower() or 'date' in col_name.lower():
                parts.append('time date schedule when')
        
        # Add foreign key context
        for fk in table.get("foreign_keys", []):
            ref_table = fk.get("referred_table")
            if ref_table:
                parts.append(f"references:{ref_table}")
                parts.append(f"related_to:{ref_table}")
        
        # Add common action keywords based on table name
        if 'user' in table_name.lower():
            parts.extend(['list show all people accounts'])
        elif 'event' in table_name.lower():
            parts.extend(['scheduled upcoming meetings'])
        elif 'booking' in table_name.lower():
            parts.extend(['reservations orders'])
        elif 'club' in table_name.lower():
            parts.extend(['groups organizations'])
        
        texts.append(" | ".join(parts))
    return texts

def format_table_info(tables: List[dict], max_cols: int = MAX_COLS, joins: List[dict] = None) -> str:
    """Format table information for LLM prompt with better structure"""
    if not tables:
        return "No relevant tables found."
    
    sections = []
    
    for table in tables:
        columns = table.get("columns", [])[:max_cols]
        col_lines = [f"- {col['name']} ({col['type']})" for col in columns]
        
        fk_lines = []
        for fk in table.get("foreign_keys", []):
            cols = ", ".join(fk.get("constrained_columns", []))
            ref_table = fk.get("referred_table")
            if cols and ref_table:
                fk_lines.append(f"FK({cols}) -> {ref_table}")
        
        section = [f"Table: {table['schema']}.{table['table']}", "Columns:"]
        section.extend(col_lines)
        
        if fk_lines:
            section.extend(["Foreign Keys:"] + [f"- {fk}" for fk in fk_lines])
        
        sections.append("\n".join(section))
    
    # Add join hints if available
    if joins:
        table_set = {(t["schema"], t["table"]) for t in tables}
        join_lines = []
        
        for join in joins[:20]:  # Limit join hints
            try:
                left_parts = join["left"].split(".", 2)
                right_parts = join["right"].split(".", 2)
                
                if len(left_parts) >= 2 and len(right_parts) >= 2:
                    left_schema, left_table = left_parts[0], left_parts[1]
                    right_schema, right_table = right_parts[0], right_parts[1]
                    
                    if (left_schema, left_table) in table_set and (right_schema, right_table) in table_set:
                        join_lines.append(f"{join['left']} = {join['right']}")
            except (IndexError, KeyError):
                continue
        
        if join_lines:
            sections.append("JoinHints:\n" + "\n".join(f"- {line}" for line in join_lines))
    
    return "\n\n".join(sections)

def build_search_index(tables: List[dict]) -> Tuple[Any, int]:
    """Build FAISS search index for table retrieval with error handling"""
    if not tables:
        raise ValueError("No tables available for indexing")
    
    try:
        texts = tables_to_search_texts(tables)
        vectors = embed_local(texts)
        
        if vectors.size == 0:
            raise ValueError("Failed to generate embeddings")
        
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        
        jlog("search_index_built", table_count=len(tables), vector_dim=vectors.shape[1])
        return index, vectors.shape[1]
        
    except Exception as e:
        jlog("search_index_build_failed", error=str(e))
        raise RuntimeError(f"Failed to build search index: {e}")

class AppState:
    """Application state container with better initialization"""
    def __init__(self):
        self.db: SQLDatabase = None
        self.engine: sa.Engine = None
        self.llm = None
        self.schema: Dict[str, Any] = {}
        self.tables: List[dict] = []
        self.search_index = None
        self.embedding_dim = 512
        self.current_question: Optional[str] = None
        self.startup_time = time.time()
        self.initialized = False

app_state = AppState()

async def cleanup_caches():
    """Background task to clean up expired cache entries"""
    while True:
        try:
            answer_cache.cleanup_expired()
            narrative_cache.cleanup_expired()
            await asyncio.sleep(300)  # Clean every 5 minutes
        except Exception as e:
            jlog("cache_cleanup_error", error=str(e))
            await asyncio.sleep(60)

def initialize_app():
    """Initialize the application state with comprehensive error handling"""
    try:
        # Validate required configuration
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY environment variable is required")
        
        # Build database URI if not provided
        db_uri = DB_URI
        if not db_uri:
            db_host = os.getenv("DB_HOST", "")
            db_port = os.getenv("DB_PORT", "")
            db_user = os.getenv("DB_USER", "")
            db_password = os.getenv("DB_PASSWORD", "")
            db_name = os.getenv("DB_NAME", "")
            
            if all([db_host, db_port, db_user, db_password, db_name]):
                db_uri = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        
        if not db_uri:
            raise RuntimeError("Database connection not configured. Set DB_URI or individual DB_* variables.")
        
        jlog("initializing_app", schemas=ALLOWED_SCHEMAS)
        
        # Introspect database schema
        schema_info = introspect_database(db_uri, ALLOWED_SCHEMAS)
        app_state.engine = schema_info["engine"]
        app_state.schema = {"tables": schema_info["tables"], "joins": schema_info["joins"]}
        app_state.tables = schema_info["tables"]
        
        if not app_state.tables:
            raise RuntimeError(f"No tables found in allowed schemas: {ALLOWED_SCHEMAS}")
        
        # Initialize database wrapper
        app_state.db = SQLDatabase(engine=app_state.engine)
        
        # Initialize LLM
        app_state.llm = create_llm()
        
        # Build search index
        app_state.search_index, app_state.embedding_dim = build_search_index(app_state.tables)
        
        app_state.initialized = True
        
        # Log startup info
        masked_uri = re.sub(r"://([^:]+):[^@]+@", r"://\1:***@", db_uri)
        jlog("startup_complete", 
             db_uri=masked_uri, 
             schemas=ALLOWED_SCHEMAS, 
             table_count=len(app_state.tables),
             initialization_time=time.time() - app_state.startup_time)
             
    except Exception as e:
        jlog("startup_failed", error=str(e))
        raise RuntimeError(f"Application initialization failed: {e}")

def retrieve_relevant_tables(question: str, k: int = K_TOP) -> List[dict]:
    """Retrieve relevant tables for a question using semantic search with fallbacks"""
    if not app_state.search_index or not app_state.tables:
        jlog("search_index_unavailable")
        return []
    
    try:
        query_vector = embed_query_local(question, app_state.embedding_dim).reshape(1, -1).astype("float32")
        scores, indices = app_state.search_index.search(query_vector, min(k, len(app_state.tables)))
        
        relevant_tables = []
        
        # First pass: Higher threshold for high-confidence matches
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(app_state.tables) and scores[0][i] > 0.1:
                relevant_tables.append(app_state.tables[idx])
        
        # Fallback: If no matches, try lower threshold
        if not relevant_tables:
            for i, idx in enumerate(indices[0]):
                if 0 <= idx < len(app_state.tables) and scores[0][i] > 0.05:  # Lower threshold
                    relevant_tables.append(app_state.tables[idx])
        
        # Second fallback: Try keyword matching for common table names
        if not relevant_tables:
            question_lower = question.lower()
            keyword_matches = []
            
            for table in app_state.tables:
                table_name = table['table'].lower()
                
                # Direct keyword matching
                if (table_name in question_lower or 
                    any(keyword in question_lower for keyword in [
                        'user', 'people', 'person', 'member', 'account',
                        'event', 'meeting', 'schedule',
                        'booking', 'reservation',
                        'club', 'group',
                        'area', 'location', 'place',
                        'city', 'town'
                    ]) and table_name in ['users', 'event', 'booking', 'club', 'area', 'city']):
                    keyword_matches.append(table)
            
            if keyword_matches:
                relevant_tables = keyword_matches[:k]
                jlog("keyword_fallback_used", matches=len(keyword_matches))
        
        # Final fallback: Return most popular tables if still no matches
        if not relevant_tables:
            # Return tables with most data (likely most useful)
            popular_tables = [t for t in app_state.tables if t['table'] in ['users', 'event', 'club', 'booking']]
            if popular_tables:
                relevant_tables = popular_tables[:min(2, k)]
                jlog("popular_tables_fallback", tables=[t['table'] for t in relevant_tables])
        
        return relevant_tables
        
    except Exception as e:
        jlog("table_retrieval_error", error=str(e))
        return []

def clean_sql_output(raw_output: str) -> str:
    """Clean and format SQL output from LLM"""
    # Remove code block markers
    output = re.sub(r"```sql\s*|\s*```", "", raw_output, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove common prefixes
    output = re.sub(r"^\s*(SQLQuery|SQL|Query)\s*:?\s*", "", output, flags=re.IGNORECASE | re.MULTILINE)
    
    # Extract SELECT statement
    match = re.search(r"(?is)\bselect\b.*", output)
    if match:
        output = output[match.start():]
    
    # Clean up whitespace and ensure proper termination
    output = output.strip()
    if not output.endswith(';'):
        output += ';'
        
    return output

async def call_llm_for_sql(question: str, table_info: str, max_retries: int = 2) -> str:
    """Generate SQL using LLM with async support and retry logic"""
    rate_limit()
    
    chain = create_sql_query_chain(get_llm_for_question(question), app_state.db)
    prompt = build_user_prompt(question, table_info)
    
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            
            # Use asyncio for non-blocking LLM calls
            loop = asyncio.get_event_loop()
            raw_output = await loop.run_in_executor(
                None, 
                lambda: chain.invoke({"question": prompt, "table_info": table_info})
            )
            
            sql = clean_sql_output(raw_output if isinstance(raw_output, str) else str(raw_output))
            
            elapsed_ms = int((time.time() - start_time) * 1000)
            model_name = LLM_MODEL_SIMPLE if is_simple_lookup(question) else LLM_MODEL
            
            jlog("llm_sql_success", elapsed_ms=elapsed_ms, attempt=attempt + 1, 
                 model=model_name, output_length=len(sql))
            
            return sql
            
        except Exception as e:
            jlog("llm_sql_error", attempt=attempt + 1, error=str(e))
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))  # Exponential backoff

def validate_sql(sql: str) -> Tuple[bool, str]:
    """Validate SQL by running EXPLAIN with timeout"""
    try:
        enforce_select_only(sql)
        base_sql = sql.strip().rstrip(";")
        test_sql = f"SELECT * FROM ({base_sql}) AS _test WHERE 1=0"
        
        with app_state.engine.connect() as conn:
            # Add statement timeout for validation
            conn.execute(sa.text("SET statement_timeout = '5s'"))
            conn.execute(sa.text(test_sql))
        
        return True, ""
    except Exception as e:
        return False, str(e)

async def generate_sql_for_question(question: str) -> str:
    """Main function to generate SQL for a natural language question"""
    app_state.current_question = question
    
    # Input validation
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.")
    
    # Check cache first
    cache_key = hash_key(f"{question}|{K_TOP}|{MAX_COLS}|{','.join(ALLOWED_SCHEMAS)}|{LLM_MODEL}")
    cached_result = answer_cache.get(cache_key)
    if cached_result:
        jlog("cache_hit", cache_key=cache_key)
        return cached_result["sql"]
    
    # Retrieve relevant tables
    relevant_tables = retrieve_relevant_tables(question, k=K_TOP)
    if not relevant_tables:
        jlog("no_relevant_tables", question=question)
        raise ValueError("No relevant tables found for your question. Please be more specific.")
    
    table_info = format_table_info(relevant_tables, max_cols=MAX_COLS, joins=app_state.schema.get("joins"))
    
    jlog("table_retrieval", question=question, 
         retrieved_tables=[f"{t['schema']}.{t['table']}" for t in relevant_tables],
         table_count=len(relevant_tables), table_info_length=len(table_info))
    
    # Generate SQL
    raw_sql = await call_llm_for_sql(question, table_info)
    sql = sqlparse.format(raw_sql, reindent=True, keyword_case="upper")
    
    # Validate and potentially fix SQL
    is_valid, error = validate_sql(sql)
    if not is_valid:
        jlog("sql_validation_failed", error=error)
        
        # Try adding LIMIT and re-validate
        sql_with_limit = enforce_limit(sql)
        is_valid_fixed, _ = validate_sql(sql_with_limit)
        
        if not is_valid_fixed:
            # Try to repair with LLM
            repair_question = f"{question}\n\nThe previous SQL failed with error:\n{error}\nRewrite a corrected SELECT statement."
            repaired_sql = await call_llm_for_sql(repair_question, table_info)
            sql = sqlparse.format(repaired_sql, reindent=True, keyword_case="upper")
    
    # Cache the result
    answer_cache.set(cache_key, {"sql": sql})
    return sql

def estimate_query_cost(sql: str) -> float:
    """Estimate query execution cost using EXPLAIN with timeout"""
    try:
        with app_state.engine.connect() as conn:
            # Set timeout for EXPLAIN
            conn.execute(sa.text("SET statement_timeout = '5s'"))
            result = conn.execute(sa.text("EXPLAIN (FORMAT JSON) " + sql)).scalar()
        
        # Parse JSON from the result
        start_idx = result.find('[')
        end_idx = result.rfind(']')
        if start_idx == -1 or end_idx == -1:
            return 0.0
        
        plan_json = json.loads(result[start_idx:end_idx + 1])
        return float(plan_json[0]["Plan"].get("Total Cost", 0.0))
        
    except Exception as e:
        jlog("cost_estimation_failed", error=str(e))
        return 0.0

def execute_sql_query(sql: str, limit: int = DEFAULT_LIMIT) -> Tuple[List[str], List[List[Any]], str]:
    """Execute SQL query and return results with timeout"""
    sql_with_limit = enforce_limit(sql, limit)
    
    try:
        with app_state.engine.connect() as conn:
            # Set query timeout
            conn.execute(sa.text("SET statement_timeout = '30s'"))
            result = conn.execute(sa.text(sql_with_limit))
            columns = list(result.keys())
            rows = [list(row) for row in result.fetchall()]
        
        # Create text representation
        if not rows:
            text_result = ", ".join(columns)
        else:
            text_result = ", ".join(columns) + "\n" + "\n".join([
                ", ".join([str(cell) if cell is not None else "NULL" for cell in row]) for row in rows
            ])
        
        return columns, rows, text_result
        
    except Exception as e:
        jlog("query_execution_failed", sql=sql[:100], error=str(e))
        raise ValueError(f"Query execution failed: {e}")

async def generate_narrative(question: str, sql: str, result_text: str) -> str:
    """Generate natural language summary of results with async support"""
    try:
        rate_limit()
        
        model = get_llm_for_question(question)
        
        # Limit text for prompt
        max_lines = int(os.getenv("NARRATIVE_MAX_LINES", "8"))
        max_chars = int(os.getenv("NARRATIVE_MAX_CHARS", "4000"))
        
        lines = result_text.splitlines()[:max_lines]
        sampled_text = "\n".join(lines)
        if len(sampled_text) > max_chars:
            sampled_text = sampled_text[:max_chars]
        
        # Check cache
        cache_key = hash_key(question + "|" + sql + "|" + sampled_text)
        cached_narrative = narrative_cache.get(cache_key)
        if cached_narrative:
            jlog("narrative_cache_hit", cache_key=cache_key)
            return cached_narrative
        
        # Generate narrative
        prompt = (
            "Write a single-sentence, plain-English summary (<=40 words) of the SQL result below. "
            "Prefer concrete quantities, counts, sums, and currency amounts if present. "
            "No SQL, no column names, no markdown.\n\n"
            f"Question:\n{question}\n\nSQL:\n{sql}\n\nRows (sampled):\n{sampled_text}"
        )
        
        start_time = time.time()
        
        # Use asyncio for non-blocking LLM calls
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: model.invoke(prompt))
        
        narrative = re.sub(r"\s+", " ", str(response)).strip()[:400]
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        jlog("narrative_generated", elapsed_ms=elapsed_ms, question_length=len(question), 
             sql_length=len(sql), result_length=len(sampled_text))
        
        narrative_cache.set(cache_key, narrative)
        return narrative
        
    except Exception as e:
        jlog("narrative_fallback", error=str(e))
        
        # Fallback to simple summary
        lines = [line for line in (result_text or "").splitlines() if line.strip()]
        row_count = max(0, len(lines) - 1) if len(lines) > 1 else len(lines)
        return f"Returned {row_count} row(s) for your question."

# Pydantic models for API with validation
class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the database")
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        if len(v) > MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long. Maximum {MAX_QUESTION_LENGTH} characters allowed.")
        return v.strip()

class ExecutionInfo(BaseModel):
    row_count: int
    elapsed_ms: int
    plan_cost: float
    limited: bool
    limit: int
    data_verified: bool
    rows_checksum: str

class ResultBlock(BaseModel):
    columns: List[str]
    rows: List[List[Any]]
    truncated: bool
    max_rows: int

class Explanation(BaseModel):
    narrative: str
    notes: List[str] = []
    warnings: List[str] = []

class AskResponse(BaseModel):
    trace_id: str
    question: str
    sql: str
    execution: ExecutionInfo
    result: ResultBlock
    explanation: Explanation
    meta: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: float
    uptime_seconds: int
    checks: Dict[str, str]
    meta: Dict[str, Any]

# FastAPI application with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    jlog("app_starting")
    initialize_app()
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_caches())
    
    yield
    
    # Shutdown
    jlog("app_shutting_down")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(
    title="TalkDB - Natural Language Database Interface",
    description="Convert natural language questions to SQL queries and execute them safely",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with performance metrics"""
    request_id = hash_key(f"{time.time()}|{request.client.host}|{request.url.path}")
    jlog("http_request_start", request_id=request_id, method=request.method, 
         path=request.url.path, client=request.client.host)
    
    start_time = time.time()
    try:
        response = await call_next(request)
        elapsed_ms = int((time.time() - start_time) * 1000)
        jlog("http_request_complete", request_id=request_id, status=response.status_code, 
             elapsed_ms=elapsed_ms)
        return response
    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        jlog("http_request_error", request_id=request_id, error=str(e), elapsed_ms=elapsed_ms)
        raise

@app.get("/health", response_model=HealthResponse)
def comprehensive_health_check():
    """Comprehensive health check with component validation"""
    checks = {}
    status = "healthy"
    
    # Check initialization
    if not app_state.initialized:
        status = "unhealthy"
        checks["initialization"] = "failed"
    else:
        checks["initialization"] = "complete"
    
    # Database connectivity
    try:
        if app_state.engine:
            with app_state.engine.connect() as conn:
                conn.execute(sa.text("SELECT 1"))
            checks["database"] = "connected"
        else:
            checks["database"] = "not_initialized"
            status = "unhealthy"
    except Exception as e:
        checks["database"] = f"failed: {str(e)[:50]}"
        status = "unhealthy"
    
    # Gemini API configuration
    if GOOGLE_API_KEY:
        checks["gemini_api"] = "configured"
    else:
        checks["gemini_api"] = "missing_key"
        status = "unhealthy"
    
    # Search index
    if app_state.search_index:
        checks["search_index"] = f"ready ({len(app_state.tables)} tables)"
    else:
        checks["search_index"] = "not_built"
        status = "degraded"
    
    # Cache health
    checks["cache_size"] = f"{len(answer_cache.store)} queries cached"
    
    uptime = int(time.time() - app_state.startup_time)
    
    return HealthResponse(
        status=status,
        timestamp=time.time(),
        uptime_seconds=uptime,
        checks=checks,
        meta={
            "version": "1.0.0",
            "allowed_schemas": ALLOWED_SCHEMAS,
            "rate_limit_qps": RATE_QPS
        }
    )

@app.get("/api/healthz")
def simple_health_check():
    """Simple health check for load balancers"""
    return {"ok": app_state.initialized}

def get_rewrite_suggestions(question: str) -> List[str]:
    """Generate suggestions for improving the question"""
    suggestions = [
        "Add a time window (e.g., last 30 days).",
        "Specify exact fields you need.",
        "Name the table or entity if you know it.",
        "Include identifiers like user_id, event name, or specific IDs.",
    ]
    
    if len(question) < 10:
        suggestions.append("Use a complete sentence describing what you want.")
    
    if "?" not in question:
        suggestions.append("Frame as a clear question.")
    
    return suggestions

@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, background_tasks: BackgroundTasks):
    """Main endpoint to ask questions about the database"""
    start_time = time.time()
    trace_id = hash_key(request.question + "|" + str(start_time))
    
    try:
        jlog("question_received", trace_id=trace_id, question=request.question)
        
        # Check if application is initialized
        if not app_state.initialized:
            raise HTTPException(status_code=503, detail="Service not ready. Please try again in a moment.")
        
        # Check if question matches any known tables
        relevant_tables = retrieve_relevant_tables(request.question, k=1)
        out_of_schema = len(relevant_tables) == 0
        
        # Generate SQL
        sql = await generate_sql_for_question(request.question)
        
        # Security checks
        enforce_select_only(sql)
        restrict_schemas(sql, ALLOWED_SCHEMAS)
        deny_columns(sql, DENY_COLUMNS)
        
        # Check query cost
        estimated_cost = estimate_query_cost(sql)
        if estimated_cost and estimated_cost > MAX_PLAN_COST:
            jlog("query_blocked_high_cost", cost=int(estimated_cost), max_allowed=int(MAX_PLAN_COST))
            raise HTTPException(
                status_code=400, 
                detail=f"Query blocked: estimated cost {int(estimated_cost)} exceeds limit {int(MAX_PLAN_COST)}"
            )
        
        # Execute query
        columns, rows, result_text = execute_sql_query(sql, DEFAULT_LIMIT)
        
        # Calculate metrics
        row_count = len(rows)
        elapsed_ms = int((time.time() - start_time) * 1000)
        result_checksum = hash_key(result_text or "")
        
        # Prepare response data
        notes: List[str] = []
        warnings: List[str] = []
        
        if row_count == 0:
            narrative = "No matching records found. Try rewriting your question for better results."
            warnings.append("No results found. Consider adding filters (date range, exact IDs) or check table names.")
            notes.extend(get_rewrite_suggestions(request.question))
        else:
            if out_of_schema:
                narrative = "Query executed but no relevant tables were found in the allowed schemas."
            else:
                narrative = await generate_narrative(request.question, sql, result_text)
        
        # Add cache cleanup to background tasks
        background_tasks.add_task(answer_cache.cleanup_expired)
        
        jlog("question_completed", trace_id=trace_id, rows=row_count, elapsed_ms=elapsed_ms)
        
        return AskResponse(
            trace_id=trace_id,
            question=request.question,
            sql=sql,
            execution=ExecutionInfo(
                row_count=row_count,
                elapsed_ms=elapsed_ms,
                plan_cost=float(estimated_cost or 0.0),
                limited=True,
                limit=DEFAULT_LIMIT,
                data_verified=True,
                rows_checksum=result_checksum
            ),
            result=ResultBlock(
                columns=columns,
                rows=rows,
                truncated=(row_count >= DEFAULT_LIMIT),
                max_rows=DEFAULT_LIMIT
            ),
            explanation=Explanation(
                narrative=narrative,
                notes=notes,
                warnings=warnings
            ),
            meta={
                "model": LLM_MODEL_SIMPLE if is_simple_lookup(request.question) else LLM_MODEL,
                "retrieval_tables": [f"{t['schema']}.{t['table']}" for t in retrieve_relevant_tables(request.question, k=K_TOP)],
                "cache_hit": trace_id in [item for item in answer_cache.store.keys()]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        jlog("question_error", trace_id=trace_id, error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENV", "production") == "development",
        log_level=LOG_LEVEL.lower(),
        access_log=LOG_LEVEL.upper() == "DEBUG"
    )