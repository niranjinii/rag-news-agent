"""
Checkpointing and State Persistence Module

Provides automatic state saving and resumption capabilities using LangGraph's
built-in checkpointing system. Supports multiple backends (SQLite, Postgres, Memory).

Key Features:
- Automatic checkpoint creation after each node
- Resume from exact failure point
- Checkpoint inspection and management
- Cleanup policies for old checkpoints
- Production-ready with multiple backend support
"""

from typing import Optional, List, Tuple
import os
import hashlib
from datetime import datetime, timedelta


class CheckpointManager:
    """
    Manages checkpointing backends for the article writer pipeline.
    
    This class provides:
    - Automatic checkpoint creation after each agent execution
    - Resume from failure point functionality
    - Checkpoint inspection and listing
    - Cleanup of old checkpoints
    - Multiple backend support (SQLite, Postgres, Memory)
    
    Example:
        manager = CheckpointManager(backend="memory")
        checkpointer = manager.get_checkpointer()
        
        # Use in graph compilation
        app = workflow.compile(checkpointer=checkpointer)
    """
    
    def __init__(
        self, 
        backend: str = "memory",  # Changed default to memory for compatibility
        db_path: str = "checkpoints.db",
        postgres_uri: Optional[str] = None
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            backend: Backend type ("sqlite", "postgres", or "memory")
            db_path: Path to SQLite database (used if backend="sqlite")
            postgres_uri: PostgreSQL connection URI (used if backend="postgres")
        """
        self.backend = backend
        self.db_path = db_path
        self.postgres_uri = postgres_uri or os.getenv("POSTGRES_URI")
        self.checkpointer = self._create_checkpointer()
        
        print(f"[CHECKPOINT] Manager initialized with backend: {backend}")
    
    def _create_checkpointer(self):
        """Create appropriate checkpointer based on backend configuration"""
        
        if self.backend == "sqlite":
            print(f"[CHECKPOINT] Attempting SQLite backend: {self.db_path}")
            
            try:
                # Try the newer import path first (LangGraph 0.2+)
                from langgraph.checkpoint.sqlite import SqliteSaver
                import sqlite3
                
                # Create connection with check_same_thread=False for multi-threading
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                checkpointer = SqliteSaver(conn)
                print(f"[CHECKPOINT] ✓ Using SqliteSaver (thread-safe mode)")
                return checkpointer
            except (ImportError, AttributeError, TypeError) as e:
                print(f"[CHECKPOINT] SqliteSaver init failed: {e}")
                
                try:
                    # Try from_conn_string method
                    from langgraph.checkpoint.sqlite import SqliteSaver
                    checkpointer = SqliteSaver.from_conn_string(self.db_path)
                    print(f"[CHECKPOINT] ✓ Using SqliteSaver (from_conn_string)")
                    return checkpointer
                except (ImportError, AttributeError) as e2:
                    print(f"[CHECKPOINT] from_conn_string failed: {e2}")
                    
                    print(f"[CHECKPOINT] ⚠ Falling back to MemorySaver (not persistent)")
                    from langgraph.checkpoint.memory import MemorySaver
                    return MemorySaver()
        
        elif self.backend == "postgres":
            if not self.postgres_uri:
                raise ValueError(
                    "PostgreSQL URI required for postgres backend. "
                    "Set POSTGRES_URI environment variable or pass postgres_uri parameter."
                )
            print(f"[CHECKPOINT] Using PostgreSQL backend")
            
            try:
                from langgraph.checkpoint.postgres import PostgresSaver
                return PostgresSaver.from_conn_string(self.postgres_uri)
            except ImportError as e:
                raise ImportError(
                    f"PostgreSQL checkpoint backend not available: {e}. "
                    "Install with: pip install langgraph[postgres]"
                )
        
        elif self.backend == "memory":
            print("[CHECKPOINT] Using MemorySaver backend (NOT PERSISTENT)")
            from langgraph.checkpoint.memory import MemorySaver
            return MemorySaver()
        
        else:
            raise ValueError(
                f"Unknown backend: {self.backend}. "
                f"Supported backends: sqlite, postgres, memory"
            )
    
    def get_checkpointer(self):
        """
        Returns the checkpointer instance for use in graph compilation.
        
        Returns:
            Checkpointer instance compatible with LangGraph
        """
        return self.checkpointer
    
    def list_threads(self, limit: int = 50) -> List[Tuple[str, str]]:
        """
        List all pipeline execution threads with their latest checkpoints.
        
        Args:
            limit: Maximum number of threads to return
        
        Returns:
            List of tuples: (thread_id, latest_checkpoint_timestamp)
        """
        
        if self.backend == "sqlite":
            import sqlite3
            
            try:
                # Check if database file exists
                if not os.path.exists(self.db_path):
                    print(f"[CHECKPOINT] Database not found: {self.db_path}")
                    return []
                
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Query for unique threads and their latest checkpoint
                cursor.execute("""
                    SELECT 
                        thread_id,
                        MAX(checkpoint_id) as latest_checkpoint
                    FROM checkpoints
                    GROUP BY thread_id
                    ORDER BY latest_checkpoint DESC
                    LIMIT ?
                """, (limit,))
                
                threads = cursor.fetchall()
                conn.close()
                
                return threads
                
            except sqlite3.OperationalError as e:
                print(f"[CHECKPOINT] Database error (might not exist yet): {e}")
                return []
            except Exception as e:
                print(f"[CHECKPOINT] Error listing threads: {e}")
                return []
        
        elif self.backend == "postgres":
            # Similar implementation for Postgres
            print("[CHECKPOINT] Postgres thread listing not yet implemented")
            return []
        
        else:
            # Memory backend doesn't persist
            print("[CHECKPOINT] Memory backend has no persistent threads")
            return []
    
    def get_checkpoint_info(self, thread_id: str) -> Optional[dict]:
        """
        Get information about a specific checkpoint thread.
        
        Args:
            thread_id: The thread ID to inspect
        
        Returns:
            Dictionary with checkpoint information or None if not found
        """
        
        if self.backend == "sqlite":
            import sqlite3
            
            try:
                if not os.path.exists(self.db_path):
                    return None
                    
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        checkpoint_id,
                        parent_checkpoint_id,
                        created_at
                    FROM checkpoints
                    WHERE thread_id = ?
                    ORDER BY checkpoint_id DESC
                    LIMIT 1
                """, (thread_id,))
                
                result = cursor.fetchone()
                conn.close()
                
                if result:
                    return {
                        "thread_id": thread_id,
                        "checkpoint_id": result[0],
                        "parent_checkpoint_id": result[1],
                        "created_at": result[2]
                    }
                
            except Exception as e:
                print(f"[CHECKPOINT] Error getting checkpoint info: {e}")
        
        return None
    
    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Remove checkpoints older than specified days.
        
        Args:
            days: Age threshold in days (checkpoints older than this will be deleted)
        
        Returns:
            Number of checkpoints deleted
        """
        
        print(f"[CHECKPOINT] Cleaning up checkpoints older than {days} days...")
        
        if self.backend == "sqlite":
            import sqlite3
            
            try:
                if not os.path.exists(self.db_path):
                    print(f"[CHECKPOINT] No database to clean: {self.db_path}")
                    return 0
                    
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Calculate cutoff timestamp
                cutoff = datetime.now() - timedelta(days=days)
                cutoff_str = cutoff.isoformat()
                
                # Delete old checkpoints
                cursor.execute("""
                    DELETE FROM checkpoints
                    WHERE created_at < ?
                """, (cutoff_str,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                print(f"[CHECKPOINT] ✓ Deleted {deleted_count} old checkpoints")
                return deleted_count
                
            except Exception as e:
                print(f"[CHECKPOINT] ✗ Error during cleanup: {e}")
                return 0
        
        elif self.backend == "postgres":
            print("[CHECKPOINT] Postgres cleanup not yet implemented")
            return 0
        
        else:
            # Memory backend doesn't need cleanup
            print("[CHECKPOINT] Memory backend has no persistent data to clean")
            return 0
    
    def delete_thread(self, thread_id: str) -> bool:
        """
        Delete all checkpoints for a specific thread.
        
        Args:
            thread_id: The thread ID to delete
        
        Returns:
            True if successful, False otherwise
        """
        
        if self.backend == "sqlite":
            import sqlite3
            
            try:
                if not os.path.exists(self.db_path):
                    print(f"[CHECKPOINT] No database found: {self.db_path}")
                    return False
                    
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM checkpoints
                    WHERE thread_id = ?
                """, (thread_id,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                conn.close()
                
                print(f"[CHECKPOINT] Deleted {deleted_count} checkpoint(s) for thread: {thread_id}")
                return True
                
            except Exception as e:
                print(f"[CHECKPOINT] Error deleting thread: {e}")
                return False
        
        return False


# Singleton instance
_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(
    backend: str = "memory",  # Changed default to memory
    db_path: str = "checkpoints.db",
    force_recreate: bool = False
) -> CheckpointManager:
    """
    Get or create the global checkpoint manager instance.
    
    This function implements the singleton pattern to ensure only one
    checkpoint manager exists per application instance.
    
    Args:
        backend: Backend type ("sqlite", "postgres", or "memory")
        db_path: Path to SQLite database
        force_recreate: Force creation of new instance (useful for testing)
    
    Returns:
        CheckpointManager instance
    """
    global _checkpoint_manager
    
    if _checkpoint_manager is None or force_recreate:
        _checkpoint_manager = CheckpointManager(
            backend=backend,
            db_path=db_path
        )
    
    return _checkpoint_manager


def generate_thread_id(topic: str, include_timestamp: bool = False) -> str:
    """
    Generate a deterministic thread ID from a topic.
    
    This ensures the same topic always generates the same thread ID,
    which is useful for resuming pipelines.
    
    Args:
        topic: Article topic
        include_timestamp: If True, includes timestamp for uniqueness
    
    Returns:
        Thread ID string
    """
    
    if include_timestamp:
        # Unique ID for each execution
        timestamp = datetime.now().isoformat()
        combined = f"{topic}_{timestamp}"
        hash_input = combined.encode()
    else:
        # Deterministic ID based on topic only
        hash_input = topic.encode()
    
    topic_hash = hashlib.md5(hash_input).hexdigest()[:12]
    return f"article_{topic_hash}"


def get_thread_id_with_timestamp(topic: str) -> str:
    """
    Generate a unique thread ID with timestamp.
    
    Use this when you want each execution to be independent,
    even for the same topic.
    
    Args:
        topic: Article topic
    
    Returns:
        Unique thread ID string
    """
    return generate_thread_id(topic, include_timestamp=True)
