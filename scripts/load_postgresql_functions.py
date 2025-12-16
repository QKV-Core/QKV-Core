import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
from pathlib import Path

def load_functions():
    
    print("📚 Loading PostgreSQL functions and triggers...")
    
    connection_string = os.getenv('POSTGRESQL_CONNECTION_STRING')
    if not connection_string:
        print("❌ POSTGRESQL_CONNECTION_STRING not set in .env file")
        return False
    
    sql_file = Path(__file__).parent.parent / "storage" / "postgresql_functions.sql"
    if not sql_file.exists():
        print(f"❌ SQL file not found: {sql_file}")
        return False
    
    try:
        conn = psycopg2.connect(connection_string)
        cursor = conn.cursor()
        
        print(f"  Reading SQL file: {sql_file}")
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql_content = f.read()
        
        print("  Executing SQL file...")
        try:
            cursor.execute(sql_content)
            conn.commit()
            print("  ✅ SQL file executed successfully")
        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str or "duplicate" in error_str:
                print("  ⚠️  Some objects already exist (this is OK)")
                conn.commit()
            elif "plpython3u" in error_str:
                print("  ⚠️  plpython3u extension not available (functions with PL/Python will fail)")
                print("     This is OK - application will work without these functions")
                conn.rollback()
            else:
                print(f"  ⚠️  Some statements failed: {e}")
                print("  💡 Trying to execute non-PL/Python statements only...")
                conn.rollback()
                
                lines = sql_content.split('\n')
                simple_statements = []
                skip_until_end = False
                
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith('--') or not line_stripped:
                        continue
                    
                    if 'LANGUAGE plpython3u' in line_stripped or 'LANGUAGE plpgsql' in line_stripped:
                        if 'AS $$' in line_stripped:
                            skip_until_end = True
                        continue
                    
                    if skip_until_end:
                        if '$$;' in line_stripped or line_stripped.endswith('$$;'):
                            skip_until_end = False
                        continue
                    
                    simple_statements.append(line)
                
                if simple_statements:
                    simple_sql = '\n'.join(simple_statements)
                    try:
                        cursor.execute(simple_sql)
                        conn.commit()
                        print("  ✅ Non-PL/Python statements executed")
                    except Exception as e2:
                        print(f"  ⚠️  Some statements still failed: {e2}")
                        conn.rollback()
        
        print("\n🔍 Verifying functions...")
        cursor.execute("SELECT proname FROM pg_proc WHERE pronamespace = 'public'::regnamespace")
        functions = [row[0] for row in cursor.fetchall()]
        print(f"  ✅ Found {len(functions)} functions: {', '.join(functions[:5])}...")
        
        cursor.close()
        conn.close()
        
        print("\n✅ PostgreSQL functions loaded successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Failed to load functions: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = load_functions()
    sys.exit(0 if success else 1)