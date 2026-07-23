import requests
import os
from pathlib import Path

def run_verification():
    port = os.getenv("PORT", "8000")
    backend_url = f"http://127.0.0.1:{port}"
    print(f"Verifying backend connectivity at: {backend_url}...")
    
    # 1. Health check
    try:
        health_resp = requests.get(f"{backend_url}/health", timeout=10)
        if health_resp.status_code == 200:
            print("  [PASS] Backend health check successful.")
            print(f"         Active LLM Provider: {health_resp.json()['metrics']['settings']['llm_provider']}")
        else:
            print(f"  [FAIL] Health check returned status: {health_resp.status_code}")
            return False
    except Exception as e:
        print(f"  [FAIL] Failed to connect to health endpoint: {e}")
        return False
        
    # 2. Create a temporary document
    test_file = Path("test_grounding.txt")
    test_content = (
        "The Verifiable RAG Assistant is developed by the Google DeepMind team.\n"
        "It supports domain-independent document ingestion and hybrid dense-sparse search.\n"
        "The system runs on Render and uses local embedding caching to prevent OOM errors.\n"
    )
    test_file.write_text(test_content, encoding="utf-8")
    print(f"Created temporary file: {test_file.name}")
    
    doc_id = None
    try:
        # 3. Upload Document
        print("Uploading document to backend...")
        with open(test_file, "rb") as f:
            upload_resp = requests.post(
                f"{backend_url}/upload",
                files={"file": (test_file.name, f, "text/plain")},
                timeout=30
            )
            
        if upload_resp.status_code == 200:
            res_json = upload_resp.json()
            doc_id = res_json.get("doc_id")
            print(f"  [PASS] Upload successful. Document registered with ID: {doc_id}")
            print(f"         Total chunks generated: {res_json.get('chunk_count')}")
        else:
            print(f"  [FAIL] Upload failed with status {upload_resp.status_code}: {upload_resp.text}")
            return False
            
        # 4. Query RAG System
        query = "Who developed the Verifiable RAG Assistant?"
        print(f"Querying system: '{query}'...")
        query_resp = requests.post(
            f"{backend_url}/query",
            json={"query": query},
            timeout=30
        )
        
        if query_resp.status_code == 200:
            query_json = query_resp.json()
            answer = query_json.get("answer", "")
            print("  [PASS] Query executed successfully.")
            print(f"         Answer: {answer}")
            print("         Citations verified:")
            for claim in query_json.get("verification_results", []):
                print(f"           - Claim: '{claim['claim']}'")
                print(f"             Status: {claim['status']} (Confidence: {claim['confidence_score']:.1f}%)")
        else:
            print(f"  [FAIL] Query failed with status {query_resp.status_code}: {query_resp.text}")
            return False
            
    except Exception as e:
        print(f"  [ERROR] Exception encountered during end-to-end verification: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 5. Clean up document from database
        if doc_id:
            print(f"Cleaning up uploaded document {doc_id}...")
            try:
                del_resp = requests.delete(f"{backend_url}/documents/{doc_id}", timeout=10)
                if del_resp.status_code == 200:
                    print("  [PASS] Document cleaned up from database.")
                else:
                    print(f"  [WARNING] Cleanup returned status: {del_resp.status_code}")
            except Exception as e:
                print(f"  [WARNING] Cleanup failed: {e}")
                
        # 6. Clean up file on disk
        if test_file.exists():
            test_file.unlink()
            print("Cleaned up temporary test file from disk.")
            
    return True

if __name__ == "__main__":
    success = run_verification()
    if success:
        print("\n=== VERIFICATION SUCCESSFUL ===")
    else:
        print("\n=== VERIFICATION FAILED ===")
        exit(1)
