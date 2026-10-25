import os
import tempfile

# Keep API tests away from the real data directory: app.main creates its stores at import.
os.environ.setdefault("DOCCOG_DATA_DIR", tempfile.mkdtemp(prefix="doccog-tests-"))
# TestClient sends "Host: testserver"; the API otherwise answers loopback hosts only.
os.environ.setdefault("DOCCOG_ALLOWED_HOSTS", "testserver")
