"""
Compatibility shim: call the real builder so your existing README steps still work.
"""
from build_catalog_real import main

if __name__ == "__main__":
    main()
