# app/feature_extractor.py
import tempfile
from typing import Dict
from zipfile import ZipFile
from pyaxmlparser import APK  # pure-Python; parses AndroidManifest.xml

PERM_PREFIX = "perm_"

def extract_features_from_apk(apk_bytes: bytes, selected_features: list[str]) -> Dict[str, float]:
    # write APK bytes to a temp file
    with tempfile.NamedTemporaryFile(suffix=".apk") as tmp:
        tmp.write(apk_bytes)
        tmp.flush()
        a = APK(tmp.name)

        # pyaxmlparser exposes permissions as a list of full names like
        # 'android.permission.INTERNET'
        perms = set(a.permissions or [])
        perms_norm = {p.replace(".", "_").upper() for p in perms}

        feats: Dict[str, float] = {}

        # 1) Permission presence -> binary features (if such features exist)
        for feat in selected_features:
            if feat.startswith(PERM_PREFIX):
                perm_key = feat[len(PERM_PREFIX):]  # e.g. 'ANDROID_PERMISSION_INTERNET'
                feats[feat] = 1.0 if any(perm_key in p for p in perms_norm) else 0.0

        # 2) Manifest stats (only if present in your selected_features)
        if "num_permissions" in selected_features:
            feats["num_permissions"] = float(len(perms))

        # min_sdk_version (returns int or None)
        if "min_sdk" in selected_features:
            try:
                feats["min_sdk"] = float(a.min_sdk_version or 0)
            except Exception:
                feats["min_sdk"] = 0.0

        # 3) Zip-level signals (size / file count)
        try:
            with ZipFile(tmp.name) as z:
                if "apk_num_files" in selected_features:
                    feats["apk_num_files"] = float(len(z.infolist()))
                if "apk_size_bytes" in selected_features:
                    feats["apk_size_bytes"] = float(sum(i.file_size for i in z.infolist()))
        except Exception:
            pass

        return feats
