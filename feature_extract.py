# feature_extraction.py
import re
import json
from urllib.parse import urlparse
import pandas as pd
import math

# small list of known URL shortener hosts (helpful)
_SHORTENERS = {
    "bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","buff.ly","is.gd","tiny.cc","lc.chat",
    "rebrand.ly","shorturl.at","su.pr","trib.al"
}

def _safe_len(s):
    return 0 if s is None else len(s)

def _has_port(netloc):
    # returns port number or 0
    if ':' in netloc:
        parts = netloc.split(':')
        if parts[-1].isdigit():
            return int(parts[-1])
    return 0

def compute_url_superset_features(url: str):
    """
    Compute a superset of URL-derived features. Not all of these may be used
    by the model; the app will select the subset matching model metadata.
    Returns a dict
    """
    if not isinstance(url, str):
        url = str(url or "")

    parsed = urlparse(url if url.startswith(("http://","https://")) else "http://"+url)
    host = parsed.netloc.lower()
    path = parsed.path or ""
    query = parsed.query or ""
    whole = url

    features = {}
    # basic counts
    features["length_url"] = _safe_len(whole)
    features["length_hostname"] = _safe_len(host)
    features["nb_dots"] = whole.count(".")
    features["nb_hyphens"] = whole.count("-")
    features["nb_at"] = whole.count("@")
    features["nb_qm"] = whole.count("?")
    features["nb_and"] = whole.count("&")
    features["nb_or"] = whole.count("|")
    features["nb_eq"] = whole.count("=")
    features["nb_underscore"] = whole.count("_")
    features["nb_tilde"] = whole.count("~")
    features["nb_percent"] = whole.count("%")
    features["nb_slash"] = whole.count("/")
    features["nb_star"] = whole.count("*")
    features["nb_colon"] = whole.count(":")
    features["nb_com"] = whole.lower().count(".com")
    features["nb_www"] = whole.lower().count("www")
    features["nb_dslash"] = whole.count("//") - 1  # after protocol there is usually one //
    features["http_in_path"] = 1 if "http" in path.lower() else 0
    features["https_token"] = 1 if "https" in host or "https" in whole.lower() else 0

    # numeric ratios
    digits = sum(ch.isdigit() for ch in whole)
    features["ratio_digits_url"] = digits / features["length_url"] if features["length_url"]>0 else 0.0
    digits_host = sum(ch.isdigit() for ch in host)
    features["ratio_digits_host"] = digits_host / features["length_hostname"] if features["length_hostname"]>0 else 0.0

    # punycode detection
    features["punycode"] = 1 if host.startswith("xn--") or "xn--" in host else 0

    # port
    features["port"] = _has_port(host)

    # tld in path/subdomain heuristics (basic)
    # tld pattern = ".<2-4 letters>" e.g. .php won't be tld but this is heuristic
    features["tld_in_path"] = 1 if re.search(r'\.[a-z]{2,4}($|/)', path.lower()) else 0
    features["tld_in_subdomain"] = 1 if re.search(r'\.[a-z]{2,4}', host.split(':')[0]) else 0

    # subdomain abnormalities
    features["nb_subdomains"] = host.count(".")
    features["prefix_suffix"] = 1 if "-" in host else 0
    # quick random-domain heuristic: high length and low vowel ratio or many digits
    vowels = sum(ch in "aeiou" for ch in host)
    features["random_domain"] = 1 if (_safe_len(host)>15 and (vowels/_safe_len(host) < 0.2 or digits_host/_safe_len(host) > 0.3)) else 0

    # shortening service
    short = host.split(':')[0]
    features["shortening_service"] = 1 if short in _SHORTENERS else 0

    # path extension check
    features["path_extension"] = 1 if re.search(r'\.(php|asp|aspx|jsp|html|cfm|cgi)$', path.lower()) else 0

    # prefix suspicious words
    suspicious_words = ['login','signin','secure','verify','update','account','confirm','bank','paypal','ebay']
    features["phish_hints"] = sum(1 for w in suspicious_words if w in whole.lower())

    # a few boolean flags derivable
    features["abnormal_subdomain"] = 1 if features["nb_subdomains"]>3 else 0
    features["domain_in_brand"] = 0
    features["brand_in_subdomain"] = 0
    features["brand_in_path"] = 0

    # fallback numeric defaults for features that are not derivable from URL alone:
    # these will be filled by medians later if model expects them but we can't compute them.
    # (we leave them out here; app will fill from medians)
    return features

def extract_features_for_model(url, meta_path="robust_meta.json"):
    """
    Build a pandas.DataFrame with columns matching the trained model.
    - Loads meta_path (JSON) which must contain keys:
         { "features": [list-of-feature-names], "medians": {feat:median, ...} }
    - Computes a superset of URL-derived features, then assembles row by model features,
      filling missing entries using medians from meta.
    """
    # load meta
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        model_features = meta.get("features", [])
        medians = meta.get("medians", {})
    except Exception:
        raise FileNotFoundError(f"Meta file '{meta_path}' not found or bad format. Run training block first.")

    sup = compute_url_superset_features(url)
    # assemble dict for model features
    row = {}
    for feat in model_features:
        if feat in sup:
            row[feat] = sup[feat]
        else:
            # if not computable from URL, fallback to median from meta or 0
            row[feat] = medians.get(feat, 0.0)
    # return single-row DataFrame
    return pd.DataFrame([row], columns=model_features)
