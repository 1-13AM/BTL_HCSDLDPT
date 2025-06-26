import numpy as np

def chi2_distance(h1, h2, eps=1e-10):
    h1, h2 = np.asarray(h1), np.asarray(h2)
    return 0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps))

def scd_distance(tokens1, tokens2, alpha=0.5, N=5):
    def extract_vec(tokens):
        return np.array([[t['max_curvature'], t['orientation']] for t in tokens[:N]])
    
    v1 = extract_vec(tokens1)
    v2 = extract_vec(tokens2)
    
    if len(v1) != len(v2):
        min_len = min(len(v1), len(v2))
        v1 = v1[:min_len]
        v2 = v2[:min_len]
    diff = v1 - v2
    return np.mean(np.sqrt(diff[:, 0]**2 + alpha * diff[:, 1]**2))

def cosine_distance(v1, v2):
    v1, v2 = np.asarray(v1), np.asarray(v2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0: return 1.0
    return 1.0 - dot / norm

def clf_distance(clf1, clf2):
    def flatten(clf_dict):
        return np.concatenate([clf_dict[k] for k in sorted(clf_dict.keys())])
    
    v1 = flatten(clf1)
    v2 = flatten(clf2)
    return np.linalg.norm(v1 - v2)

def compute_distance(feat1, feat2, weights=None):
    weights = weights or {'cch': 1.0, 'scd': 1.0, 'bas': 1.0, 'clf': 1.0}

    try:
        d_cch = chi2_distance(feat1['cch'], feat2['cch'])
    except:
        d_cch = 0

    try:
        d_scd = scd_distance(feat1['scd_tokens'], feat2['scd_tokens'])
    except:
        d_scd = 0

    try:
        d_bas = cosine_distance(feat1['bas_vector_fd'], feat2['bas_vector_fd'])
    except:
        d_bas = 0

    try:
        d_clf = clf_distance(feat1['clf'], feat2['clf'])
    except:
        d_clf = 0

    total = (
        weights['cch'] * d_cch +
        weights['scd'] * d_scd +
        weights['bas'] * d_bas +
        weights['clf'] * d_clf
    )
    return total
