"""ShopSense - content-based product recommender (TF-IDF + cosine similarity). Run: python app.py"""
import re, pickle
import numpy as np, pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.metrics.pairwise import linear_kernel

app = Flask(__name__)
df = pd.read_pickle("data/df.pkl").reset_index(drop=True)
VEC = pickle.load(open("data/vec_tfidf.pkl", "rb"))   # TF-IDF matrix from your notebook

RULES = [("Nails", r"nail|polish|manicure"), ("Hair Care", r"hair|shampoo|conditioner"),
 ("Fragrance", r"fragrance|perfume|cologne|eau de|toilette|body spray"), ("Oral Care", r"oral|tooth|mouthwash"),
 ("Men's Grooming", r"shaving|razor|\bmen\b|grooming"), ("Makeup", r"makeup|foundation|lip|mascara|eye ?shadow|concealer|blush|eyeliner"),
 ("Skin Care", r"skin|face|facial|cream|lotion|moistur|sunscreen|cleanser"), ("Bath & Body", r"bath|body|soap|shower|deodorant|wash"),
 ("Health & Wellness", r"health|vitamin|supplement|superfood|incontinence|feminine"), ("Home", r"household|cleaning|air fresh|home")]
def cat_of(text):
    t = str(text).lower()
    return next((c for c, p in RULES if re.search(p, t)), "More")

PROD = []
for i, r in df.iterrows():
    imgs = [u.strip() for u in str(r["ImageURL"]).split("|") if u.strip().startswith("http")]
    brand = "" if pd.isna(r["Brand"]) else str(r["Brand"]).replace(",", "").title()
    desc = "" if pd.isna(r["Description"]) else str(r["Description"]).replace(",", "")[:300]
    PROD.append(dict(id=int(i), name=r["Name"], brand=brand, cat=cat_of(f'{r["Category"]} {r["Name"]}'),
                     rating=float(r["Rating"]), reviews=int(r["ReviewCount"]), imgs=imgs[:6], desc=desc))
CARD = lambda p: {k: p[k] for k in ("id", "name", "brand", "cat", "rating", "reviews")} | {"img": p["imgs"][0] if p["imgs"] else ""}

@app.route("/")
def home(): return render_template("index.html")

@app.route("/api/products")
def products():
    q, cat = request.args.get("q", "").lower().strip(), request.args.get("cat", "All")
    sort, page = request.args.get("sort", "popular"), int(request.args.get("page", 1))
    items = [p for p in PROD if p["imgs"] and (cat == "All" or p["cat"] == cat)
             and (not q or all(re.search(r"\b" + re.escape(w), (p["name"] + " " + p["brand"]).lower()) for w in q.split()))]
    if sort == "rating": items.sort(key=lambda p: (p["rating"], p["reviews"]), reverse=True)
    elif sort == "reviews": items.sort(key=lambda p: p["reviews"], reverse=True)
    else: items.sort(key=lambda p: p["reviews"] * (1 + p["rating"]), reverse=True)
    per = 24
    return jsonify(total=len(items), items=[CARD(p) for p in items[(page - 1) * per: page * per]])

@app.route("/api/suggest")
def suggest():
    q = request.args.get("q", "").lower().strip()
    out = [CARD(p) for p in PROD if q and p["imgs"] and re.search(r"\b" + re.escape(q), p["name"].lower())][:6]
    return jsonify(out)

@app.route("/api/product/<int:i>")
def product(i):
    if not 0 <= i < len(PROD): return jsonify(error="not found"), 404
    sims = linear_kernel(VEC[i], VEC).ravel(); sims[i] = -1      # cosine similarity (TF-IDF rows are L2-normalised)
    recs, seen = [], {PROD[i]["name"]}
    for j in sims.argsort()[::-1]:
        if PROD[j]["name"] in seen or not PROD[j]["imgs"]: continue
        seen.add(PROD[j]["name"]); recs.append(CARD(PROD[j]) | {"score": round(float(sims[j]) * 100, 1)})
        if len(recs) == 12: break
    return jsonify(product=PROD[i], recs=recs)

@app.route("/api/stats")
def stats():
    d = pd.DataFrame(PROD); rated = d[d.rating > 0]
    return jsonify(products=len(d), brands=int(d[d.brand != ""].brand.nunique()), reviews=int(d.reviews.sum()),
        avg_rating=round(float(rated.rating.mean()), 2), features=int(VEC.shape[1]),
        cats=d.cat.value_counts().to_dict(), hero=[CARD(p) for p in sorted(PROD, key=lambda p: -p["reviews"]) if p["imgs"]][:60:3])

if __name__ == "__main__":
    app.run(debug=True, port=5000)
