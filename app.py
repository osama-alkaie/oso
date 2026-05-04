
import random                            
import pandas as pd                        # لقراءة ملفات Excel 
from fastapi import FastAPI, HTTPException 
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel             
from typing import List, Dict              
import uvicorn                             


# إعدادات الخوارزمية الجينية 
POP_SIZE   = 40   # عدد الحلول في كل جيل
GENE_COUNT = 10   # (طول الكروموسوم)
NUM_GENS   = 20   # عدد الأجيال التي تمر عليها الخوارزمية
CR         = 0.8  
MR         = 0.2 
T_SIZE     = 3    
ELIT       = 2    


DATA_PATH = "./data"


# تعريف نماذج البيانات المرجعة من API    
class OneProduct(BaseModel):
    """بيانات منتج واحد مقترح"""
    product_id: int     
    category: str       
    price: float        
    score: float        
    rating: float       # متوسط تقييمات المستخدمين للمنتج

class ResponseBody(BaseModel):
    """الرد الكامل من endpoint /recommend"""
    user_id: int                        
    user_location: str                  
    recommendations: List[OneProduct]   



# تحميل البيانات من ملفات Excel 
def load_all_data(path):
    """تقرأ الملفات الأربعة وتنظف أسماء الأعمدة"""
    try:
        u = pd.read_excel(f"{path}/users.xlsx")     
        p = pd.read_excel(f"{path}/products.xlsx")  
        b = pd.read_excel(f"{path}/behavior.xlsx")  
        r = pd.read_excel(f"{path}/ratings.xlsx")   

        # أزالة المسافات من أسماء الأعمدة في اكسل
        for df in [u, p, b, r]:
            df.columns = [str(c).strip() for c in df.columns]
        return u, p, b, r
    except Exception as err:
        raise RuntimeError(f"data load error: {err}")

# نحمل البيانات مرة واحدة عند بدء تشغيل ال server
users_df, products_df, behavior_df, ratings_df = load_all_data(DATA_PATH)


# نحسب متوسط تقييم كل منتج مرة واحدة 
avg_rat = ratings_df.groupby("product_id")["rating"].mean().to_dict()


# دالة البحث عن مستخدم 
def get_user_by_id(uid):
    """تبحث عن المستخدم في الداتا وترجع صفه، أو None لو مو موجود"""
    res = users_df[users_df["user_id"] == uid]
    if res.empty:
        return None
    return res.iloc[0]



#  دالة بناء خريطة تفضيلات المستخدم 
def build_scores(uid):
    """
    تحسب درجة لكل منتج بناءً على سلوك المستخدم وتقييماته.
    الدرجة الأعلى = المنتج أكثر ملاءمة لهذا المستخدم.
    """
    scores = {}

    # نحسب درجة السلوك
    beh = behavior_df[behavior_df["user_id"] == uid]
    for _, row in beh.iterrows():
        pid = int(row["product_id"])
        val = (
            int(row.get("viewed",    0)) * 0.2   
            + int(row.get("clicked",   0)) * 0.5  
            + int(row.get("purchased", 0)) * 1.0  
        )
        scores[pid] = scores.get(pid, 0.0) + val

    # نضيف درجة التقييم
    rat = ratings_df[ratings_df["user_id"] == uid]
    for _, row in rat.iterrows():
        pid = int(row["product_id"])
        r_val = (float(row.get("rating", 0)) / 5.0) * 1.5
        scores[pid] = scores.get(pid, 0.0) + r_val

    return scores



# دالة حساب ال Fitness 
def fitness(chrom, score_map):
    """
    تحسب جودة الكروموسوم (قائمة التوصيات).
    النتيجة بين 0 و1 — كلما اقتربت من 1 كلما كانت التوصية أفضل.
    """
    if len(chrom) == 0:
        return 0.0
    total = 0.0
    for pid in chrom:
        total += score_map.get(pid, 0.0)  # نجمع درجات كل المنتجات في الكروموسوم
    return total / (len(chrom) * 3.2)



# دالة Tournament Selection 
def tournament(pop, fits):
    """
    تختار أفضل حل من بين T_SIZE حلول مختارة عشوائياً.
    هذا أعدل من اختيار الأفضل مباشرة لأنه يحافظ على التنوع.
    """
    idxs = random.sample(range(len(pop)), T_SIZE)  # نختار T_SIZE حلول عشوائياً
    best = max(idxs, key=lambda i: fits[i])         # نرجع الأفضل من بينهم
    return pop[best][:]



#دالة ال  Crossover (دمج حلين)
def crossover(p1, p2):
    """
    نأخذ جزء من الأب الأول وجزء من الأب الثاني لنكوّن ابن جديد.
    نتأكد أن الابن لا يحتوي على منتجات مكررة.
    """
    if random.random() >= CR or len(p1) <= 1:
        return p1[:]

    # نختار نقطة قطع عشوائية
    pt = random.randint(1, len(p1) - 1)

    # dict.fromkeys يزيل التكرار ويحافظ على الترتيب
    child = list(dict.fromkeys(p1[:pt] + p2[pt:]))
    all_pids = products_df["product_id"].tolist()

    # لو فضي مواقع فارغة منكملها بمنتجات عشوائية
    while len(child) < len(p1):
        pick = random.choice(all_pids)
        if pick not in child:
            child.append(pick)
    return child[:len(p1)]



#دالة ال  Mutation
def mutate(chrom):
    """
    نبدل موقعين عشوائيين في الكروموسوم لإضافة تنوع.
    هذا يمنع الخوارزمية من الوقوع في أفضل حل محلي.
    """

    if random.random() >= MR or len(chrom) < 2:
        return chrom
    c = chrom[:]
    # نختار موقعين مختلفين ونبدلهم
    i, j = random.sample(range(len(c)), 2)
    c[i], c[j] = c[j], c[i]
    return c



# الدالة الرئيسية للخوارزمية الجينية 
def run_ga(uid):
    """
    تشغل الخوارزمية الجينية كاملة للمستخدم وترجع أفضل قائمة توصيات.
    """
    # نبني خريطة تفضيلات المستخدم من بياناته
    score_map = build_scores(uid)

    all_products = products_df["product_id"].unique().tolist()
    gsize = min(len(all_products), GENE_COUNT)
    if gsize == 0:
        return []

    # نولد الجيل الأول عشوائياً 
    pop = [random.sample(all_products, gsize) for _ in range(POP_SIZE)]

    # نكرر على عدد الأجيال المحددة
    for gen in range(NUM_GENS):
        # نحسب ال fitness 
        fits = [fitness(ind, score_map) for ind in pop]

        # نرتب الحلول من الأفضل للأسوأ
        sorted_idx = sorted(range(POP_SIZE), key=lambda i: fits[i], reverse=True)

        new_pop = [pop[sorted_idx[i]][:] for i in range(ELIT)]

        #نبني باقي الجيل الجديد عبر ال   selection وال crossover وال mutation
        while len(new_pop) < POP_SIZE:
            p1 = tournament(pop, fits)  
            p2 = tournament(pop, fits)   
            kid = crossover(p1, p2)      
            kid = mutate(kid)           
            new_pop.append(kid)

        
        pop = new_pop

    
    final_fits = [fitness(ind, score_map) for ind in pop]
    best_i = max(range(POP_SIZE), key=lambda i: final_fits[i])
    best_chrom = pop[best_i]

    return [(pid, score_map.get(pid, 0.0)) for pid in best_chrom]



#  تهيئة تطبيق FastAPI 
app = FastAPI(title="BIA601 Recommender")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # نسمح لأي domain يتصل بالـ API
    allow_credentials=True,
    allow_methods=["*"],       # نسمح لكل أنواع الطلبات (GET, POST, ...)
    allow_headers=["*"],
)


@app.get("/recommend", response_model=ResponseBody)
def recommend(user_id: int):
    """
    يستقبل user_id، يشغّل الخوارزمية الجينية، ويرجع قائمة المنتجات المقترحة.
    """
    # نتحقق أن المستخدم موجود في الداتا
    user = get_user_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")

    # نشغل الخوارزمية الجينية ونستلم القائمة
    results = run_ga(user_id)
    if not results:
        raise HTTPException(status_code=404, detail="no data for this user")

    recs = []
    for pid, raw_sc in results:
       
        prow = products_df[products_df["product_id"] == pid]
        if prow.empty:
            continue
        p = prow.iloc[0]

        
        norm = round(min(raw_sc / 3.2, 1.0), 2)

        recs.append(OneProduct(
            product_id = int(pid),
            category   = str(p["category"]),
            price      = float(p["price"]),
            score      = norm,
            rating     = round(float(avg_rat.get(pid, 0.0)), 1)  
        ))

    return ResponseBody(
        user_id         = int(user["user_id"]),
        user_location   = str(user.get("location", "Unknown")),
        recommendations = recs
    )


# جلب أول 20 مستخدم 
@app.get("/users")
def get_users():
    """
    يرجع قائمة بأول 20 مستخدم من الداتا.
    يُستخدم في الفرونت لعرض الـ chips (اختيار سريع).
    """
    data = users_df.head(20)[["user_id", "location", "age"]].to_dict(orient="records")
    return {"users": data}


# تشغيل ال server 
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
