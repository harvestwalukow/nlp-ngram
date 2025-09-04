# app.py — News N-gram Game (Backend + Simple UI)
from __future__ import annotations
import os, re, time, math, random, unicodedata, statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import ORJSONResponse, HTMLResponse
from pydantic import BaseModel, Field, constr

# =========================
# Config (env optional)
# =========================
PORT = int(os.getenv("PORT", 8000))
NEWS_CSV = os.getenv("NEWS_CSV", "data/judul_fix.csv")
TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 15))  # default 15s
QUESTIONS_PER_PLAYER = int(os.getenv("QUESTIONS_PER_PLAYER", 7))
SEED = int(os.getenv("SEED", 42))
MIN_FREQ = int(os.getenv("MIN_FREQ", 2))  # min token freq to avoid <unk>

# =========================
# Utils
# =========================
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\- ]+", flags=re.UNICODE)

def normalize_text_for_model(s: str) -> str:
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _punct_re.sub(" ", s)
    s = _ws_re.sub(" ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    if not s: return []
    return s.split()

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# =========================
# Data
# =========================
def load_titles(csv_path: str = None) -> List[str]:
    path = csv_path or NEWS_CSV
    df = pd.read_csv(path)
    cols = [c.strip().lower() for c in df.columns]
    df.columns = cols
    if "title" not in df.columns:
        raise ValueError("Kolom 'title' tidak ditemukan di CSV.")
    # Filter dasar: drop NA, ke string
    titles = df["title"].dropna().astype(str).tolist()
    return titles

# =========================
# Stopwords & Validators
# =========================
ID_STOPWORDS = {
    # paling umum
    "yang","dan","di","ke","dari","untuk","pada","dengan","atau","ini","itu","para","sebuah","sebuahnya",
    "sebagai","adalah","akan","jadi","agar","karena","oleh","dalam","kata","saat","tak","tidak","bukan",
    "ya","pun","juga","sudah","belum","masih","hingga","sampai","serta","antara","tanpa","keputusan",
    "lebih","kurang","bisa","dapat","telah","telah","bahwa","oleh","terhadap","agar","guna",
    # bentuk preposisi/konjungsi lain
    "atau","namun","tapi","tetapi","sementara","meski","meskipun","jika","apabila","bila","walau","walaupun",
    "karena","sebab","supaya","sehingga","sebelum","sesudah","ketika","ketika","kalau",
    # artikel/pronomina netral
    "sebuah","sebuah","sebuahnya","ia","dia","mereka","kami","kita","saya","aku","anda",
    # penanda waktu umum (biar blank bukan kata trivial)
    "hari","bulan","tahun","besok","kemarin","hariini","hari-ini","hari_ini","malam","pagi","siang","sore"
}

def is_numeric_like(tok: str) -> bool:
    # angka, angka-bercampur, tanggal singkat, dsb.
    if tok.isdigit():
        return True
    if re.fullmatch(r"\d+[\-/.]\d+([\-/\.]\d+)?", tok):
        return True
    if re.fullmatch(r"\d+[a-z]+", tok):
        return True
    return False

def is_good_blank_token(tok: str) -> bool:
    if not tok: return False
    if tok in {"<s>", "</s>", "<unk>"}: return False
    if tok in ID_STOPWORDS: return False
    if is_numeric_like(tok): return False
    if re.fullmatch(r"[\-]+", tok): return False
    # minimal ada huruf
    if not re.search(r"[a-z]", tok):
        return False
    return True

# =========================
# KN Trigram Model
# =========================
UNK = "<unk>"

class KNTrigramModel:
    def __init__(self, min_freq: int = MIN_FREQ, seed: int = SEED):
        self.min_freq = min_freq
        random.seed(seed)
        self.unigram = {}
        self.bigram = {}
        self.trigram = {}
        self.continuation_unigram = {}
        self.vocab = set()
        self.total_unigrams = 0
        self.D = 0.75

    def _replace_rare(self, toks_list: List[List[str]]) -> List[List[str]]:
        from collections import Counter
        flat = [t for toks in toks_list for t in toks]
        freq = Counter(flat)
        def map_tok(t): return t if freq[t] >= self.min_freq else UNK
        return [[map_tok(t) for t in toks] for toks in toks_list]

    def fit(self, raw_texts: List[str]):
        from collections import Counter, defaultdict
        tok_seqs = [tokenize(normalize_text_for_model(s)) for s in raw_texts]
        # filter absurd: ≥5 token dan proporsi huruf masuk akal
        good_tex = []
        for t in tok_seqs:
            if not t: 
                continue
            if len(t) < 5:
                continue
            # rasio token valid (bukan angka/pure symbol) minimal 70%
            valid = sum(1 for x in t if is_good_blank_token(x))
            if valid / max(1, len(t)) < 0.7:
                continue
            good_tex.append(t)
        tok_seqs = good_tex

        tok_seqs = self._replace_rare(tok_seqs)
        seqs = [["<s>", "<s>"] + t + ["</s>"] for t in tok_seqs]

        uni, bi, tri = Counter(), Counter(), Counter()
        for seq in seqs:
            for w in seq: uni[w] += 1
            for i in range(len(seq)-1): bi[(seq[i], seq[i+1])] += 1
            for i in range(len(seq)-2): tri[(seq[i], seq[i+1], seq[i+2])] += 1

        self.unigram, self.bigram, self.trigram = dict(uni), dict(bi), dict(tri)
        self.vocab = set(self.unigram.keys())
        self.total_unigrams = sum(self.unigram.values())

        unigram_ctx = defaultdict(set)
        for (w1,w2,w3), c in self.trigram.items():
            unigram_ctx[w3].add((w1,w2))
        self.continuation_unigram = {w3: len(ctxs) for w3, ctxs in unigram_ctx.items()}

    def _p_cont(self, w:str) -> float:
        denom = max(1, len(self.bigram))
        return self.continuation_unigram.get(w, 0)/denom

    def _pk_uni(self, w:str) -> float:
        p = self._p_cont(w)
        if p==0: p = self.unigram.get(w,0)/max(1,self.total_unigrams)
        return p

    def _pk_bi(self, w2:str, w3:str) -> float:
        c_bg = self.bigram.get((w2,w3),0)
        c_hist, types = 0, 0
        for (a,b),c in self.bigram.items():
            if a==w2:
                c_hist += c
                if c>0: types += 1
        if c_bg>0:
            p_mle = max(c_bg-self.D,0)/max(1,c_hist)
            lamb = (self.D*types)/max(1,c_hist)
        else:
            p_mle = 0.0
            lamb = (self.D*types)/max(1,c_hist) if c_hist>0 else 1.0
        return p_mle + lamb*self._pk_uni(w3)

    def _pk_tri(self, w1:str, w2:str, w3:str) -> float:
        c_tri = self.trigram.get((w1,w2,w3),0)
        c_hist = self.bigram.get((w1,w2),0)
        types = sum(1 for (a,b,c),cnt in self.trigram.items() if a==w1 and b==w2 and cnt>0)
        if c_tri>0:
            p_mle = max(c_tri-self.D,0)/max(1,c_hist)
            lamb = (self.D*types)/max(1,c_hist)
        else:
            p_mle = 0.0
            lamb = (self.D*types)/max(1,c_hist) if c_hist>0 else 1.0
        return p_mle + lamb*self._pk_bi(w2,w3)

    def _tok(self,w:str)->str: return w if w in self.vocab else UNK

    def avg_logprob_window(self, tokens:List[str], start:int, end:int)->float:
        toks = ["<s>","<s>"]+[self._tok(t) for t in tokens]+["</s>"]
        off=2; L=[]
        for i in range(0,len(toks)-2):
            if not (i+2 < off+start or i > off+end-1):
                p = self._pk_tri(toks[i],toks[i+1],toks[i+2])
                L.append(math.log(clamp(p,1e-12,1.0)))
        return sum(L)/len(L) if L else -12.0

    def score_insert_word(self, left:List[str], right:List[str], word:str)->float:
        seq = left+[word]+right
        pos = len(left)
        return self.avg_logprob_window(seq, pos, pos+1)

    def sample_baseline(self, k:int=20)->List[str]:
        pool = [(w,c) for w,c in self.unigram.items() if w not in {"<s>","</s>"}]
        tot = sum(c for _,c in pool)
        if not tot: return [UNK]*k
        res=[]
        for _ in range(k):
            r=random.randint(1,tot); s=0
            for w,c in pool:
                s+=c
                if s>=r: res.append(w); break
        return res

# =========================
# Game Engine
# =========================
@dataclass
class Question:
    question_id: str
    player_id: str
    index: int
    title_original: str
    left_text: str
    blank_text: str
    right_text: str
    blank_token_count: int
    expires_at: float
    answered_at: float = 0.0
    score_bucket: int | None = None

@dataclass
class PlayerState:
    id: str
    name: str
    scores: List[int] = field(default_factory=list)

@dataclass
class SessionState:
    session_id: str
    players: List[PlayerState]
    questions: List[Question]
    created_at: float

class GameEngine:
    def __init__(self):
        self.rng = random.Random(SEED)
        self.titles_original = load_titles(NEWS_CSV)
        self.titles_norm = [tokenize(normalize_text_for_model(t)) for t in self.titles_original]
        self.model = KNTrigramModel(min_freq=MIN_FREQ, seed=SEED)
        self.model.fit(self.titles_original)
        self.sessions: Dict[str, SessionState] = {}

        # Precompute kandidat indeks judul yang layak (>=5 token dan punya kandidat blank yang bagus)
        self.eligible_indices = [i for i, toks in enumerate(self.titles_norm) if self._has_good_blank(toks)]

    def _has_good_blank(self, tokens: List[str]) -> bool:
        if len(tokens) < 5:
            return False
        lo, hi = 1, len(tokens)-2
        mid = (lo+hi)//2
        # cari di sekitar tengah
        positions = list(range(max(lo, mid-2), min(hi, mid+3)))
        for pos in positions:
            if is_good_blank_token(tokens[pos]):
                return True
        # atau cari di seluruh rentang bila area tengah gagal
        for pos in range(lo, hi):
            if is_good_blank_token(tokens[pos]):
                return True
        return False

    def _pick_questions(self, n:int)->List[int]:
        cand = [i for i in self.eligible_indices]
        if len(cand) < n:
            raise ValueError("Dataset tidak cukup setelah filtering. Tambah data atau kurangi jumlah soal.")
        self.rng.shuffle(cand)
        return cand[:n]

    def _select_blank_pos(self, tokens: List[str]) -> Tuple[int, int]:
        """Pilih posisi blank yang baik: bukan stopword/angka, prioritas area tengah."""
        lo, hi = 1, len(tokens)-2
        mid = (lo+hi)//2
        center = list(range(max(lo, mid-2), min(hi, mid+3)))
        self.rng.shuffle(center)
        for pos in center:
            if is_good_blank_token(tokens[pos]):
                return pos, pos+1
        # fallback: cari dari kiri ke kanan
        for pos in range(lo, hi):
            if is_good_blank_token(tokens[pos]):
                return pos, pos+1
        # ultimate fallback: paksa tengah (meski jelek)
        pos = max(lo, min(hi-1, mid))
        return pos, pos+1

    def _build_question(self, pid:str, index:int, t_idx:int)->Question:
        title = self.titles_original[t_idx]
        ntoks = self.titles_norm[t_idx]
        start,end = self._select_blank_pos(ntoks)
        gold_norm = ntoks[start]

        orig_words = title.split()
        if len(orig_words)==len(ntoks):
            left = " ".join(orig_words[:start])
            blank = orig_words[start]
            right = " ".join(orig_words[end:])
        else:
            left = " ".join(ntoks[:start]); blank = gold_norm; right = " ".join(ntoks[end:])

        qid = f"q_{index:03d}"
        exp = time.time()+TIME_PER_QUESTION
        return Question(qid, pid, index, title, left, blank, right, 1, exp)

    def create_session(self, p1:str, p2:str)->SessionState:
        P = QUESTIONS_PER_PLAYER
        chosen = self._pick_questions(P*2)
        qs=[]
        for i in range(P):
            qs.append(self._build_question("p1", i+1, chosen[i]))
        for i in range(P):
            idx=P+i
            qs.append(self._build_question("p2", idx+1, chosen[idx]))

        sid=f"s_{int(time.time()*1000)}_{self.rng.randint(1000,9999)}"
        sess=SessionState(sid, [PlayerState("p1",p1), PlayerState("p2",p2)], qs, time.time())
        self.sessions[sid]=sess
        return sess

    def _score_bucket(self, left:list, right:list, player_word:str, gold_word:str):
        p = self.model.score_insert_word(left,right,player_word)
        g = self.model.score_insert_word(left,right,gold_word)
        r_vals = [self.model.score_insert_word(left,right,w) for w in self.model.sample_baseline(20)]
        r = statistics.median(r_vals) if r_vals else -8.0

        denom = g - r
        if abs(denom)<1e-6:
            if p>=g: bucket, z = 100, 1.0
            elif p>=r: bucket, z = 70, 0.6
            else: bucket, z = 30, 0.0
        else:
            z = clamp((p-r)/denom, 0.0, 1.0)
            bucket = 30 if z<0.25 else 50 if z<0.5 else 70 if z<0.8 else 100
        return bucket, p, g, r, z

    def answer(self, sid:str, qid:str, text:str)->Dict[str,Any]:
        sess=self.sessions.get(sid); 
        if not sess: return {"status":"error","message":"session not found"}
        q=next((x for x in sess.questions if x.question_id==qid), None)
        if not q: return {"status":"error","message":"question not found"}
        if q.answered_at>0: return {"status":"already_answered","score_bucket":q.score_bucket}

        now=time.time()
        if now>q.expires_at:
            q.answered_at=now; q.score_bucket=0
            next(p for p in sess.players if p.id==q.player_id).scores.append(0)
            return {"status":"timeout","score_bucket":0,"gold_text":q.blank_text}

        ans = tokenize(normalize_text_for_model(text))
        if len(ans)!=1: return {"status":"error","message":"Jawaban harus 1 kata."}
        player_word = ans[0]

        left = tokenize(normalize_text_for_model(q.left_text))
        right = tokenize(normalize_text_for_model(q.right_text))
        gold = tokenize(normalize_text_for_model(q.blank_text))[0] if q.blank_text else "<unk>"

        bucket,p,g,r,z = self._score_bucket(left,right,player_word,gold)
        q.answered_at=now; q.score_bucket=bucket
        next(ply for ply in sess.players if ply.id==q.player_id).scores.append(bucket)

        recon = (q.left_text+" " if q.left_text else "")+text+(" "+q.right_text if q.right_text else "")
        return {
            "status":"ok",
            "score_bucket":bucket,
            "raw":{"p_logprob_avg":round(p,4),"g_logprob_avg":round(g,4),"r_logprob_median":round(r,4),"z":round(z,4)},
            "gold_text":q.blank_text,
            "reconstructed_title":recon.strip(),
            "answered_at":now
        }

    def get_question(self, sid:str, index:int)->Dict[str,Any]:
        sess=self.sessions.get(sid)
        if not sess: return {"status":"error","message":"session not found"}
        if not (1<=index<=len(sess.questions)): return {"status":"error","message":"index out of range"}
        q=sess.questions[index-1]
        if q.answered_at==0: q.expires_at=time.time()+TIME_PER_QUESTION
        return {
            "status":"ok",
            "question_id":q.question_id,
            "player_id":q.player_id,
            "index":q.index,
            "title_left":q.left_text,
            "blank":"_____",
            "title_right":q.right_text,
            "blank_token_count":q.blank_token_count,
            "expires_at":q.expires_at,
            "display_title_full":f"{q.left_text} _____ {q.right_text}".strip()
        }

    def result(self, sid:str)->Dict[str,Any]:
        sess=self.sessions.get(sid)
        if not sess: return {"status":"error","message":"session not found"}
        res=[]; 
        for p in sess.players:
            res.append({"id":p.id,"name":p.name,"scores":p.scores,"total":sum(p.scores)})
        winner=None; is_tie=False
        if len(res)==2:
            if res[0]["total"]>res[1]["total"]: winner=res[0]["id"]
            elif res[1]["total"]>res[0]["total"]: winner=res[1]["id"]
            else: is_tie=True
        return {"players":res,"winner":winner,"is_tie":is_tie}

# =========================
# FastAPI & Schemas
# =========================
class SessionCreate(BaseModel):
    players: List[str] = Field(..., min_length=2, max_length=20, min_items=2, max_items=2)
    num_questions_per_player: int | None = None
    time_per_question_sec: int | None = None
    seed: int | None = None

class AnswerPost(BaseModel):
    session_id: str
    question_id: str
    answer_text: str = Field(..., min_length=1, max_length=40)
    client_time: float | None = None

app = FastAPI(title="News N-gram Game Backend (Single-File + UI)", default_response_class=ORJSONResponse)
engine = GameEngine()

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/session")
def create_session(payload: SessionCreate):
    global QUESTIONS_PER_PLAYER, TIME_PER_QUESTION, SEED, engine
    p = payload.players
    QUESTIONS_PER_PLAYER = payload.num_questions_per_player or QUESTIONS_PER_PLAYER
    TIME_PER_QUESTION = payload.time_per_question_sec or TIME_PER_QUESTION
    SEED = payload.seed or SEED
    random.seed(SEED)
    # Re-init engine agar SEED/waktu/parameter baru konsisten & eligible_indices dihitung ulang bila perlu
    engine = GameEngine()
    s = engine.create_session(p1=p[0], p2=p[1])
    return {"session_id":s.session_id,"players":[{"id":pl.id,"name":pl.name} for pl in s.players],"total_questions":len(s.questions)}

@app.get("/question")
def get_question(session_id: str = Query(...), index: int = Query(..., ge=1)):
    return engine.get_question(session_id, index)

@app.post("/answer")
def post_answer(payload: AnswerPost):
    return engine.answer(payload.session_id, payload.question_id, payload.answer_text)

@app.get("/result")
def get_result(session_id: str = Query(...)):
    return engine.result(session_id)

# =========================
# Simple UI (route "/")
# =========================
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>News N-gram Game</title>
<style>
  :root { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }
  body { margin: 0; background:#0b1020; color:#e9eef7; }
  .wrap { max-width: 820px; margin: 24px auto; padding: 16px; }
  .card { background:#171c31; border:1px solid #263154; border-radius:14px; padding:18px; box-shadow:0 6px 20px rgba(0,0,0,.25);}
  h1 { margin:8px 0 16px; font-size:24px; }
  label { display:block; margin:8px 0 6px; font-size:14px; color:#a9b7d9;}
  input[type=text]{ width:100%; padding:10px 12px; border-radius:10px; border:1px solid #31406d; background:#0e1428; color:#e9eef7;}
  button { background:#3b82f6; color:white; border:none; padding:10px 14px; border-radius:10px; cursor:pointer; font-weight:600;}
  button:disabled { opacity:.5; cursor:not-allowed;}
  .row{ display:flex; gap:12px; }
  .row > div { flex:1;}
  .titlebox{ font-size:20px; line-height:1.4; padding:14px; background:#0e1428; border:1px dashed #31406d; border-radius:10px; margin:10px 0;}
  .muted{ color:#9fb1d1; font-size:13px;}
  .mono{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
  .score { font-weight:700; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#263154; color:#cfe1ff; font-size:12px; }
  .timer { font-weight:800; font-size:18px; color:#ffd166; }
  .ok { color:#7bed9f; } .bad{ color:#f87171;}
  /* overlay switch player */
  .overlay {
    position: fixed; inset: 0; background: rgba(11,16,32,.9);
    display: none; align-items: center; justify-content: center; z-index: 50;
  }
  .overlay .panel {
    background:#0e1428; border:1px solid #31406d; border-radius:16px; padding:24px 28px; text-align:center;
    box-shadow:0 10px 30px rgba(0,0,0,.35);
  }
  .overlay h2{ margin:0 0 8px; font-size:22px; }
  .overlay .desc{ color:#a9b7d9; }
  .kv { margin-top:8px; font-size:13px; }
  .kv code{ background:#0b1020; padding:2px 6px; border-radius:6px; border:1px solid #263154; }
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>News N-gram Game</h1>
      <div id="step-setup">
        <div class="row">
          <div>
            <label>Nama Pemain 1</label>
            <input id="p1" type="text" placeholder="Player One" maxlength="20"/>
          </div>
          <div>
            <label>Nama Pemain 2</label>
            <input id="p2" type="text" placeholder="Player Two" maxlength="20"/>
          </div>
        </div>
        <div style="margin-top:12px;">
          <button id="btnStart">Start Game</button>
          <span class="muted">7 soal per pemain · <b>15 dtk/soal</b> · jawaban 1 kata</span>
        </div>
      </div>

      <div id="step-play" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
          <div>Giliran: <span id="who" class="pill">P1</span></div>
          <div>Soal <span id="qnum">1</span>/14 · <span class="timer" id="timer">15</span> dtk</div>
        </div>

        <div class="titlebox mono" id="titleBox">...</div>
        <form id="answerForm">
          <label>Jawab (1 kata)</label>
          <input id="answer" type="text" autocomplete="off" placeholder="ketik satu kata & Enter" maxlength="40"/>
          <div style="margin-top:10px;">
            <button id="btnSubmit" type="submit">Submit</button>
            <span class="muted">Enter = submit · Tidak ada tombol skip</span>
          </div>
        </form>

        <div id="feedback" style="margin-top:12px;"></div>
      </div>

      <div id="step-result" style="display:none;">
        <h3>Hasil Akhir</h3>
        <div id="resultBox" class="mono"></div>
        <div style="margin-top:12px;">
          <button onclick="location.reload()">Main Lagi</button>
        </div>
      </div>
    </div>
  </div>

  <!-- overlay switch player -->
  <div class="overlay" id="switchOverlay">
    <div class="panel">
      <h2 id="switchTitle">Giliran Pemain Berikutnya</h2>
      <div class="desc">Bersiap ya… game akan lanjut dalam <span id="switchCountdown">2</span> detik.</div>
    </div>
  </div>

<script>
const BASE = location.origin; // sama-origin

let sessionId = null;
let index = 1;          // 1..14
let currentQ = null;
let timerId = null;
let deadline = 0;

function $(id){ return document.getElementById(id); }
function playerLabel(i){ return (i<=7 ? "P1" : "P2"); }

function setStep(step){
  $("step-setup").style.display = step==="setup"?"block":"none";
  $("step-play").style.display  = step==="play" ?"block":"none";
  $("step-result").style.display= step==="result"?"block":"none";
}

async function startGame(){
  const p1 = $("p1").value.trim() || "Player One";
  const p2 = $("p2").value.trim() || "Player Two";
  const body = { players:[p1, p2], num_questions_per_player: 7, time_per_question_sec: 15 };
  const r = await fetch(BASE+"/session",{method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
  const js = await r.json();
  if(!js.session_id){ alert("Gagal membuat sesi"); return; }
  sessionId = js.session_id;
  index = 1;
  setStep("play");
  await loadQuestion();
}

async function loadQuestion(){
  clearInterval(timerId);
  $("feedback").innerHTML="";
  $("answer").value="";
  $("btnSubmit").disabled=false;
  $("who").textContent = playerLabel(index);
  $("qnum").textContent = index;

  const r = await fetch(`${BASE}/question?session_id=${encodeURIComponent(sessionId)}&index=${index}`);
  const js = await r.json();
  if(js.status!=="ok"){ alert(js.message || "Gagal ambil soal"); return; }
  currentQ = js;
  $("titleBox").textContent = js.display_title_full;

  // timer
  deadline = js.expires_at*1000;
  tick();
  timerId = setInterval(tick, 200);
}
function tick(){
  const left = Math.max(0, Math.ceil((deadline - Date.now())/1000));
  $("timer").textContent = left;
  if(left<=0){
    clearInterval(timerId);
    $("btnSubmit").disabled = true;
    $("feedback").innerHTML = `<div class="bad">Waktu habis. Jawaban: <b>${currentQ ? currentQ.blank : ""}</b></div>`;
    // lanjut otomatis setelah 1.2 detik
    setTimeout(nextStep, 1200);
  }
}

async function submitAnswer(ev){
  ev.preventDefault();
  if(!$("answer").value.trim()) return;
  $("btnSubmit").disabled=true;
  const payload = {
    session_id: sessionId,
    question_id: currentQ.question_id,
    answer_text: $("answer").value.trim()
  };
  const r = await fetch(BASE+"/answer",{method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
  const js = await r.json();

  if(js.status==="error"){
    $("feedback").innerHTML = `<div class="bad">${js.message}</div>`;
    $("btnSubmit").disabled=false;
    return;
  }
  if(js.status==="timeout"){
    $("feedback").innerHTML = `<div class="bad">Timeout. Gold: <b>${js.gold_text}</b></div>`;
  }else if(js.status==="already_answered"){
    $("feedback").innerHTML = `<div class="bad">Soal ini sudah dijawab.</div>`;
  }else{
    const cls = js.score_bucket>=70 ? "ok" : "bad";
    const raw = js.raw || {};
    // tampilkan skor n-gram
    $("feedback").innerHTML = `
      <div class="${cls}">Skor: <span class="score">${js.score_bucket}</span> · Gold: <b>${js.gold_text}</b></div>
      <div class="kv mono">
        <div>p_logprob_avg = <code>${raw.p_logprob_avg?.toFixed ? raw.p_logprob_avg.toFixed(4) : raw.p_logprob_avg}</code>,
             g_logprob_avg = <code>${raw.g_logprob_avg?.toFixed ? raw.g_logprob_avg.toFixed(4) : raw.g_logprob_avg}</code>,
             r_logprob_median = <code>${raw.r_logprob_median?.toFixed ? raw.r_logprob_median.toFixed(4) : raw.r_logprob_median}</code>,
             z = <code>${raw.z?.toFixed ? raw.z.toFixed(4) : raw.z}</code>
        </div>
      </div>`;
  }
  clearInterval(timerId);
  setTimeout(nextStep, 900);
}

function showSwitchOverlay(nextPlayerLabel){
  const ov = $("switchOverlay");
  const ttl = $("switchTitle");
  const cd = $("switchCountdown");
  ttl.textContent = `Giliran ${nextPlayerLabel}`;
  let remain = 2;
  cd.textContent = remain;
  ov.style.display = "flex";
  const iv = setInterval(()=>{
    remain -= 1;
    cd.textContent = Math.max(0, remain);
    if(remain <= 0){
      clearInterval(iv);
      ov.style.display = "none";
      loadQuestion();
    }
  }, 1000);
}

async function nextStep(){
  const prevPlayer = playerLabel(index);
  index++;
  if(index>14){
    // finish
    const r = await fetch(`${BASE}/result?session_id=${encodeURIComponent(sessionId)}`);
    const js = await r.json();
    const p = js.players || [];
    const line = p.map(pp=>`${pp.name}: [${pp.scores.join(", ")}]  total=${pp.total}`).join("<br>");
    const win = js.is_tie ? "<b>Seri!</b>" : `<b>Pemenang: ${ (p.find(x=>x.id===js.winner)||{}).name || "-" }</b>`;
    $("resultBox").innerHTML = line + "<br><br>" + win;
    setStep("result");
  } else {
    const nextPlayer = playerLabel(index);
    if(prevPlayer !== nextPlayer){
      // jeda pergantian pemain 2 detik
      showSwitchOverlay(nextPlayer);
    } else {
      await loadQuestion();
    }
  }
}

document.addEventListener("DOMContentLoaded", ()=>{
  setStep("setup");
  $("btnStart").onclick = startGame;
  $("answerForm").addEventListener("submit", submitAnswer);
  $("answer").addEventListener("keydown", (e)=>{ if(e.key==="Enter"){} }); // Enter = submit
});
</script>
</body>
</html>
    """

# Optional: run directly (local)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=True)
