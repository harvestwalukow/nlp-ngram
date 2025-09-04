# app.py — News N-gram Game (Backend + Simple UI) - Final Stable Version
from __future__ import annotations
import os, re, time, math, random, unicodedata, statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import ORJSONResponse, HTMLResponse
from pydantic import BaseModel, Field, constr

# =========================
# Config
# =========================
PORT = int(os.getenv("PORT", 8000))
NEWS_CSV = os.getenv("NEWS_CSV", "data/judul_fix.csv")
TIME_PER_QUESTION = int(os.getenv("TIME_PER_QUESTION", 10))
TIME_PER_QUESTION_HARD = int(os.getenv("TIME_PER_QUESTION_HARD", 15))
QUESTIONS_PER_PLAYER = int(os.getenv("QUESTIONS_PER_PLAYER", 7))
MIN_FREQ = int(os.getenv("MIN_FREQ", 2))

# =========================
# Utils
# =========================
_ws_re = re.compile(r"\s+")
_punct_re = re.compile(r"[^\w\- ]+", flags=re.UNICODE)
def normalize_text_for_model(s: str) -> str:
    s = s.lower(); s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _punct_re.sub(" ", s); s = _ws_re.sub(" ", s).strip()
    return s
def tokenize(s: str) -> List[str]:
    return s.split() if s else []
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# =========================
# Data
# =========================
def load_titles(csv_path: str = None) -> List[str]:
    path = csv_path or NEWS_CSV
    df = pd.read_csv(path); df.columns = [c.strip().lower() for c in df.columns]
    if "title" not in df.columns: raise ValueError("Kolom 'title' tidak ditemukan di CSV.")
    return df["title"].dropna().astype(str).tolist()

# =========================
# Stopwords & Validators
# =========================
ID_STOPWORDS = {"yang","dan","di","ke","dari","untuk","pada","dengan","atau","ini","itu","para","sebuah","sebagai","adalah","akan","jadi","agar","karena","oleh","dalam","kata","saat","tak","tidak","bukan","ya","pun","juga","sudah","belum","masih","hingga","sampai","serta","antara","tanpa","lebih","kurang","bisa","dapat","telah","bahwa","terhadap","guna","namun","tapi","tetapi","sementara","meski","meskipun","jika","apabila","bila","walau","walaupun","sebab","supaya","sehingga","sebelum","sesudah","ketika","kalau","ia","dia","mereka","kami","kita","saya","aku","anda","hari","bulan","tahun","besok","kemarin"}
def is_numeric_like(tok: str) -> bool:
    if tok.isdigit(): return True
    if re.fullmatch(r"\d+[\-/.]\d+([\-/\.]\d+)?", tok): return True
    if re.fullmatch(r"\d+[a-z]+", tok): return True
    return False
def is_good_blank_token(tok: str) -> bool:
    if not tok or tok in {"<s>", "</s>", "<unk>"} or tok in ID_STOPWORDS: return False
    if is_numeric_like(tok) or re.fullmatch(r"[\-]+", tok) or not re.search(r"[a-z]", tok): return False
    return True

# =========================
# KN Trigram Model
# =========================
UNK = "<unk>"
class KNTrigramModel:
    def __init__(self, min_freq: int = MIN_FREQ):
        self.min_freq = min_freq; self.unigram = {}; self.bigram = {}; self.trigram = {}
        self.continuation_unigram = {}; self.vocab = set(); self.total_unigrams = 0; self.D = 0.75
    def _replace_rare(self, toks_list: List[List[str]]) -> List[List[str]]:
        from collections import Counter
        flat = [t for toks in toks_list for t in toks]
        freq = Counter(flat)
        def map_tok(t): return t if freq[t] >= self.min_freq else UNK
        return [[map_tok(t) for t in toks] for toks in toks_list]
    def fit(self, raw_texts: List[str]):
        from collections import Counter, defaultdict
        tok_seqs = [tokenize(normalize_text_for_model(s)) for s in raw_texts]
        good_tex = [t for t in tok_seqs if t and len(t) >= 5 and sum(1 for x in t if is_good_blank_token(x)) / len(t) >= 0.7]
        tok_seqs = self._replace_rare(good_tex)
        seqs = [["<s>", "<s>"] + t + ["</s>"] for t in tok_seqs]
        uni, bi, tri = Counter(), Counter(), Counter()
        for seq in seqs:
            for w in seq: uni[w] += 1
            for i in range(len(seq)-1): bi[(seq[i], seq[i+1])] += 1
            for i in range(len(seq)-2): tri[(seq[i], seq[i+1], seq[i+2])] += 1
        self.unigram, self.bigram, self.trigram = dict(uni), dict(bi), dict(tri)
        self.vocab = set(self.unigram.keys()); self.total_unigrams = sum(self.unigram.values())
        unigram_ctx = defaultdict(set)
        for (w1,w2,w3), c in self.trigram.items(): unigram_ctx[w3].add((w1,w2))
        self.continuation_unigram = {w3: len(ctxs) for w3, ctxs in unigram_ctx.items()}
    def _p_cont(self, w:str) -> float: return self.continuation_unigram.get(w, 0) / max(1, len(self.bigram))
    def _pk_uni(self, w:str) -> float:
        p = self._p_cont(w)
        return p if p > 0 else self.unigram.get(w,0) / max(1,self.total_unigrams)
    def _pk_bi(self, w2:str, w3:str) -> float:
        c_bg, c_hist, types = self.bigram.get((w2,w3),0), 0, 0
        for (a,b),c in self.bigram.items():
            if a==w2: c_hist += c; types += 1 if c>0 else 0
        p_mle = max(c_bg-self.D,0)/max(1,c_hist) if c_bg>0 else 0.0
        lamb = (self.D*types)/max(1,c_hist) if c_hist>0 else 1.0
        return p_mle + lamb*self._pk_uni(w3)
    def _pk_tri(self, w1:str, w2:str, w3:str) -> float:
        c_tri, c_hist = self.trigram.get((w1,w2,w3),0), self.bigram.get((w1,w2),0)
        types = sum(1 for (a,b,c),cnt in self.trigram.items() if a==w1 and b==w2 and cnt>0)
        p_mle = max(c_tri-self.D,0)/max(1,c_hist) if c_tri>0 else 0.0
        lamb = (self.D*types)/max(1,c_hist) if c_hist>0 else 1.0
        return p_mle + lamb*self._pk_bi(w2,w3)
    def _tok(self,w:str)->str: return w if w in self.vocab else UNK
    def avg_logprob_window(self, tokens:List[str], start:int, end:int)->float:
        toks, off, L = ["<s>","<s>"]+[self._tok(t) for t in tokens]+["</s>"], 2, []
        for i in range(len(toks)-2):
            if not (i+2 < off+start or i > off+end-1):
                L.append(math.log(clamp(self._pk_tri(toks[i],toks[i+1],toks[i+2]), 1e-12, 1.0)))
        return sum(L)/len(L) if L else -12.0
    def score_insert_word(self, left:List[str], right:List[str], word:str)->float:
        return self.avg_logprob_window(left+[word]+right, len(left), len(left)+1)
    def generate_distractors(self, gold_word: str, k: int = 3, exclude: set = None) -> List[str]:
        distractors, pool = set(), [w for w in self.vocab if is_good_blank_token(w) and w != gold_word and (not exclude or w not in exclude)]
        if not pool: return [UNK] * k
        attempts = 0
        while len(distractors) < k and attempts < 10:
            sample_size = min(len(pool), (k - len(distractors)) * 2 + 1)
            for s in random.sample(pool, sample_size):
                if s not in distractors: distractors.add(s)
                if len(distractors) == k: break
            attempts += 1
        while len(distractors) < k and pool:
            word = pool.pop(random.randint(0, len(pool)-1))
            if word not in distractors: distractors.add(word)
        return list(distractors)

# =========================
# Game Engine
# =========================
@dataclass
class Question:
    question_id: str; player_id: str; index: int; title_original: str
    left_text: str; right_text: str; middle_text: str | None
    blank_texts: List[str]; expires_at: float
    is_hard: bool = False
    choices_sets: List[List[Dict[str, Any]]] = field(default_factory=list)
    answered_at: float = 0.0
    score_bucket: int | None = None

@dataclass
class PlayerState:
    id: str; name: str
    scores: List[int] = field(default_factory=list)
    streak_100: int = 0
    next_question_is_hard: bool = False

@dataclass
class SessionState:
    session_id: str; players: List[PlayerState]; questions: List[Question]; created_at: float

class GameEngine:
    def __init__(self):
        random.seed()
        self.rng = random.Random() 
        self.titles_original = load_titles(NEWS_CSV)
        self.titles_norm = [tokenize(normalize_text_for_model(t)) for t in self.titles_original]
        self.model = KNTrigramModel()
        print("Fitting N-gram model..."); self.model.fit(self.titles_original); print("Model fitting complete.")
        self.sessions: Dict[str, SessionState] = {}
        self.eligible_indices = [i for i, toks in enumerate(self.titles_norm) if self._has_good_blank(toks)]
        self.eligible_hard_indices = [i for i, toks in enumerate(self.titles_norm) if self._has_good_blank_for_hard(toks)]
        print(f"Found {len(self.eligible_indices)} eligible titles for normal questions.")
        print(f"Found {len(self.eligible_hard_indices)} eligible titles for hard questions.")

    def _has_good_blank(self, tokens: List[str]) -> bool: return len(tokens) >= 5 and any(is_good_blank_token(tok) for tok in tokens[1:-1])
    def _has_good_blank_for_hard(self, tokens: List[str]) -> bool: return len(tokens) >= 7 and len([i for i, tok in enumerate(tokens) if is_good_blank_token(tok)]) >= 2
    def _pick_question_indices(self, n:int, hard: bool = False)->List[int]:
        pool = self.eligible_hard_indices if hard else self.eligible_indices
        if len(pool) < n:
            pool.extend(self.eligible_indices)
            if len(pool) < n: raise ValueError(f"Dataset tidak cukup. Butuh {n}, tersedia {len(pool)}.")
        return self.rng.sample(pool, n)
    def _select_two_blank_pos(self, tokens: List[str]) -> Tuple[int, int]:
        good_indices = [i for i, tok in enumerate(tokens) if 1 <= i < len(tokens) - 1 and is_good_blank_token(tok)]
        pos1 = self.rng.choice(good_indices)
        good_indices2 = [i for i in good_indices if abs(i - pos1) > 1] or [i for i in good_indices if i != pos1]
        pos2 = self.rng.choice(good_indices2)
        return tuple(sorted((pos1, pos2)))
    def _generate_choices(self, left_ctx, right_ctx, gold_word, exclude=None):
        distractors = self.model.generate_distractors(gold_word, k=3, exclude=exclude)
        candidates = [gold_word] + distractors
        scored = [{'text': w, 'log_prob': self.model.score_insert_word(left_ctx, right_ctx, w)} for w in candidates]
        scored.sort(key=lambda x: x['log_prob'], reverse=True)
        scores = [100, 75, 50, 25]
        return [{'text': c['text'], 'score': scores[i]} for i, c in enumerate(scored)]
    def _build_normal_question(self, pid, index, t_idx):
        title, ntoks = self.titles_original[t_idx], self.titles_norm[t_idx]
        pos = self.rng.choice([i for i, t in enumerate(ntoks) if is_good_blank_token(t)] or [len(ntoks)//2])
        gold = ntoks[pos]
        choices = self._generate_choices(ntoks[:pos], ntoks[pos+1:], gold)
        orig_words = title.split()
        left, blank, right = (" ".join(ntoks[:pos]), gold, " ".join(ntoks[pos+1:]))
        if len(orig_words) == len(ntoks): left, blank, right = " ".join(orig_words[:pos]), orig_words[pos], " ".join(orig_words[pos+1:])
        qid = f"q_{pid}_{index:03d}_normal"
        expires = time.time() + TIME_PER_QUESTION
        return Question(qid, pid, index, title, left, right, None, [blank], expires, is_hard=False, choices_sets=[choices])
    def _build_hard_question(self, pid, index, t_idx):
        title, ntoks = self.titles_original[t_idx], self.titles_norm[t_idx]
        pos1, pos2 = self._select_two_blank_pos(ntoks)
        gold1, gold2 = ntoks[pos1], ntoks[pos2]
        choices1 = self._generate_choices(ntoks[:pos1], ntoks[pos1+1:pos2] + [gold2] + ntoks[pos2+1:], gold1)
        choices2 = self._generate_choices([gold1] + ntoks[:pos1] + ntoks[pos1+1:pos2], ntoks[pos2+1:], gold2, exclude={c['text'] for c in choices1})
        orig_words = title.split()
        left, mid, right = " ".join(ntoks[:pos1]), " ".join(ntoks[pos1+1:pos2]), " ".join(ntoks[pos2+1:])
        blanks = [gold1, gold2]
        if len(orig_words) == len(ntoks):
            left, mid, right = " ".join(orig_words[:pos1]), " ".join(orig_words[pos1+1:pos2]), " ".join(orig_words[pos2+1:])
            blanks = [orig_words[pos1], orig_words[pos2]]
        qid = f"q_{pid}_{index:03d}_hard"
        expires = time.time() + TIME_PER_QUESTION_HARD
        return Question(qid, pid, index, title, left, right, mid, blanks, expires, is_hard=True, choices_sets=[choices1, choices2])
    def create_session(self, p1:str, p2:str)->SessionState:
        sid=f"s_{int(time.time()*1000)}_{self.rng.randint(1000,9999)}"
        players = [PlayerState("p1",p1), PlayerState("p2",p2)]
        sess=SessionState(sid, players, [], time.time())
        self.sessions[sid]=sess
        return sess
    def answer(self, sid:str, qid:str, text:str)->Dict[str,Any]:
        sess, now = self.sessions.get(sid), time.time()
        if not sess: return {"status":"error", "message":"session not found"}
        q = next((x for x in sess.questions if x.question_id==qid), None)
        player = next((p for p in sess.players if p.id == q.player_id), None)
        if not q or not player: return {"status":"error", "message":"question or player not found"}
        if q.answered_at>0: return {"status":"already_answered", "score_bucket":q.score_bucket}
        
        q.answered_at = now; bucket = 0; is_timeout = now > q.expires_at or not text
        
        if is_timeout:
            player.streak_100 = 0
            if q.is_hard: player.next_question_is_hard = False
        elif q.is_hard:
            raw_ans1, raw_ans2 = text.split("||") if "||" in text else (None, None)
            ans1 = normalize_text_for_model(raw_ans1) if raw_ans1 else ""
            ans2 = normalize_text_for_model(raw_ans2) if raw_ans2 else ""
            correct_count = (1 if ans1 == q.blank_texts[0] else 0) + (1 if ans2 == q.blank_texts[1] else 0)
            if correct_count == 2: bucket = 100
            elif correct_count == 1: bucket = 50
            
            # ### PERUBAHAN LOGIKA DI SINI ###
            if bucket >= 50:
                player.next_question_is_hard = True
            else: # bucket == 0
                player.next_question_is_hard = False
            player.streak_100 = 0 # Streak selalu reset setelah soal sulit
        else: # Soal Normal
            normalized_text = normalize_text_for_model(text)
            chosen_option = next((c for c_set in q.choices_sets for c in c_set if c['text'] == normalized_text), None)
            if chosen_option:
                bucket = chosen_option['score']
                if bucket == 100: player.streak_100 += 1
                else: player.streak_100 = 0
                if player.streak_100 >= 3:
                    player.next_question_is_hard = True
                    player.streak_100 = 0
        
        q.score_bucket = bucket; player.scores.append(bucket)
        return { "status": "timeout" if is_timeout else "ok", "score_bucket": bucket, "blank_texts": q.blank_texts, "choices_sets": q.choices_sets, "is_hard": q.is_hard }
    def get_question(self, sid:str, index:int)->Dict[str,Any]:
        sess = self.sessions.get(sid)
        if not sess: return {"status":"error", "message":"session not found"}
        player_id = "p1" if index <= QUESTIONS_PER_PLAYER else "p2"
        player = next(p for p in sess.players if p.id == player_id)
        is_hard = player.next_question_is_hard
        t_idx = self._pick_question_indices(1, hard=is_hard)[0]
        question_num_for_player = index if player_id == "p1" else index - QUESTIONS_PER_PLAYER
        q = self._build_hard_question(player_id, question_num_for_player, t_idx) if is_hard else self._build_normal_question(player_id, question_num_for_player, t_idx)
        sess.questions.append(q)
        shuffled_choices1 = self.rng.sample(q.choices_sets[0], len(q.choices_sets[0]))
        shuffled_choices2 = self.rng.sample(q.choices_sets[1], len(q.choices_sets[1])) if q.is_hard else []
        display_title = f"{q.left_text} ____ {q.middle_text or ''} {'____' if q.is_hard else ''} {q.right_text}".strip().replace("  ", " ")
        return {"status": "ok", "question_id": q.question_id, "player_id": q.player_id, "index": index, "display_title_full": display_title, "choices_sets": [[{'text': c['text']} for c in shuffled_choices1], [{'text': c['text']} for c in shuffled_choices2]], "expires_at": q.expires_at, "is_hard": q.is_hard}
    def result(self, sid:str)->Dict[str,Any]:
        sess=self.sessions.get(sid)
        if not sess: return {"status":"error","message":"session not found"}
        res=[{"id":p.id,"name":p.name,"scores":p.scores,"total":sum(p.scores)} for p in sess.players]
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
    players: List[str] = Field(..., min_length=1, max_length=20, min_items=2, max_items=2)
class AnswerPost(BaseModel): session_id: str; question_id: str; answer_text: str

app = FastAPI(title="News N-gram Game Backend (Dynamic Difficulty)", default_response_class=ORJSONResponse)
engine = GameEngine()

@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/session")
def create_session(payload: SessionCreate):
    global engine; engine = GameEngine() 
    s = engine.create_session(p1=payload.players[0], p2=payload.players[1])
    return {"session_id":s.session_id,"players":[{"id":pl.id,"name":pl.name} for pl in s.players],"total_questions":QUESTIONS_PER_PLAYER*2}
@app.get("/question")
def get_question(session_id: str = Query(...), index: int = Query(..., ge=1)): return engine.get_question(session_id, index)
@app.post("/answer")
def post_answer(payload: AnswerPost): return engine.answer(payload.session_id, payload.question_id, payload.answer_text)
@app.get("/result")
def get_result(session_id: str = Query(...)): return engine.result(session_id)

# =========================
# Simple UI (route "/")
# =========================
@app.get("/", response_class=HTMLResponse)
def ui():
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>News N-gram Game (Dynamic)</title>
<style>
  :root {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  body {{ margin: 0; background:#0b1020; color:#e9eef7; }}
  .wrap {{ max-width: 820px; margin: 24px auto; padding: 16px; }}
  .card {{ background:#171c31; border:1px solid #263154; border-radius:14px; padding:18px; box-shadow:0 6px 20px rgba(0,0,0,.25);}}
  h1, h3 {{ margin:8px 0 16px; }}
  label {{ display:block; margin:8px 0 6px; font-size:14px; color:#a9b7d9;}}
  input[type=text]{{ width:100%; box-sizing: border-box; padding:10px 12px; border-radius:10px; border:1px solid #31406d; background:#0e1428; color:#e9eef7;}}
  button {{ background:#3b82f6; color:white; border:none; padding:10px 14px; border-radius:10px; cursor:pointer; font-weight:600;}}
  button:disabled {{ opacity:.5; cursor:not-allowed;}}
  .row{{ display:flex; gap:12px; }} .row > div {{ flex:1;}}
  .titlebox{{ font-size:20px; line-height:1.4; padding:14px; background:#0e1428; border:1px dashed #31406d; border-radius:10px; margin:10px 0;}}
  .muted{{ color:#9fb1d1; font-size:13px;}}
  .mono{{ font-family: monospace; }}
  .pill {{ display:inline-block; padding:2px 8px; border-radius:999px; background:#263154; color:#cfe1ff; font-size:12px; }}
  .timer {{ font-weight:800; font-size:18px; color:#ffd166; }}
  .ok {{ color:#7bed9f; }} .bad{{ color:#f87171;}} .special{{color:#f59e0b; font-weight:bold;}}
  .choicesContainer {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }}
  .choiceBtn {{ background: #263154; padding: 12px; text-align: center; font-size: 16px; width: 100%; box-sizing: border-box; }}
  .choiceBtn:not(:disabled):hover {{ background: #31406d; }}
  .choiceBtn.selected {{ background: #f59e0b; }}
  .choiceBtn.correct {{ background: #22c55e; }}
  .choiceBtn.incorrect {{ background: #ef4444; opacity: 0.7; }}
  
  /* New Two-Player Layout Styles */
  .game-container {{ 
    display: flex; 
    gap: 20px; 
    align-items: flex-start; 
    min-height: 400px;
    position: relative;
  }}
  .player-section {{ 
    flex: 1; 
    background: #171c31; 
    border: 1px solid #263154; 
    border-radius: 14px; 
    padding: 18px; 
    display: flex; 
    flex-direction: column;
    min-height: 350px;
    transition: all 0.3s ease;
  }}
  .player-section.active {{ 
    border: 2px solid #3b82f6; 
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3);
  }}
  .player-section.inactive {{ 
    opacity: 0.6;
  }}
  .player-title {{ 
    font-size: 18px; 
    font-weight: bold; 
    margin-bottom: 12px; 
    color: #e9eef7;
  }}
  .player-score {{ 
    font-size: 14px; 
    color: #a9b7d9; 
    margin-bottom: 16px;
    padding: 8px;
    background: #0e1428;
    border-radius: 8px;
    text-align: center;
  }}
  .player-question {{ 
    font-size: 16px; 
    line-height: 1.4; 
    padding: 14px; 
    background: #0e1428; 
    border: 1px dashed #31406d; 
    border-radius: 10px; 
    margin-bottom: 16px;
    flex-grow: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
  }}
  .player-choices {{ 
    margin-top: auto;
  }}
  .choice-grid {{ 
    display: grid; 
    grid-template-columns: 1fr 1fr; 
    gap: 12px; 
  }}
  .choice-btn {{ 
    background: #263154; 
    color: #e9eef7; 
    border: 1px solid #31406d; 
    border-radius: 8px; 
    padding: 12px 8px; 
    text-align: center; 
    font-size: 14px; 
    cursor: pointer; 
    transition: all 0.2s ease;
    min-height: 40px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .choice-btn:hover:not(:disabled) {{ 
    background: #31406d; 
    border-color: #3b82f6;
  }}
  .choice-btn:disabled {{ 
    opacity: 0.5; 
    cursor: not-allowed; 
  }}
  .choice-btn.correct {{ 
    background: #22c55e; 
    border-color: #16a34a;
  }}
  .choice-btn.incorrect {{ 
    background: #ef4444; 
    border-color: #dc2626;
    opacity: 0.7;
  }}
  .central-elements {{ 
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    justify-content: center; 
    min-width: 120px;
    padding: 0 10px;
  }}
  .time-display {{ 
    background: #263154; 
    color: #e9eef7; 
    padding: 12px; 
    border-radius: 50%; 
    width: 60px; 
    height: 60px; 
    display: flex; 
    align-items: center; 
    justify-content: center; 
    font-weight: bold; 
    font-size: 14px;
    margin-bottom: 16px;
  }}
  .question-counter {{ 
    background: #0e1428; 
    color: #a9b7d9; 
    padding: 8px 12px; 
    border-radius: 12px; 
    font-size: 12px; 
    text-align: center;
    border: 1px solid #31406d;
  }}
  .current-player {{ 
    background: #3b82f6; 
    color: white; 
    padding: 6px 12px; 
    border-radius: 12px; 
    font-size: 12px; 
    font-weight: bold;
    text-align: center;
    margin-top: 8px;
  }}
  #hardQuestionControls {{ display: none; text-align: center; margin-top: 10px; }}
  .hard-choices {{ border: 1px solid #4a5568; padding: 8px; border-radius: 8px; margin-bottom: 8px;}}
  .hard-choices-label {{ font-size: 12px; color: #a9b7d9; margin-bottom: 4px; }}
  .overlay {{ position: fixed; inset: 0; background: rgba(11,16,32,.9); display: none; align-items: center; justify-content: center; z-index: 50; }}
  .overlay .panel {{ background:#0e1428; border:1px solid #31406d; border-radius:16px; padding:24px 28px; text-align:center; box-shadow:0 10px 30px rgba(0,0,0,.35); }}
  .overlay h2{{ margin:0 0 8px; font-size:22px; }} .overlay .desc{{ color:#a9b7d9; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>News N-gram Game</h1>
      <div id="step-setup">
        <div class="row">
          <div> <label>Nama Pemain 1</label> <input id="p1" type="text" value="Pemain 1" maxlength="20"/> </div>
          <div> <label>Nama Pemain 2</label> <input id="p2" type="text" value="Pemain 2" maxlength="20"/> </div>
        </div>
        <div style="margin-top:12px;"> <button id="btnStart">Start Game</button> <span id="setup-info" class="muted"></span> </div>
      </div>
      <div id="step-play" style="display:none;">
        <div id="difficulty-notice" class="special" style="text-align:center; margin-bottom: 16px;"></div>
        
        <!-- Two Player Split Layout -->
        <div class="game-container">
          <!-- Player 1 Section (Left) -->
          <div class="player-section" id="player1-section">
            <div class="player-title" id="player1-title">Pemain 1</div>
            <div class="player-score" id="player1-score">[skor disini]</div>
            <div class="player-question" id="player1-question">Soal disini</div>
            <div class="player-choices" id="player1-choices">
              <div class="choice-grid">
                <button class="choice-btn" data-choice="a">a</button>
                <button class="choice-btn" data-choice="b">b</button>
                <button class="choice-btn" data-choice="c">c</button>
                <button class="choice-btn" data-choice="d">d</button>
              </div>
            </div>
          </div>
          
          <!-- Central Elements -->
          <div class="central-elements">
            <div class="time-display" id="timer">waktu</div>
            <div class="question-counter" id="qnum">1/7 soal</div>
            <div class="current-player" id="currentPlayer">Pemain 1</div>
          </div>
          
          <!-- Player 2 Section (Right) -->
          <div class="player-section" id="player2-section">
            <div class="player-title" id="player2-title">Pemain 2</div>
            <div class="player-score" id="player2-score">[skor disini]</div>
            <div class="player-question" id="player2-question">Soal disini</div>
            <div class="player-choices" id="player2-choices">
              <div class="choice-grid">
                <button class="choice-btn" data-choice="a">a</button>
                <button class="choice-btn" data-choice="b">b</button>
                <button class="choice-btn" data-choice="c">c</button>
                <button class="choice-btn" data-choice="d">d</button>
              </div>
            </div>
          </div>
        </div>
        
        <!-- Hidden elements for hard questions -->
        <div id="hardQuestionControls" style="display:none;">
            <div class="hard-choices">
              <div class="hard-choices-label">PILIHAN KATA PERTAMA</div>
              <div id="choicesBox2" class="choicesContainer"></div>
            </div>
            <div class="hard-choices">
              <div class="hard-choices-label">PILIHAN KATA KEDUA</div>
              <div id="choicesBox3" class="choicesContainer"></div>
            </div>
            <button id="submitHard" disabled>Submit Jawaban Ganda</button>
        </div>
        <div id="feedback" style="margin-top:16px;"></div>
      </div>
      <div id="step-result" style="display:none;">
        <h3>Hasil Akhir</h3>
        <div id="resultBox" class="mono"></div>
        <div style="margin-top:12px;"> <button onclick="location.reload()">Main Lagi</button> </div>
      </div>
    </div>
  </div>
  <div class="overlay" id="switchOverlay">
    <div class="panel"> <h2 id="switchTitle"></h2> <div class="desc">Bersiap... game akan lanjut dalam <span id="switchCountdown"></span> detik.</div> </div>
  </div>

<script>
const TRANSITION_TIME = 5;
const QUESTIONS_PER_PLAYER = {QUESTIONS_PER_PLAYER};
const TIME_PER_QUESTION = {TIME_PER_QUESTION};
const TOTAL_QUESTIONS = QUESTIONS_PER_PLAYER * 2; 
let sessionId = null, index = 1, currentQ = null, timerId = null, deadline = 0;
let playerNames = {{}}, selections = {{}};
let playerScores = {{ p1: [], p2: [] }};

function $(id){{ return document.getElementById(id); }}
function setStep(step){{
  $("step-setup").style.display = step==="setup"?"block":"none";
  $("step-play").style.display  = step==="play" ?"block":"none";
  $("step-result").style.display= step==="result"?"block":"none";
}}
async function startGame(){{
  try {{
    const p1 = $("p1").value.trim() || "Pemain 1";
    const p2 = $("p2").value.trim() || "Pemain 2";
    playerNames = {{ p1, p2 }};
    
    // Initialize scores
    playerScores = {{ p1: [], p2: [] }};
    updateScores();
    
    console.log("Starting game with players:", playerNames);
    
    const body = {{ players:[p1, p2] }};
    const r = await fetch("/session",{{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
    const js = await r.json();
    
    console.log("Session response:", js);
    
    if(!js.session_id){{ 
      alert("Gagal membuat sesi: " + (js.message || "Unknown error")); 
      return; 
    }}
    
    sessionId = js.session_id; 
    index = 1;
    setStep("play");
    await loadQuestion();
  }} catch (error) {{
    console.error("Error starting game:", error);
    alert("Error starting game: " + error.message);
  }}
}}
async function loadQuestion(){{
  try {{
    clearInterval(timerId);
    $("feedback").innerHTML = ""; $("difficulty-notice").innerHTML = ""; selections = {{}};
    
    const currentPlayer = index <= TOTAL_QUESTIONS/2 ? 'p1' : 'p2';
    const currentPlayerName = playerNames[currentPlayer];
    
    console.log("Loading question", index, "for player", currentPlayerName);
    
    // Update central elements
    $("qnum").textContent = `${{index}}/${{TOTAL_QUESTIONS}}`;
    $("currentPlayer").textContent = currentPlayerName;
    
    // Update player sections
    updatePlayerSections(currentPlayer);

    const r = await fetch(`/question?session_id=${{sessionId}}&index=${{index}}`);
    const js = await r.json();
    
    console.log("Question response:", js);
    
    if(js.status!=="ok"){{ 
      alert("Error loading question: " + (js.message || "Unknown error")); 
      return; 
    }}
    currentQ = js;
  
  // Update the active player's question
  const activePlayerSection = currentPlayer === 'p1' ? 'player1' : 'player2';
  $(`${{activePlayerSection}}-question`).textContent = js.display_title_full;
  
  // Clear all choice buttons and reset their state
  document.querySelectorAll('.choice-btn').forEach(btn => {{
    btn.disabled = false;
    btn.className = 'choice-btn';
    btn.onclick = null;
  }});
  
  // Hide hard question controls initially
  $("hardQuestionControls").style.display = "none";

  if (js.is_hard) {{
      $("difficulty-notice").textContent = "🔥 SOAL SULIT: Pilih 2 Kata! 🔥";
      $("hardQuestionControls").style.display = "block";
      // Clear previous hard question choices
      ["choicesBox2", "choicesBox3"].forEach(id => $(id).innerHTML = "");
      js.choices_sets[0].forEach(c => createChoiceButton(c.text, 1, "choicesBox2"));
      js.choices_sets[1].forEach(c => createChoiceButton(c.text, 2, "choicesBox3"));
  }} else {{
      // Update choice buttons for normal questions
      const choiceButtons = document.querySelectorAll(`#${{activePlayerSection}}-choices .choice-btn`);
      js.choices_sets[0].forEach((choice, idx) => {{
        if (choiceButtons[idx]) {{
          choiceButtons[idx].textContent = choice.text;
          choiceButtons[idx].onclick = () => submitAnswer(choice.text);
        }}
      }});
  }}
    deadline = js.expires_at*1000;
    tick();
    timerId = setInterval(tick, 200);
  }} catch (error) {{
    console.error("Error loading question:", error);
    alert("Error loading question: " + error.message);
  }}
}}

function updatePlayerSections(activePlayer) {{
  // Update player titles and visual states
  $("player1-title").textContent = `Pemain 1${{activePlayer === 'p1' ? ' (Aktif)' : ''}}`;
  $("player2-title").textContent = `Pemain 2${{activePlayer === 'p2' ? ' (Aktif)' : ''}}`;
  
  // Update visual states
  $("player1-section").className = `player-section${{activePlayer === 'p1' ? ' active' : ' inactive'}}`;
  $("player2-section").className = `player-section${{activePlayer === 'p2' ? ' active' : ' inactive'}}`;
  
  // Update scores
  updateScores();
  
  // Reset question displays for inactive player
  if (activePlayer === 'p1') {{
    $("player2-question").textContent = "Menunggu giliran...";
  }} else {{
    $("player1-question").textContent = "Menunggu giliran...";
  }}
}}
function createChoiceButton(text, group, containerId) {{
    const btn = document.createElement("button");
    btn.className = "choiceBtn"; btn.textContent = text;
    btn.onclick = () => {{
        if (currentQ.is_hard) {{ handleHardSelection(btn, group); }}
        else {{ submitAnswer(text); }}
    }};
    $(containerId).appendChild(btn);
}}
function handleHardSelection(btn, group) {{
    const containerId = group === 1 ? "choicesBox2" : "choicesBox3";
    $(containerId).querySelectorAll('.choiceBtn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    selections[group] = btn.textContent;
    $("submitHard").disabled = !(selections[1] && selections[2]);
}}
// Initialize hard question submit handler
function initHardQuestionHandler() {{
  const submitBtn = $("submitHard");
  if (submitBtn) {{
    submitBtn.onclick = () => {{
      const combinedAnswer = `${{selections[1]}}||${{selections[2]}}`;
      submitAnswer(combinedAnswer);
    }};
  }}
}}
async function submitAnswer(answerText){{
  // Disable all buttons
  document.querySelectorAll('.choice-btn').forEach(b => b.disabled = true);
  document.querySelectorAll('.choiceBtn').forEach(b => b.disabled = true);
  $("submitHard").disabled = true;
  
  const isTimeout = !answerText;
  const payload = {{ session_id: sessionId, question_id: currentQ.question_id, answer_text: isTimeout ? "" : answerText }};
  const r = await fetch("/answer",{{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(payload)}});
  const js = await r.json();
  clearInterval(timerId);
  
  if (js.status === "ok" || js.status === "timeout") {{
    const score = js.score_bucket;
    const currentPlayer = index <= TOTAL_QUESTIONS/2 ? 'p1' : 'p2';
    
    // Update scores
    playerScores[currentPlayer].push(score);
    updateScores();
    
    let feedbackHTML = `<div class="${{score >= 50 ? 'ok' : 'bad'}}">Skor: <b class="score">${{score}}</b></div>`;
    if (js.is_hard) {{
        feedbackHTML += `<div>Jawaban Benar: <b>${{js.blank_texts[0]}}</b> & <b>${{js.blank_texts[1]}}</b></div>`;
        highlightHardChoices(js.blank_texts);
    }} else {{
        feedbackHTML += `<div>Jawaban Benar: <b>${{js.blank_texts[0]}}</b></div>`;
        highlightNormalChoices(js.blank_texts[0]);
    }}
    $("feedback").innerHTML = feedbackHTML;
  }} else {{
     $("feedback").innerHTML = `<div class="bad">${{js.message || "Terjadi kesalahan"}}</div>`;
  }}
  setTimeout(nextStep, 3500);
}}
function highlightNormalChoices(goldText) {{
    const currentPlayer = index <= TOTAL_QUESTIONS/2 ? 'p1' : 'p2';
    const activePlayerSection = currentPlayer === 'p1' ? 'player1' : 'player2';
    $(`${{activePlayerSection}}-choices`).querySelectorAll('.choice-btn').forEach(b => {{
        if (b.textContent === goldText) b.classList.add('correct');
        else if (b.classList.contains('selected')) b.classList.add('incorrect');
    }});
}}
function highlightHardChoices(goldTexts) {{
    [1, 2].forEach((group, i) => {{
        const containerId = `choicesBox${{group+1}}`;
        $(containerId).querySelectorAll('.choiceBtn').forEach(b => {{
            if (b.textContent === goldTexts[i]) b.classList.add('correct');
            else if (b.classList.contains('selected')) b.classList.add('incorrect');
        }});
    }});
}}
function showSwitchOverlay(nextPlayerName){{
  const ov = $("switchOverlay");
  $("switchTitle").textContent = `Giliran ${{nextPlayerName}}`;
  let remain = TRANSITION_TIME;
  $("switchCountdown").textContent = remain;
  ov.style.display = "flex";
  const iv = setInterval(()=>{{
    remain -= 1;
    $("switchCountdown").textContent = Math.max(0, remain);
    if(remain <= 0){{
      clearInterval(iv);
      ov.style.display = "none";
      loadQuestion();
    }}
  }}, 1000);
}}
async function nextStep(){{
  const prevPlayerId = index <= TOTAL_QUESTIONS/2 ? 'p1' : 'p2';
  index++;
  if(index > TOTAL_QUESTIONS){{
    const r = await fetch(`/result?session_id=${{sessionId}}`);
    const js = await r.json();
    const p = js.players || [];
    let resultHTML = p.map(pp => `${{pp.name}}: [${{pp.scores.join(", ")}}] &nbsp; <b>Total = ${{pp.total}}</b>`).join("<br>");
    const winnerName = (p.find(x => x.id === js.winner) || {{}}).name;
    const win = js.is_tie ? "<b>Hasilnya Seri!</b>" : `<b>Pemenang: ${{winnerName || "-"}}</b>`;
    $("resultBox").innerHTML = resultHTML + "<br><br>" + win;
    setStep("result");
  }} else {{
    const nextPlayerId = index <= TOTAL_QUESTIONS/2 ? 'p1' : 'p2';
    if(prevPlayerId !== nextPlayerId){{
      showSwitchOverlay(playerNames[nextPlayerId]);
    }} else {{
      await loadQuestion();
    }}
  }}
}}
function tick(){{
  const left = Math.max(0, Math.ceil((deadline - Date.now())/1000));
  $("timer").textContent = left;
  if(left<=0){{
    clearInterval(timerId);
    submitAnswer(null);
  }}
}}

function updateScores() {{
  const p1Total = playerScores.p1.reduce((a, b) => a + b, 0);
  const p2Total = playerScores.p2.reduce((a, b) => a + b, 0);
  $("player1-score").textContent = `Skor: ${{p1Total}}`;
  $("player2-score").textContent = `Skor: ${{p2Total}}`;
}}
document.addEventListener("DOMContentLoaded", ()=>{{
  $("setup-info").innerHTML = `${{QUESTIONS_PER_PLAYER}} soal per pemain · <b>${{TIME_PER_QUESTION}} dtk/soal</b>`;
  setStep("setup");
  $("btnStart").onclick = startGame;
  initHardQuestionHandler();
}});
</script>
</body>
</html>
    """

# For Vercel deployment
app = app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)