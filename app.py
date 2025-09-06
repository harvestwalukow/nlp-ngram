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
# PENINGKATAN KUALITAS SOAL: Atur panjang minimal kalimat
MIN_SENTENCE_LENGTH = 8

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
    session_id: str; players: List[PlayerState]; questions: List[Question]; created_at: float; mode: str

class GameEngine:
    def __init__(self):
        random.seed()
        self.rng = random.Random() 
        self.titles_original = load_titles(NEWS_CSV)
        self.titles_norm = [tokenize(normalize_text_for_model(t)) for t in self.titles_original]
        self.model = KNTrigramModel()
        print("Fitting N-gram model..."); self.model.fit(self.titles_original); print("Model fitting complete.")
        self.sessions: Dict[str, SessionState] = {}
        # PENINGKATAN KUALITAS SOAL: Filter judul yang memenuhi syarat di awal
        self.eligible_indices = [i for i, toks in enumerate(self.titles_norm) if self._is_eligible_for_game(toks)]
        self.eligible_hard_indices = [i for i, toks in enumerate(self.titles_norm) if self._is_eligible_for_game(toks, hard=True)]
        print(f"Found {len(self.eligible_indices)} eligible titles for normal questions.")
        print(f"Found {len(self.eligible_hard_indices)} eligible titles for hard questions.")

    def _get_valid_blank_indices(self, tokens: List[str]) -> List[int]:
        n = len(tokens)
        # Kata rumpang tidak boleh di 2 kata pertama atau terakhir. Indeks valid: 2 s/d n-3
        # Contoh: 8 kata (indeks 0-7), posisi valid 2, 3, 4, 5
        if n < MIN_SENTENCE_LENGTH:
            return []
        
        valid_indices = []
        # Loop dari indeks ke-2 (posisi ketiga) hingga sebelum 2 kata terakhir
        for i in range(2, n - 2):
            if is_good_blank_token(tokens[i]):
                valid_indices.append(i)
        return valid_indices

    def _is_eligible_for_game(self, tokens: List[str], hard: bool = False) -> bool:
        if len(tokens) < MIN_SENTENCE_LENGTH:
            return False
        
        valid_indices = self._get_valid_blank_indices(tokens)
        if hard:
            # Untuk soal sulit, butuh minimal 2 posisi valid yang tidak bersebelahan
            if len(valid_indices) < 2:
                return False
            # Cek apakah ada setidaknya satu pasang yang tidak bersebelahan
            for i in range(len(valid_indices)):
                for j in range(i + 1, len(valid_indices)):
                    if valid_indices[j] > valid_indices[i] + 1:
                        return True
            return False
        else:
            return bool(valid_indices)

    def _pick_question_indices(self, n:int, hard: bool = False)->List[int]:
        pool = self.eligible_hard_indices if hard else self.eligible_indices
        if len(pool) < n:
            pool.extend(self.eligible_indices) # Fallback jika soal sulit habis
            if len(pool) < n: raise ValueError(f"Dataset tidak cukup. Butuh {n}, tersedia {len(pool)}.")
        return self.rng.sample(pool, n)

    def _select_two_blank_pos(self, tokens: List[str]) -> Tuple[int, int]:
        good_indices = self._get_valid_blank_indices(tokens)
        pos1 = self.rng.choice(good_indices)
        # Pilih posisi kedua yang tidak bersebelahan
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
    
    def _build_uraian_question(self, pid, index, t_idx):
        title, ntoks = self.titles_original[t_idx], self.titles_norm[t_idx]
        valid_pos = self._get_valid_blank_indices(ntoks)
        pos = self.rng.choice(valid_pos) if valid_pos else len(ntoks) // 2
        gold = ntoks[pos]
        orig_words = title.split()
        left, blank, right = (" ".join(ntoks[:pos]), gold, " ".join(ntoks[pos+1:]))
        if len(orig_words) == len(ntoks): left, blank, right = " ".join(orig_words[:pos]), orig_words[pos], " ".join(orig_words[pos+1:])
        qid = f"q_{pid}_{index:03d}_uraian"
        expires = time.time() + TIME_PER_QUESTION
        return Question(qid, pid, index, title, left, right, None, [blank], expires)

    def _build_normal_question(self, pid, index, t_idx):
        title, ntoks = self.titles_original[t_idx], self.titles_norm[t_idx]
        valid_pos = self._get_valid_blank_indices(ntoks)
        pos = self.rng.choice(valid_pos) if valid_pos else len(ntoks) // 2
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

    def create_session(self, p1:str, p2:str, mode:str)->SessionState:
        sid=f"s_{int(time.time()*1000)}_{self.rng.randint(1000,9999)}"
        players = [PlayerState("p1",p1), PlayerState("p2",p2)]
        sess=SessionState(sid, players, [], time.time(), mode=mode)
        self.sessions[sid]=sess
        return sess
        
    def _score_uraian_answer(self, left_ctx, right_ctx, player_word, gold_word):
        p_score = self.model.score_insert_word(left_ctx, right_ctx, player_word)
        g_score = self.model.score_insert_word(left_ctx, right_ctx, gold_word)
        r_scores = [self.model.score_insert_word(left_ctx, right_ctx, w) for w in self.model.generate_distractors(gold_word, k=20)]
        r_median = statistics.median(r_scores) if r_scores else -10.0
        denominator = g_score - r_median
        if abs(denominator) < 1e-6: return 100 if p_score >= g_score else 30
        z = clamp((p_score - r_median) / denominator, 0.0, 1.0)
        return 30 if z < 0.25 else 50 if z < 0.5 else 70 if z < 0.8 else 100

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
        
        elif sess.mode == 'uraian':
            player_word = normalize_text_for_model(text.split()[0] if text else "")
            gold_word = q.blank_texts[0]
            left_ctx = tokenize(normalize_text_for_model(q.left_text))
            right_ctx = tokenize(normalize_text_for_model(q.right_text))
            bucket = self._score_uraian_answer(left_ctx, right_ctx, player_word, gold_word)

        elif sess.mode == 'pilihan_ganda':
            if q.is_hard:
                raw_ans1, raw_ans2 = text.split("||") if "||" in text else (None, None)
                ans1 = normalize_text_for_model(raw_ans1) if raw_ans1 else ""
                ans2 = normalize_text_for_model(raw_ans2) if raw_ans2 else ""
                correct_count = (1 if ans1 == q.blank_texts[0] else 0) + (1 if ans2 == q.blank_texts[1] else 0)
                if correct_count == 2: bucket = 100
                elif correct_count == 1: bucket = 50
                
                if bucket >= 50: player.next_question_is_hard = True
                else: player.next_question_is_hard = False
                player.streak_100 = 0
            else:
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
        question_num_for_player = index if player_id == "p1" else index - QUESTIONS_PER_PLAYER
        
        if sess.mode == 'uraian':
            t_idx = self._pick_question_indices(1, hard=False)[0]
            q = self._build_uraian_question(player_id, question_num_for_player, t_idx)
        elif sess.mode == 'pilihan_ganda':
            is_hard = player.next_question_is_hard
            t_idx = self._pick_question_indices(1, hard=is_hard)[0]
            q = self._build_hard_question(player_id, question_num_for_player, t_idx) if is_hard else self._build_normal_question(player_id, question_num_for_player, t_idx)
        else:
            return {"status":"error", "message":"Mode tidak valid"}
        
        sess.questions.append(q)

        shuffled_choices1 = self.rng.sample(q.choices_sets[0], len(q.choices_sets[0])) if q.choices_sets and q.choices_sets[0] else []
        shuffled_choices2 = self.rng.sample(q.choices_sets[1], len(q.choices_sets[1])) if q.is_hard and len(q.choices_sets) > 1 else []
        display_title = f"{q.left_text} ____ {q.middle_text or ''} {'____' if q.is_hard else ''} {q.right_text}".strip().replace("  ", " ")
        
        return {"status": "ok", "question_id": q.question_id, "player_id": q.player_id, "index": index, "display_title_full": display_title, "choices_sets": [[{'text': c['text']} for c in shuffled_choices1], [{'text': c['text']} for c in shuffled_choices2]], "expires_at": q.expires_at, "is_hard": q.is_hard, "mode": sess.mode}

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
    players: List[constr(min_length=1, max_length=20)] = Field(..., min_items=2, max_items=2)
    mode: str
class AnswerPost(BaseModel): session_id: str; question_id: str; answer_text: str

app = FastAPI(title="News N-gram Game (Multi-Mode)", default_response_class=ORJSONResponse)
engine = GameEngine()

@app.get("/health")
def health(): return {"status":"ok"}
@app.post("/session")
def create_session(payload: SessionCreate):
    global engine; engine = GameEngine() 
    s = engine.create_session(p1=payload.players[0], p2=payload.players[1], mode=payload.mode)
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
<title>News N-gram Game (Multi-Mode)</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
  :root {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  body {{ 
    margin: 0; 
    background: #ffffff; 
    color: #000000; 
    min-height: 100vh;
    transition: background 0.8s ease-in-out, color 0.8s ease-in-out;
    position: relative;
    overflow-x: hidden;
  }}
  
  body::before {{
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(68deg, #f784c5 0%, #1b602f 100%);
    opacity: 0;
    transition: opacity 0.8s ease-in-out;
    z-index: -1;
    pointer-events: none;
  }}
  
  .colorful-theme body::before {{
    opacity: 1;
  }}
  
  /* Particle Effect */
  .theme-particles {{
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 1000;
    overflow: hidden;
  }}
  
  .particle {{
    position: absolute;
    width: 3px;
    height: 3px;
    background: #f784c5;
    border-radius: 50%;
    animation: particleFloat 1.5s ease-out forwards;
  }}
  
  @keyframes particleFloat {{
    0% {{
      opacity: 1;
      transform: translateY(0) scale(1);
    }}
    100% {{
      opacity: 0;
      transform: translateY(-80vh) scale(0.5);
    }}
  }}
  .wrap {{ max-width: 800px; margin: 40px auto; padding: 20px; }}
  .card {{ 
    background: #ffffff; 
    border: 1px solid #e5e5e5; 
    border-radius: 8px;
    padding: 32px; 
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    transition: all 0.8s ease-in-out;
    transform: translateY(0) scale(1);
  }}
  
  /* Remove card styling during gameplay */
  .card.playing {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
  }}
  
  .card.theme-transitioning {{
    animation: cardSmooth 0.8s ease-in-out;
  }}
  
  @keyframes cardSmooth {{
    0% {{ 
      transform: translateY(0) scale(1);
      opacity: 1;
    }}
    50% {{ 
      transform: translateY(-15px) scale(0.98);
      opacity: 0.9;
    }}
    100% {{ 
      transform: translateY(0) scale(1);
      opacity: 1;
    }}
  }}
  h1, h2, h3 {{ margin: 0 0 24px; text-align: center; font-weight: 300; }}
  h1 {{ font-size: 28px; color: #000000; }}
  h2 {{ font-size: 24px; color: #333333; }}
  h3 {{ font-size: 20px; color: #333333; }}
  p {{ text-align: left; line-height: 1.6; color: #666666; }}
  label {{ display:block; margin:8px 0 6px; font-size:14px; color:#666666;}}
  input[type=text]{{ width:100%; box-sizing: border-box; padding:12px 16px; border-radius:4px; border:1px solid #cccccc; background:#ffffff; color:#000000; font-size:16px;}}
  input[type=text]:focus {{ outline: none; border-color: #000000; }}
  button {{ background:#000000; color:#ffffff; border:1px solid #000000; padding:12px 24px; border-radius:4px; cursor:pointer; font-weight:400; font-size:16px; transition: all 0.2s ease;}}
  button:hover {{ background:#ffffff; color:#000000; }}
  button:disabled {{ opacity:0.5; cursor:not-allowed;}}
  .row{{ display:flex; gap:16px; }} .row > div {{ flex:1;}}
  .center-row {{ display: flex; justify-content: center; gap: 20px; margin-top: 20px; }}
  .titlebox{{ 
    font-size:18px; 
    line-height:1.5; 
    padding:20px; 
    background:#f8f8f8; 
    border:1px solid #e5e5e5; 
    border-radius:4px; 
    margin:16px 0; 
    color:#000000;
    min-height: 60px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .muted{{ color:#999999; font-size:13px;}}
  .mono{{ font-family: 'Inter', 'SF Mono', Monaco, 'Cascadia Code', monospace; }}
  .pill {{ display:inline-block; padding:4px 12px; border-radius:16px; background:#f0f0f0; color:#000000; font-size:12px; font-weight:500; }}
  .timer {{ font-weight:600; font-size:18px; color:#000000; }}
  .ok {{ color:#000000; font-weight:600; }} .bad{{ color:#666666;}} .special{{color:#000000; font-weight:600;}}
  .choicesContainer {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 16px; }}
  .choiceBtn, .mode-button {{ background: #ffffff; border: 1px solid #cccccc; color: #000000; padding: 12px; text-align: center; font-size: 16px; width: 100%; box-sizing: border-box; border-radius: 4px; }}
  .choiceBtn:not(:disabled):hover, .mode-button:not(:disabled):hover {{ background: #f8f8f8; border-color: #000000; }}
  .choiceBtn.selected {{ background: #000000; color: #ffffff; }}
  .choiceBtn.correct {{ background: #000000; color: #ffffff; }}
  .choiceBtn.incorrect {{ background: #f0f0f0; color: #999999; opacity: 0.7; }}
  #hardQuestionControls {{ display: none; text-align: center; margin-top: 10px; }}
  .hard-choices {{ border: 1px solid #e5e5e5; padding: 16px; border-radius: 4px; margin-bottom: 12px; background: #f8f8f8;}}
  .hard-choices-label {{ font-size: 12px; color: #666666; margin-bottom: 8px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px; }}
  .overlay {{ position: fixed; inset: 0; background: rgba(0,0,0,0.8); display: none; align-items: center; justify-content: center; z-index: 50; }}
  .overlay .panel {{ background:#ffffff; border:1px solid #e5e5e5; border-radius:8px; padding:32px; text-align:center; box-shadow:0 4px 16px rgba(0,0,0,0.2); }}
  .overlay h2{{ margin:0 0 16px; font-size:24px; color: #000000; }} .overlay .desc{{ color:#666666; }}
  
  /* Start Page Layout Styles */
  .start-header {{ 
    background: #ffffff; 
    border-radius: 8px;
    padding: 32px; 
    margin-bottom: 24px;
    text-align: center;
    border: 1px solid #e5e5e5;
  }}
  .start-header h1 {{ 
    margin: 0 0 32px 0; 
    font-size: 28px;
    color: #000000; 
    font-weight: 700;
  }}
  .player-inputs {{ 
    display: flex;
    flex-direction: column;
    gap: 16px; 
    margin-bottom: 32px; 
  }}
  .input-group {{ 
    flex: 1; 
  }}
  .input-group input {{ 
    width: 100%; 
    box-sizing: border-box; 
    padding: 12px 16px; 
    border-radius: 4px; 
    border: 1px solid #cccccc; 
    background: #ffffff; 
    color: #000000; 
    font-size: 16px; 
  }}
  .input-group input:focus {{ 
    outline: none; 
    border-color: #000000; 
  }}
  .mode-selection {{
    display: flex;
    gap: 16px; 
    justify-content: center;
  }}
  .mode-btn {{
    background: #ffffff; 
    color: #000000; 
    border: 1px solid #cccccc; 
    padding: 12px 24px; 
    border-radius: 4px;
    cursor: pointer;
    font-size: 16px; 
    font-weight: 400;
    transition: all 0.2s ease;
    flex: 1; 
    max-width: 150px; 
  }}
  .mode-btn:hover {{
    background: #f8f8f8; 
    border-color: #000000; 
  }}
  .mode-btn:active {{ 
    background: #000000; 
    border-color: #000000; 
    color: #ffffff;
  }}
  
  .start-rules {{ 
    background: #f8f8f8; 
    color: #000000; 
    border-radius: 8px; 
    padding: 32px; 
    border: 1px solid #e5e5e5; 
  }}
  .rules-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 24px; 
    margin-bottom: 32px; 
  }}
  .rule-item {{
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 12px; 
  }}
  .rule-number {{
    background: #000000; 
    color: #ffffff;
    width: 40px; 
    height: 40px; 
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 16px;
    flex-shrink: 0;
  }}
  .rule-text {{
    font-size: 14px;
    line-height: 1.4;
    color: #666666; 
  }}
  
  .mode-descriptions {{
    display: flex; 
    flex-direction: column; 
    gap: 20px; 
  }}
  .mode-desc {{
    margin-bottom: 0; 
  }}
  .mode-desc-text {{
    font-size: 16px;
    line-height: 1.6;
    color: #000000; 
  }}
  .mode-desc-text strong {{ 
    color: #000000; 
    font-weight: 600; 
  }}
  
  /* Theme Toggle */
  .theme-toggle {{
    position: fixed;
    top: 20px;
    right: 20px;
    background: #000000;
    color: #ffffff;
    border: 1px solid #000000;
    padding: 8px 16px;
    border-radius: 20px;
    cursor: pointer;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    z-index: 100;
    transform: scale(1);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }}
  .theme-toggle:hover {{
    background: #ffffff;
    color: #000000;
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }}
  .theme-toggle:active {{
    transform: scale(0.95);
  }}
  
  /* Colorful Theme */
  .colorful-theme {{
    --primary-bg: #f784c5;
    --secondary-bg: #1b602f;
    --text-primary: #ffffff;
    --text-secondary: #000000;
    --accent: #1b602f;
    --border: #ffffff;
  }}
  
  .colorful-theme body {{
    background: linear-gradient(68deg, #f784c5 0%, #1b602f 100%) !important;
    background-attachment: fixed;
    min-height: 100vh;
    color: #ffffff;
  }}
  
  .colorful-theme {{
    background: linear-gradient(68deg, #f784c5 0%, #1b602f 100%) !important;
    min-height: 100vh;
  }}
  
  .colorful-theme html {{
    background: linear-gradient(68deg, #f784c5 0%, #1b602f 100%) !important;
    min-height: 100vh;
  }}
  
  .colorful-theme .card {{
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: #000000;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  }}
  
  .colorful-theme h1, .colorful-theme h2, .colorful-theme h3 {{
    color: #1b602f;
  }}
  
  .colorful-theme .start-header {{
    background: rgba(255, 255, 255, 0.95);
    border: none;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
  }}
  
  .colorful-theme .start-header h1 {{
    color: #1b602f;
    font-weight: 700;
  }}
  
  .colorful-theme .input-group input {{
    border: 1px solid rgba(27, 96, 47, 0.3);
    background: rgba(255, 255, 255, 0.9);
    color: #000000;
    backdrop-filter: blur(5px);
  }}
  
  .colorful-theme .input-group input:focus {{
    border-color: #1b602f;
    box-shadow: 0 0 0 3px rgba(27, 96, 47, 0.15);
    background: rgba(255, 255, 255, 0.95);
  }}
  
  .colorful-theme .mode-btn {{
    background: #1b602f;
    color: #ffffff;
    border: none;
    box-shadow: 0 4px 16px rgba(27, 96, 47, 0.3);
  }}
  
  .colorful-theme .mode-btn:hover {{
    background: #f784c5;
    color: #ffffff;
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(247, 132, 197, 0.4);
  }}
  
  .colorful-theme .mode-btn:active {{
    background: #1b602f;
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(27, 96, 47, 0.3);
  }}
  
  .colorful-theme .start-rules {{
    background: rgba(255, 255, 255, 0.95);
    border: none;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
  }}
  
  .colorful-theme .rule-number {{
    background: linear-gradient(135deg, #f784c5 0%, #1b602f 100%);
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(247, 132, 197, 0.3);
  }}
  
  .colorful-theme .rule-text {{
    color: #000000;
  }}
  
  .colorful-theme .mode-desc-text {{
    color: #000000;
  }}
  
  .colorful-theme .mode-desc-text strong {{
    color: #f784c5;
  }}
  
  .colorful-theme button {{
    background: #1b602f;
    color: #ffffff;
    border: none;
    box-shadow: 0 4px 16px rgba(27, 96, 47, 0.3);
  }}
  
  .colorful-theme button:hover {{
    background: #f784c5;
    color: #ffffff;
    box-shadow: 0 6px 20px rgba(247, 132, 197, 0.4);
    transform: translateY(-1px);
  }}
  
  .colorful-theme .choiceBtn, .colorful-theme .mode-button {{
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(27, 96, 47, 0.3);
    color: #1b602f;
    backdrop-filter: blur(5px);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  }}
  
  .colorful-theme .choiceBtn:hover, .colorful-theme .mode-button:hover {{
    background: #f784c5;
    border-color: transparent;
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(247, 132, 197, 0.4);
  }}
  
  .colorful-theme .choiceBtn.selected {{
    background: #1b602f;
    color: #ffffff;
    border-color: transparent;
    box-shadow: 0 4px 16px rgba(27, 96, 47, 0.4);
  }}
  
  .colorful-theme .choiceBtn.correct {{
    background: #1b602f;
    color: #ffffff;
    border-color: transparent;
    box-shadow: 0 4px 16px rgba(27, 96, 47, 0.4);
  }}
  
  .colorful-theme .titlebox {{
    background: rgba(255, 255, 255, 0.95);
    border: none;
    color: #000000;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    backdrop-filter: blur(10px);
  }}
  
  .colorful-theme .pill {{
    background: #f784c5;
    color: #ffffff;
  }}
  
  .colorful-theme .timer {{
    color: #1b602f;
  }}
  
  .colorful-theme .ok {{
    color: #1b602f;
  }}
  
  .colorful-theme .special {{
    color: #f784c5;
  }}
  
  .colorful-theme .theme-toggle {{
    background: #1b602f;
    color: #ffffff;
    border: 2px solid #1b602f;
    box-shadow: 0 2px 8px rgba(27, 96, 47, 0.3);
  }}
  
  .colorful-theme .theme-toggle:hover {{
    background: #f784c5;
    border-color: #f784c5;
    color: #ffffff;
    transform: scale(1.05);
    box-shadow: 0 4px 16px rgba(247, 132, 197, 0.4);
  }}
  
  .colorful-theme .theme-toggle:active {{
    transform: scale(0.95);
  }}
  
  /* Game Status Bar */
  .game-status-bar {{
    background: transparent;
    padding: 12px 0;
    margin-bottom: 16px;
    border: none;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  
  .colorful-theme .game-status-bar {{
    background: transparent;
  }}
  
  /* Countdown Timer Bar */
  .timer-container {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  
  .timer-bar {{
    width: 200px;
    height: 8px;
    background: #e5e5e5;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }}
  
  .timer-fill {{
    height: 100%;
    background: #3b82f6;
    border-radius: 4px;
    transition: width 0.2s linear;
    position: relative;
  }}
  
  .timer-fill.warning {{
    background: #f59e0b;
  }}
  
  .timer-fill.danger {{
    background: #ef4444;
  }}
  
  .colorful-theme .timer-bar {{
    background: rgba(255, 255, 255, 0.3);
  }}
  
  .colorful-theme .timer-fill {{
    background: #1b602f;
  }}
  
  .colorful-theme .timer-fill.warning {{
    background: #f784c5;
  }}
  
  .colorful-theme .timer-fill.danger {{
    background: #ef4444;
  }}
  
  /* Theme Toggle Animation */
  .theme-toggle.animating {{
    animation: themeToggleSmooth 0.8s ease-in-out;
  }}
  
  @keyframes themeToggleSmooth {{
    0% {{ 
      transform: scale(1);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    50% {{ 
      transform: scale(1.2);
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }}
    100% {{ 
      transform: scale(1);
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
  }}
  
  .colorful-theme .theme-toggle.animating {{
    animation: colorfulThemeToggleSmooth 0.8s ease-in-out;
  }}
  
  @keyframes colorfulThemeToggleSmooth {{
    0% {{ 
      transform: scale(1);
      box-shadow: 0 2px 8px rgba(27, 96, 47, 0.3);
    }}
    50% {{ 
      transform: scale(1.2);
      box-shadow: 0 8px 24px rgba(247, 132, 197, 0.5);
    }}
    100% {{ 
      transform: scale(1);
      box-shadow: 0 2px 8px rgba(27, 96, 47, 0.3);
    }}
  }}
</style>
</head>
<body>
  <button class="theme-toggle" id="themeToggle">💗💚</button>
  <div class="wrap">
    <div class="card">
      <div id="step-home" style="display:none;">
        <div class="start-header">
          <h1>Selamat Datang di News N-gram Game!</h1>
          <div class="player-inputs">
            <div class="input-group">
              <input id="player1-name" type="text" value="Pemain 1" maxlength="20" placeholder="Nama Pemain 1"/>
      </div>
            <div class="input-group">
              <input id="player2-name" type="text" value="Pemain 2" maxlength="20" placeholder="Nama Pemain 2"/>
            </div>
          </div>
      <div class="mode-selection">
        <button class="mode-btn" data-mode="uraian">Uraian</button>
        <button class="mode-btn" data-mode="pilihan_ganda">Pilgan</button>
      </div>
    </div>
    
        <div class="start-rules">
        <div class="rules-grid">
          <div class="rule-item">
            <div class="rule-number">1</div>
            <div class="rule-text">Setiap pemain menjawab 7 pertanyaan</div>
          </div>
          <div class="rule-item">
            <div class="rule-number">2</div>
            <div class="rule-text">10 detik per pertanyaan</div>
          </div>
          <div class="rule-item">
            <div class="rule-number">3</div>
            <div class="rule-text">Maksimal 100 poin untuk setiap pertanyaan</div>
          </div>
          <div class="rule-item">
            <div class="rule-number">4</div>
            <div class="rule-text">Poin terbanyak dinyatakan menang</div>
        </div>
      </div>
      
      <div class="mode-descriptions">
        <div class="mode-desc">
              <div class="mode-desc-text">
                <strong>Mode Uraian:</strong> Pemain harus mengetik sendiri satu kata yang paling cocok untuk mengisi bagian yang kosong.
                <br><strong>Mode Pilihan Ganda:</strong> Pilih salah satu dari empat kata yang tersedia. Jika pemain menjawab benar 3 kali berturut-turut, maka pemain tersebut akan dihadapkan dengan soal sulit dengan 2 kata rumpang!
        </div>
        </div>
      </div>
    </div>
        </div>
      </div>

      <div id="step-mode-select" style="display:none;">
        <h2>Pilih Mode Permainan</h2>
        <div class="center-row">
            <button class="mode-button" data-mode="uraian">Mode Uraian</button>
            <button class="mode-button" data-mode="pilihan_ganda">Mode Pilihan Ganda</button>
        </div>
      </div>

      <div id="step-player-setup" style="display:none;">
        <h2 id="setup-title">Masukkan Nama Pemain</h2>
        <div class="row">
          <div> <label>Nama Pemain 1</label> <input id="p1" type="text" value="Pemain 1" maxlength="20"/> </div>
          <div> <label>Nama Pemain 2</label> <input id="p2" type="text" value="Pemain 2" maxlength="20"/> </div>
        </div>
        <div style="margin-top:12px;"> <button id="btnStart">Mulai Permainan</button> </div>
      </div>

      <div id="step-play" style="display:none;">
        <div class="game-status-bar">
          <div>Giliran: <span id="who" class="pill"></span></div>
          <div class="timer-container">
            <span>Soal <span id="qnum"></span></span>
            <div class="timer-bar">
              <div class="timer-fill" id="timerFill"></div>
        </div>
            <span class="timer" id="timer"></span>
          </div>
        </div>
        <div id="difficulty-notice" class="special" style="text-align:center; margin-top: 8px; display: none;"></div>
        <div class="titlebox mono" id="titleBox" style="display: none;"></div>
        
        <div id="ui-pilihan-ganda" style="display:none;">
            <div id="choicesBox1" class="choicesContainer"></div>
            <div id="hardQuestionControls">
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
        </div>

        <div id="ui-uraian" style="display:none;">
            <form id="answerForm">
                <label for="answer-input">Jawab (1 kata)</label>
                <input id="answer-input" type="text" autocomplete="off" placeholder="Ketik jawabanmu lalu tekan Enter..." maxlength="40"/>
                <div style="margin-top:10px;"><button type="submit">Submit</button></div>
            </form>
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
const TOTAL_QUESTIONS = QUESTIONS_PER_PLAYER * 2; 
let sessionId = null, index = 1, currentQ = null, timerId = null, deadline = 0, currentMode = null;
let playerNames = {{}}, selections = {{}};

function $(id){{ return document.getElementById(id); }}
function setStep(step){{
  ['home', 'mode-select', 'player-setup', 'play', 'result'].forEach(s => {{
      $(`step-${{s}}`).style.display = (s === step) ? 'block' : 'none';
  }});
  
  // Add/remove playing class for card styling
  const card = document.querySelector('.card');
  if (card) {{
    if (step === 'play') {{
      card.classList.add('playing');
    }} else {{
      card.classList.remove('playing');
    }}
  }}
}}
async function startGame(){{
  const p1 = $("p1").value.trim() || "Pemain 1";
  const p2 = $("p2").value.trim() || "Pemain 2";
  playerNames = {{ p1, p2 }};
  const body = {{ players:[p1, p2], mode: currentMode }};
  const r = await fetch("/session",{{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
  const js = await r.json();
  if(!js.session_id){{ alert("Gagal membuat sesi"); return; }}
  sessionId = js.session_id; index = 1;
  setStep("play");
  await loadQuestion();
}}
async function loadQuestion(){{
  clearInterval(timerId);
  $("feedback").innerHTML = ""; 
  const difficultyNotice = $("difficulty-notice");
  difficultyNotice.innerHTML = ""; 
  difficultyNotice.style.display = "none";
  selections = {{}};
  $("who").textContent = playerNames[index <= TOTAL_QUESTIONS/2 ? 'p1' : 'p2'];
  $("qnum").textContent = `${{index}}/${{TOTAL_QUESTIONS}}`;
  
  // Reset timer bar
  const timerFill = $("timerFill");
  if (timerFill) {{
    timerFill.style.width = '100%';
    timerFill.classList.remove('warning', 'danger');
  }}

  const r = await fetch(`/question?session_id=${{sessionId}}&index=${{index}}`);
  const js = await r.json();
  if(js.status!=="ok"){{ alert(js.message); return; }}
  currentQ = js;
  const titleBox = $("titleBox");
  if (js.display_title_full && js.display_title_full.trim()) {{
    titleBox.textContent = js.display_title_full;
    titleBox.style.display = "block";
  }} else {{
    titleBox.style.display = "none";
  }}

  if(currentMode === 'pilihan_ganda') {{
    $("ui-pilihan-ganda").style.display = 'block';
    $("ui-uraian").style.display = 'none';
    ["choicesBox1", "choicesBox2", "choicesBox3"].forEach(id => $(id).innerHTML = "");
    if (js.is_hard) {{
        difficultyNotice.textContent = "🔥 SOAL SULIT: Pilih 2 Kata! 🔥";
        difficultyNotice.style.display = "block";
        $("choicesBox1").style.display = "none";
        $("hardQuestionControls").style.display = "block";
        js.choices_sets[0].forEach(c => createChoiceButton(c.text, 1, "choicesBox2"));
        js.choices_sets[1].forEach(c => createChoiceButton(c.text, 2, "choicesBox3"));
    }} else {{
        $("hardQuestionControls").style.display = "none";
        $("choicesBox1").style.display = "grid";
        js.choices_sets[0].forEach(c => createChoiceButton(c.text, 1, "choicesBox1"));
    }}
  }} else {{ // Mode Uraian
    $("ui-pilihan-ganda").style.display = 'none';
    $("ui-uraian").style.display = 'block';
    $("answer-input").value = "";
    // PERBAIKAN BUG: Aktifkan kembali input untuk soal selanjutnya
    $("answer-input").disabled = false;
    $("answer-input").focus();
  }}
  deadline = js.expires_at*1000;
  tick();
  timerId = setInterval(tick, 200);
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
$("submitHard").onclick = () => {{
    const combinedAnswer = `${{selections[1]}}||${{selections[2]}}`;
    submitAnswer(combinedAnswer);
}};
$("answerForm").onsubmit = (e) => {{
    e.preventDefault();
    const answerText = $("answer-input").value.trim();
    if(answerText) submitAnswer(answerText);
}};
async function submitAnswer(answerText){{
  document.querySelectorAll('.choiceBtn').forEach(b => b.disabled = true);
  $("submitHard").disabled = true;
  $("answer-input").disabled = true;
  const isTimeout = !answerText;
  const payload = {{ session_id: sessionId, question_id: currentQ.question_id, answer_text: isTimeout ? "" : answerText }};
  const r = await fetch("/answer",{{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(payload)}});
  const js = await r.json();
  clearInterval(timerId);
  
  if (js.status === "ok" || js.status === "timeout") {{
    const score = js.score_bucket;
    let feedbackHTML = `<div class="${{score >= 50 ? 'ok' : 'bad'}}">Skor: <b class="score">${{score}}</b></div>`;
    if (js.is_hard) {{
        feedbackHTML += `<div>Jawaban Benar: <b>${{js.blank_texts[0]}}</b> & <b>${{js.blank_texts[1]}}</b></div>`;
        highlightHardChoices(js.blank_texts);
    }} else {{
        feedbackHTML += `<div>Jawaban Benar: <b>${{js.blank_texts[0]}}</b></div>`;
        if (currentMode === 'pilihan_ganda') highlightNormalChoices(js.blank_texts[0]);
    }}
    $("feedback").innerHTML = feedbackHTML;
  }} else {{
     $("feedback").innerHTML = `<div class="bad">${{js.message || "Terjadi kesalahan"}}</div>`;
  }}
  setTimeout(nextStep, 3500);
}}
function highlightNormalChoices(goldText) {{
    $("choicesBox1").querySelectorAll('.choiceBtn').forEach(b => {{
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
  
  // Update timer bar
  const totalTime = currentQ ? (currentQ.is_hard ? 15 : 10) : 10;
  const percentage = (left / totalTime) * 100;
  const timerFill = $("timerFill");
  
  if (timerFill) {{
    timerFill.style.width = percentage + '%';
    
    // Change color based on time remaining
    timerFill.classList.remove('warning', 'danger');
    if (percentage <= 20) {{
      timerFill.classList.add('danger');
    }} else if (percentage <= 40) {{
      timerFill.classList.add('warning');
    }}
  }}
  
  if(left<=0){{
    clearInterval(timerId);
    submitAnswer(null);
  }}
}}
document.addEventListener("DOMContentLoaded", ()=>{{
  setStep("home");
  
  // Theme toggle functionality
  let isColorfulTheme = false;
  const themeToggle = $("themeToggle");
  const body = document.body;
  
  themeToggle.onclick = () => {{
    // Add animation classes
    themeToggle.classList.add('animating');
    const card = document.querySelector('.card');
    if (card) {{
      card.classList.add('theme-transitioning');
    }}
    
    // Create particle effect
    createParticleEffect();
    
    // Remove animation classes after animation completes
    setTimeout(() => {{
      themeToggle.classList.remove('animating');
      if (card) {{
        card.classList.remove('theme-transitioning');
      }}
    }}, 800);
    
    // Delay theme change for smooth effect
    setTimeout(() => {{
      isColorfulTheme = !isColorfulTheme;
      if (isColorfulTheme) {{
        body.classList.add('colorful-theme');
        themeToggle.textContent = 'Minimalist';
      }} else {{
        body.classList.remove('colorful-theme');
        themeToggle.textContent = '💗💚';
      }}
    }}, 100);
  }};
  
  // Particle effect function
  function createParticleEffect() {{
    const particleContainer = document.createElement('div');
    particleContainer.className = 'theme-particles';
    document.body.appendChild(particleContainer);
    
    // Create 12 particles for better performance
    for (let i = 0; i < 12; i++) {{
      const particle = document.createElement('div');
      particle.className = 'particle';
      particle.style.left = Math.random() * 100 + '%';
      particle.style.top = '100vh';
      particle.style.background = isColorfulTheme ? '#1b602f' : '#f784c5';
      particle.style.animationDelay = Math.random() * 0.3 + 's';
      particleContainer.appendChild(particle);
    }}
    
    // Remove particle container after animation
    setTimeout(() => {{
      if (document.body.contains(particleContainer)) {{
        document.body.removeChild(particleContainer);
      }}
    }}, 1500);
  }}
  
  // Handle mode selection and direct game start
  document.querySelectorAll('.mode-btn').forEach(btn => {{
      btn.onclick = async () => {{
          // Get player names from input fields
          const p1Name = $("player1-name").value.trim() || "Pemain 1";
          const p2Name = $("player2-name").value.trim() || "Pemain 2";
          
          // Set current mode
          currentMode = btn.dataset.mode;
          
          // Set player names
          playerNames = {{ p1: p1Name, p2: p2Name }};
          
          // Start the game directly
          const body = {{ players: [p1Name, p2Name], mode: currentMode }};
          const r = await fetch("/session", {{method:"POST", headers:{{"Content-Type":"application/json"}}, body:JSON.stringify(body)}});
          const js = await r.json();
          if(!js.session_id){{ alert("Gagal membuat sesi"); return; }}
          sessionId = js.session_id; index = 1;
          setStep("play");
          await loadQuestion();
      }}
  }});
  
  // Keep existing mode selection functionality for backward compatibility
  document.querySelectorAll('.mode-button').forEach(btn => {{
      btn.onclick = () => {{
          currentMode = btn.dataset.mode;
          $("setup-title").textContent = `Masukkan Nama Pemain (Mode ${{(currentMode === 'uraian' ? 'Uraian' : 'Pilihan Ganda')}})`;
          setStep('player-setup');
      }}
  }});
  $("btnStart").onclick = startGame;
}});
</script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)