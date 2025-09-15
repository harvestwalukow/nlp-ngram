# app.py — News N-gram Game (Backend + Simple UI) - Final Stable Version
from __future__ import annotations
import os, re, time, math, random, unicodedata, statistics
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field

import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import ORJSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, constr

from pathlib import Path
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

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


@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = Path("templates/ui.html")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)