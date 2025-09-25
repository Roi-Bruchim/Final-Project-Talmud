#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, re, json, argparse, random, warnings, glob
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============== Utils & parsing ==============
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

URL_RE = re.compile(
    r"Masekhet:\s*(?P<masekhet>\d+),\s*Page:\s*(?P<page>\S+),\s*Side:\s*(?P<side>\S+),\s*Line:\s*(?P<line>\d+),\s*Word:\s*(?P<word>\d+)",
    flags=re.IGNORECASE
)

def parse_url(url: str):
    if not isinstance(url, str): return None
    m = URL_RE.search(url)
    if not m: return None
    gd = m.groupdict()
    # return all fields so we can choose grouping granularity
    return {
        "masekhet": gd["masekhet"],
        "page": gd["page"],
        "side": gd["side"],
        "line": int(gd["line"]),
        "word": int(gd["word"]),
    }

def make_group_id(p: dict, group_by: str) -> str:
    # group_by in {"word","line","page","side","masekhet"}
    if group_by == "word":
        return f"{p['masekhet']}|{p['page']}|{p['side']}|{p['line']}|{p['word']}"
    if group_by == "line":
        return f"{p['masekhet']}|{p['page']}|{p['side']}|{p['line']}"
    if group_by == "page":
        return f"{p['masekhet']}|{p['page']}|{p['side']}"
    if group_by == "side":
        return f"{p['masekhet']}|{p['page']}"
    if group_by == "masekhet":
        return f"{p['masekhet']}"
    return f"{p['masekhet']}|{p['page']}|{p['side']}|{p['line']}"

def norm_lex(s: str) -> str:
    if not isinstance(s, str): return ""
    parts = [p.strip() for p in re.split(r"\s*\|\s*", s) if p.strip()]
    return " ".join(re.sub(r"\s+", "_", p) for p in parts)

def norm_lemma(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.strip()
    if s == "" or s.lower() in {"none", "no data"}: return ""
    return re.sub(r"\s+", "_", s)

# ============== IO ==============
def load_all_csvs_recursive(folder_path: str, label_value: str):
    paths = glob.glob(os.path.join(folder_path, "**", "*.csv"), recursive=True)
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "url" not in df.columns or "merged_lexicon" not in df.columns:
            continue
        keep = ["url", "merged_lexicon"]
        if "Lema" in df.columns:
            keep.append("Lema")
        tmp = df[keep].copy()
        tmp["label"] = label_value
        dfs.append(tmp)
    if not dfs:
        raise ValueError(f"No suitable CSVs found under {folder_path}")
    out = pd.concat(dfs, ignore_index=True)
    print(f"[LOAD] {label_value}: files={len(paths)} rows={sum(len(pd.read_csv(p, nrows=1))>=0 for p in paths)} (raw concat={len(out)})")
    return out

def build_sequences(df: pd.DataFrame, group_by: str, min_len_lex: int):
    # parse url
    parsed = df["url"].map(parse_url)
    df = df[parsed.notnull()].copy()
    df["_parsed"] = parsed.dropna()
    # sort keys
    df["word_idx"] = df["_parsed"].map(lambda p: p["word"])
    # normalize tokens
    df["lex_tok"] = df["merged_lexicon"].map(norm_lex)
    df["lemma_tok"] = df["Lema"].map(norm_lemma) if "Lema" in df.columns else ""
    # filter trash
    def bad(s):
        if not isinstance(s, str): return True
        t = s.strip().lower()
        return (t == "" or t == "no data")
    df = df[~df["lex_tok"].map(bad)]

    # grouping id
    df["group_id"] = df["_parsed"].map(lambda p: make_group_id(p, group_by))
    # aggregate in order
    rows = []
    for gid, g in df.sort_values(["group_id", "word_idx"]).groupby("group_id"):
        lex_tokens, lemma_tokens = [], []
        for _, r in g.iterrows():
            if r["lex_tok"]:
                lex_tokens.extend(r["lex_tok"].split())
            if r["lemma_tok"]:
                lemma_tokens.append(r["lemma_tok"])
        if len(lex_tokens) < min_len_lex:
            continue
        rows.append((gid, " ".join(lex_tokens), " ".join(lemma_tokens)))
    seq = pd.DataFrame(rows, columns=["group_id", "lex_seq", "lemma_seq"])
    return seq

# ============== Vocab / Dataset ==============
class Vocab:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq
        self.pad = "<pad>"; self.unk = "<unk>"
        self.itos = [self.pad, self.unk]
        self.stoi = {self.pad: 0, self.unk: 1}
    def build(self, texts):
        c = Counter()
        for t in texts:
            for tok in str(t).split():
                c[tok] += 1
        for tok, f in c.items():
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
    def __len__(self): return len(self.itos)
    def encode(self, text): return [self.stoi.get(tok, 1) for tok in str(text).split()]
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

class SeqDS(Dataset):
    def __init__(self, lex_texts, lemma_texts, labels, lv: Vocab, mv: Vocab|None,
                 max_len_lex=256, max_len_lemma=128, use_lemma=True):
        self.lex, self.lem, self.y = lex_texts, lemma_texts, labels
        self.lv, self.mv = lv, mv
        self.L, self.Lm = max_len_lex, max_len_lemma
        self.use_lemma = use_lemma and (mv is not None)
    def __len__(self): return len(self.lex)
    def __getitem__(self, i):
        lx = self.lv.encode(self.lex[i])[:self.L]
        lm = (self.mv.encode(self.lem[i])[:self.Lm] if self.use_lemma else [])
        y  = int(self.y[i])
        return torch.tensor(lx, dtype=torch.long), torch.tensor(lm, dtype=torch.long), y

def collate(batch):
    lex, lem, y = zip(*batch)
    pad = lambda seqs: nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    Xlex = pad(lex)
    Xlem = pad(lem) if any(len(s) > 0 for s in lem) else torch.empty((len(batch), 0), dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    return Xlex, Xlem, y

# ============== Model ==============
class KimCNN(nn.Module):
    def __init__(self, Vlex, Elex=128, Vlem=None, Elem=64, filters=(2,3,4), C=128,
                 dropout=0.4, num_classes=2, use_lemma=True):
        super().__init__()
        self.use_lemma = use_lemma and (Vlem is not None)
        self.emb_lex = nn.Embedding(Vlex, Elex, padding_idx=0)
        self.convs_lex = nn.ModuleList([nn.Conv1d(Elex, C, k) for k in filters])
        if self.use_lemma:
            self.emb_lem = nn.Embedding(Vlem, Elem, padding_idx=0)
            self.convs_lem = nn.ModuleList([nn.Conv1d(Elem, C//2, k) for k in (2,3)])
        out_dim = len(filters)*C + (C if self.use_lemma else 0)
        self.fc1 = nn.Linear(out_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(128, num_classes)
    def _block(self, x, convs):
        x = x.transpose(1,2)
        feats=[]
        for conv in convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            feats.append(p)
        return torch.cat(feats, dim=1)
    def forward(self, lex_ids, lem_ids=None):
        lx = self.emb_lex(lex_ids)
        feat = self._block(lx, self.convs_lex)
        if self.use_lemma and lem_ids is not None and lem_ids.size(1)>0:
            lm = self.emb_lem(lem_ids)
            feat = torch.cat([feat, self._block(lm, self.convs_lem)], dim=1)
        h = self.drop(F.relu(self.fc1(feat)))
        h = self.drop(F.relu(self.fc2(h)))
        return self.out(h)

# ============== Train / Eval ==============
@torch.no_grad()
def evaluate(model, loader, crit, dev):
    model.eval(); totL=0; corr=0; n=0; ys=[]; ps=[]
    for Xlex,Xlem,y in loader:
        Xlex,Xlem,y = Xlex.to(dev), Xlem.to(dev), y.to(dev)
        logits = model(Xlex, Xlem if Xlem.numel()>0 else None)
        loss   = crit(logits, y)
        totL  += loss.item()*y.size(0)
        pred   = logits.argmax(1)
        corr  += (pred==y).sum().item()
        n     += y.size(0)
        ys.extend(y.cpu().tolist()); ps.extend(pred.cpu().tolist())
    return totL/n, corr/n, np.array(ys), np.array(ps)

# ============== Main ==============
def main(args):
    set_seed(args.seed)
    dev = get_device()
    print("Device:", dev)

    # ---- Load all files (recursive) ----
    bav = load_all_csvs_recursive(args.bavli_dir, "bavli")
    yer = load_all_csvs_recursive(args.yeru_dir,  "yerushalmi")
    raw = pd.concat([bav, yer], ignore_index=True)
    print(f"[RAW] total rows: {len(raw)}")

    # ---- Build sequences with chosen grouping ----
    seq = build_sequences(raw, group_by=args.group_by, min_len_lex=args.min_len_lex)
    # map group_id -> label (majority vote per group; usually uniform anyway)
    tmp = raw.dropna(subset=["url"]).copy()
    tmp["_p"] = tmp["url"].map(parse_url)
    tmp = tmp[tmp["_p"].notnull()]
    tmp["group_id"] = tmp["_p"].map(lambda p: make_group_id(p, args.group_by))
    lab_map = tmp.groupby("group_id")["label"].agg(lambda s: s.mode().iat[0]).reset_index()
    seq = seq.merge(lab_map, on="group_id", how="inner")

    # clean & stats
    before = len(seq)
    seq = seq.dropna(subset=["lex_seq","label"])
    seq = seq[seq["lex_seq"].str.strip().astype(bool)]
    print(f"[SEQ] built={before} | after_clean={len(seq)} | by='{args.group_by}'")

    # balance (optional)
    if not args.no_balance:
        minc = seq["label"].value_counts().min()
        seq  = seq.groupby("label", group_keys=False).apply(lambda x: x.sample(minc, random_state=args.seed)).reset_index(drop=True)
        print(f"[BALANCE] per class={minc} | total={len(seq)}")
    else:
        print(f"[BALANCE] skipped | class_counts:\n{seq['label'].value_counts()}")

    # labels & split
    le = LabelEncoder(); y_all = le.fit_transform(seq["label"])
    Xl = seq["lex_seq"].astype(str).values
    Xm = seq["lemma_seq"].fillna("").astype(str).values

    Xl_tr,Xl_tmp,Xm_tr,Xm_tmp,y_tr,y_tmp = train_test_split(Xl,Xm,y_all, test_size=args.test_size, random_state=args.seed, stratify=y_all)
    Xl_va,Xl_te,Xm_va,Xm_te,y_va,y_te = train_test_split(Xl_tmp,Xm_tmp,y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp)

    print(f"[SPLIT] train={len(y_tr)} | val={len(y_va)} | test={len(y_te)} | classes={list(le.classes_)}")

    # vocabs on TRAIN only
    use_lemma = any(t.strip() for t in Xm_tr)
    lv = Vocab(min_freq=args.min_freq); lv.build(Xl_tr)
    mv = Vocab(min_freq=args.min_freq) if use_lemma else None
    if use_lemma: mv.build(Xm_tr)

    # datasets
    train_ds = SeqDS(Xl_tr,Xm_tr,y_tr, lv,mv, args.max_len_lex,args.max_len_lemma, use_lemma)
    val_ds   = SeqDS(Xl_va,Xm_va,y_va, lv,mv, args.max_len_lex,args.max_len_lemma, use_lemma)
    test_ds  = SeqDS(Xl_te,Xm_te,y_te, lv,mv, args.max_len_lex,args.max_len_lemma, use_lemma)

    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  collate_fn=collate)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_ld  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # model
    model = KimCNN(
        Vlex=len(lv), Elex=args.emb_dim_lex,
        Vlem=(len(mv) if use_lemma else None), Elem=args.emb_dim_lemma,
        filters=tuple(map(int, args.filter_sizes.split(","))),
        C=args.num_filters, dropout=args.dropout,
        num_classes=len(le.classes_), use_lemma=use_lemma
    ).to(dev)

    # loss/opt
    if args.class_weights:
        classes, counts = np.unique(y_tr, return_counts=True)
        w = counts.sum()/np.maximum(counts,1)
        w = torch.tensor(w/w.mean(), dtype=torch.float32, device=dev)
        print("[CLASS WEIGHTS]", w.cpu().numpy())
    else:
        w = None
    crit = nn.CrossEntropyLoss(weight=w)
    opt  = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train (ES + LR reduce)
    best_va=0; best=None; bad=0
    for ep in range(1, args.epochs+1):
        model.train(); totL=0; cor=0; n=0
        for Xlex,Xlem,y in train_ld:
            Xlex,Xlem,y = Xlex.to(dev), Xlem.to(dev), y.to(dev)
            opt.zero_grad()
            logits = model(Xlex, Xlem if Xlem.numel()>0 else None)
            loss   = crit(logits, y)
            loss.backward(); opt.step()
            totL += loss.item()*y.size(0)
            cor  += (logits.argmax(1)==y).sum().item()
            n    += y.size(0)
        vaL, vaA, _, _ = evaluate(model, val_ld, crit, dev)
        print(f"Epoch {ep:02d} | train {totL/n:.4f}/{cor/n:.4f} | val {vaL:.4f}/{vaA:.4f}")
        if vaA > best_va + 1e-4:
            best_va=vaA; best = model.state_dict(); bad=0
        else:
            bad+=1
            if bad % max(1, args.patience//2) == 0:
                for g in opt.param_groups: g["lr"] = max(args.min_lr, g["lr"]*0.5)
                print("↓ lr =", opt.param_groups[0]["lr"])
            if bad >= args.patience:
                print("Early stopping."); break
    if best is not None:
        model.load_state_dict(best)

    teL, teA, yt, yp = evaluate(model, test_ld, crit, dev)
    print("\n=== Confusion Matrix (Test) ==="); print(confusion_matrix(yt, yp))
    print("\n=== Classification Report (Test) ==="); print(classification_report(yt, yp, target_names=le.classes_))

    # save
    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "kimcnn.pt"))
    with open(os.path.join(args.out_dir, "label_encoder.json"), "w", encoding="utf-8") as f:
        json.dump({"classes_": le.classes_.tolist()}, f, ensure_ascii=False)
    # save vocabs (fixed)
    Vocab.save(lv, os.path.join(args.out_dir, "lex_vocab.json"))
    if use_lemma and mv is not None:
        Vocab.save(mv, os.path.join(args.out_dir, "lemma_vocab.json"))
    print(f"\n[SAVED] {args.out_dir} | Test Acc: {teA:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bavli_dir", type=str, default="Data/csv_Bavli")
    ap.add_argument("--yeru_dir",  type=str, default="Data/csv_Yerushalmi")
    ap.add_argument("--out_dir",   type=str, default="models/cnn_structure_torch")

    ap.add_argument("--group_by", choices=["word","line","page","side","masekhet"], default="line",
                    help="רמת איגוד: word=כל מילה דגימה; line=ברירת מחדל; page/side/masekhet להקשר גדול יותר")
    ap.add_argument("--min_len_lex", type=int, default=3, help="סינון רצפים מבניים קצרים מדי")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--no_balance", action="store_true", help="אל תאזן מחלקות (משאיר יחס מקורי)")

    ap.add_argument("--min_freq",      type=int, default=1)
    ap.add_argument("--max_len_lex",   type=int, default=256)
    ap.add_argument("--max_len_lemma", type=int, default=128)

    ap.add_argument("--emb_dim_lex",   type=int, default=128)
    ap.add_argument("--emb_dim_lemma", type=int, default=64)
    ap.add_argument("--filter_sizes",  type=str, default="2,3,4")
    ap.add_argument("--num_filters",   type=int, default=128)
    ap.add_argument("--dropout",       type=float, default=0.4)

    ap.add_argument("--epochs",     type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr",         type=float, default=2e-3)
    ap.add_argument("--min_lr",     type=float, default=1e-5)
    ap.add_argument("--patience",   type=int, default=6)
    ap.add_argument("--class_weights", action="store_true")

    args = ap.parse_args()
    main(args)
