#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, json, argparse, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Optional, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ===== ייבוא מהקובץ שלך =====
from structure_cnn_torch_min import (
    KimCNN, Vocab, build_sequences, set_seed, get_device,
    load_all_csvs_recursive
)

# ---------- דטאסט לאינפרנס/ניתוח ----------
class SeqDSEval(Dataset):
    def __init__(self, lex_texts, lemma_texts, lv: Vocab, mv: Optional[Vocab],
                 max_len_lex=256, max_len_lemma=128, use_lemma=True):
        self.lex, self.lem = lex_texts, lemma_texts
        self.lv, self.mv = lv, mv
        self.L, self.Lm = max_len_lex, max_len_lemma
        self.use_lemma = use_lemma and (mv is not None)
    def __len__(self): return len(self.lex)
    def __getitem__(self, i):
        lx = self.lv.encode(self.lex[i])[:self.L]
        lm = (self.mv.encode(self.lem[i])[:self.Lm] if self.use_lemma else [])
        return torch.tensor(lx, dtype=torch.long), torch.tensor(lm, dtype=torch.long)

def collate_eval(batch):
    lex, lem = zip(*batch)
    pad = lambda seqs: nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0)
    Xlex = pad(lex)
    Xlem = pad(lem) if any(len(s) > 0 for s in lem) else torch.empty((len(batch), 0), dtype=torch.long)
    return Xlex, Xlem

# ---------- טעינת ווקאב ----------
def load_vocab(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    v = Vocab(min_freq=1)
    v.itos = data["itos"]
    v.stoi = {tok:i for i,tok in enumerate(v.itos)}
    return v

# ---------- Top n-grams לפילטרים (lex) + שמירת channel ----------
@torch.no_grad()
def top_ngrams_for_filters_lex_with_channels(
    model: KimCNN, lv: Vocab, Xlex_ids: torch.Tensor, top_k=10, device="cpu"
) -> Dict[Tuple[int,int,int], List[Tuple[float,str,int,int]]]:
    """
    מחזיר מפה: (fi, kernel_size, channel_idx) -> רשימת [(score, ngram, example_idx, start)]
    """
    model.eval()
    convs = model.convs_lex
    Elex = model.emb_lex

    batch = Xlex_ids.to(device)            # [N, T]
    emb = Elex(batch)                      # [N, T, E]
    x = emb.transpose(1,2)                 # [N, E, T]

    per_filter_chan = defaultdict(list)

    for fi, conv in enumerate(convs):
        k = conv.kernel_size[0]
        feat = F.relu(conv(x))             # [N, C, T-k+1]
        N, C, L = feat.shape
        # מקסימום לאורך הזמן לכל ערוץ
        vals, idxs = torch.max(feat, dim=2)  # [N, C]
        # נעבור ערוץ-ערוץ כדי לקחת את הטופ לפי ערוץ (כדי לשייך ל-channel)
        for cidx in range(C):
            vcol = vals[:, cidx]                 # [N]
            ic  = idxs[:, cidx]                  # [N]
            tk = min(top_k, vcol.numel())
            topv, topi = torch.topk(vcol, k=tk)
            for v, bi in zip(topv.tolist(), topi.tolist()):
                start = ic[bi].item()
                toks = batch[bi, start:start+k].tolist()
                toks = [t for t in toks if t != 0]
                ngram = " ".join([lv.itos[t] if t < len(lv.itos) else "<unk>" for t in toks])
                per_filter_chan[(fi, k, cidx)].append((v, ngram, int(bi), int(start)))

    # מיין פנימית
    out = {}
    for key, lst in per_filter_chan.items():
        lst.sort(key=lambda x: -x[0])
        out[key] = lst[:top_k]
    return out

# ---------- Saliency לדוגמה (lex + “ענף lemma דמה” במקרה הצורך) ----------
def token_saliency_lex(model: KimCNN, lv: Vocab, lex_ids: torch.Tensor,
                       device="cpu", target_class=None):
    """
    אם המודל אומן עם use_lemma=True, נוסיף ענף lemma מאופס כדי לשמור על ממד זהה. נגזור גרדיאנט רק על lex.
    """
    model.eval()
    lex_ids = lex_ids.to(device).unsqueeze(0)      # [1, T]
    lex_ids.requires_grad_(False)

    emb_lex = model.emb_lex(lex_ids).detach().clone().requires_grad_(True)  # [1, T, E]
    x_lex = emb_lex.transpose(1,2)                                         # [1, E, T]

    feats = []
    for conv in model.convs_lex:
        c = F.relu(conv(x_lex))
        p = F.max_pool1d(c, c.size(2)).squeeze(2)
        feats.append(p)

    # lemma דמה אם צריך
    if getattr(model, "use_lemma", False):
        T = lex_ids.size(1)
        lem_ids = torch.zeros((1, T), dtype=torch.long, device=device)
        emb_lem = model.emb_lem(lem_ids)
        x_lem = emb_lem.transpose(1,2)
        feats_lem = []
        for conv in model.convs_lem:
            c = F.relu(conv(x_lem))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            feats_lem.append(p)
        feat_lem = torch.cat(feats_lem, dim=1) if len(feats_lem) > 0 else None
    else:
        feat_lem = None

    feat_lex = torch.cat(feats, dim=1)
    feat = feat_lex if (feat_lem is None) else torch.cat([feat_lex, feat_lem], dim=1)

    h = F.relu(model.fc1(feat)); h = model.drop(h)
    h = F.relu(model.fc2(h));    h = model.drop(h)
    logits = model.out(h)
    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    loss = logits[0, target_class]
    loss.backward()

    grads = emb_lex.grad.detach().squeeze(0)  # [T, E]
    sal = grads.norm(dim=1)                   # [T]
    toks = lex_ids.squeeze(0).detach().cpu().tolist()
    tokens = [lv.itos[t] if t < len(lv.itos) else "<unk>" for t in toks]
    return tokens, sal.cpu().numpy().tolist(), target_class

# ---------- Probe: תרומת כל פיצ'ר (פילטר/ערוץ) לכל מחלקה ----------
@torch.no_grad()
def feature_class_contrib(model: KimCNN) -> Dict[int, List[Tuple[int,float]]]:
    """
    עבור כל ממד בכניסת fc1 (כלומר כל ערוץ של כל פילטר לאחר ה-pooling),
    נבנה וקטור יחידה ונעביר דרך fc1->relu->dropout->fc2->relu->dropout->out.
    נחזיר, לכל מחלקה, את רשימת (feature_index, logit) ממוינת מירידה (תרומה חזקה).
    הערה: זה קירוב נוח להסבר (יש ReLU ו-nonlinearity).
    """
    model.eval()
    in_dim = model.fc1.in_features
    num_classes = model.out.out_features
    contrib = {c: [] for c in range(num_classes)}

    # נכבה דרופאאוט לגמרי (eval כבר דואג, אבל ליתר ביטחון)
    def forward_with_feat(feat_vec):
        h = F.relu(model.fc1(feat_vec))
        h = model.drop(h)  # אין השפעה ב-eval
        h = F.relu(model.fc2(h))
        h = model.drop(h)
        logits = model.out(h)
        return logits

    device = next(model.parameters()).device
    eye = torch.eye(in_dim, device=device)  # [in_dim, in_dim]
    # נריץ בקבוצות כדי לא לצרוך זיכרון עצום
    BS = 1024
    for s in range(0, in_dim, BS):
        e = min(in_dim, s+BS)
        block = eye[s:e]                # [B, in_dim]
        logits = forward_with_feat(block)
        for i, logit in enumerate(logits):  # לכל פיצ'ר
            for c in range(num_classes):
                contrib[c].append((s+i, float(logit[c].item())))
    # מיין לכל מחלקה
    for c in range(num_classes):
        contrib[c].sort(key=lambda x: -x[1])
    return contrib  # מפה: class_id -> [(feature_index, score), ...]

# ---------- מיפוי ממד->(סוג/פילטר/קנאל) ----------
def build_feature_index_map(model: KimCNN) -> List[Tuple[str,int,int,int]]:
    """
    מחזיר רשימה באורך in_dim של tuples:
    ("lex"/"lemma", conv_index, kernel_size, channel_idx)
    סדר: קודם כל convs_lex (לפי הסדר, כל אחד עם C ערוצים),
          אח"כ convs_lem (שני קונבולושים, כל אחד עם C//2 ערוצים).
    """
    mapping = []
    # lex
    C = model.convs_lex[0].out_channels
    for fi, conv in enumerate(model.convs_lex):
        k = conv.kernel_size[0]
        for cidx in range(C):
            mapping.append(("lex", fi, k, cidx))
    # lemma (אם קיים)
    if getattr(model, "use_lemma", False):
        C2 = model.convs_lem[0].out_channels  # זה C//2
        for fi, conv in enumerate(model.convs_lem):
            k = conv.kernel_size[0]
            for cidx in range(C2):
                mapping.append(("lemma", fi, k, cidx))
    return mapping  # len == fc1.in_features

# ---------- MAIN ----------
def main(args):
    set_seed(args.seed)
    dev = get_device()
    print("Device:", dev)

    # טען דאטה גולמי
    bav = load_all_csvs_recursive(args.bavli_dir, "bavli")
    yer = load_all_csvs_recursive(args.yeru_dir,  "yerushalmi")
    raw = pd.concat([bav, yer], ignore_index=True)

    # בנה רצפים (כמו באימון)
    seq = build_sequences(raw, group_by=args.group_by, min_len_lex=args.min_len_lex)
    seq = seq.dropna(subset=["lex_seq"])
    seq = seq[seq["lex_seq"].str.strip().astype(bool)]
    print(f"[SEQ] total sequences for analysis: {len(seq)}")

    # טען ווקאב(ים)
    lv = load_vocab(os.path.join(args.out_dir, "lex_vocab.json"))
    mv_path = os.path.join(args.out_dir, "lemma_vocab.json")
    mv = load_vocab(mv_path) if os.path.exists(mv_path) else None
    use_lemma = (mv is not None)

    # טקסטים
    Xl = seq["lex_seq"].astype(str).values
    Xm = seq["lemma_seq"].fillna("").astype(str).values if "lemma_seq" in seq.columns else np.array([""]*len(seq))

    # דטאסט + לואדר
    eval_ds = SeqDSEval(Xl, Xm, lv, mv, max_len_lex=args.max_len_lex, max_len_lemma=args.max_len_lemma, use_lemma=use_lemma)
    eval_ld = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_eval)

    # טען מודל עם אותם היפר-פרמטרים
    model = KimCNN(
        Vlex=len(lv), Elex=args.emb_dim_lex,
        Vlem=(len(mv) if use_lemma else None), Elem=args.emb_dim_lemma,
        filters=tuple(map(int, args.filter_sizes.split(","))),
        C=args.num_filters, dropout=args.dropout,
        num_classes=2, use_lemma=use_lemma
    ).to(dev)

    state_path = os.path.join(args.out_dir, "kimcnn.pt")
    model.load_state_dict(torch.load(state_path, map_location=dev, weights_only=False))
    model.eval()

    # ----- אסוף את כל ה-lex ids לרוחב קבוע T -----
    T = args.max_len_lex
    tensors = []
    with torch.no_grad():
        for i, (Xlex, Xlem) in enumerate(eval_ld):
            if Xlex.size(1) < T:
                Xlex = F.pad(Xlex, (0, T - Xlex.size(1)), value=0)
            else:
                Xlex = Xlex[:, :T]
            tensors.append(Xlex)
            if (i+1) >= args.max_batches:
                break

    if len(tensors) == 0:
        print("No data found for analysis.")
        return

    all_lex_ids = torch.cat(tensors, dim=0)  # [N, T]
    print(f"[ANALYZE] using {all_lex_ids.size(0)} examples for filter analysis, T={T}")

    # ----- 1) TOP N-GRAMS לכל פילטר (lex) -----
    top_map_chan = top_ngrams_for_filters_lex_with_channels(
        model, lv, all_lex_ids, top_k=args.top_k, device=dev
    )

    # ----- 2) תרומת פילטרים לפי מחלקה (Probe) -----
    contrib_by_class = feature_class_contrib(model)
    feat_map = build_feature_index_map(model)

    # הדפסה יפה:
    sep = "="*72
    sub = "-"*72

    print("\n" + sep)
    print("דוח ניתוח פילטרים – קריא למרצה")
    print(sep)

    # A) Top n-grams לכל פילטר (מקובץ לפי k)
    print("\nA) Top n-grams לכל פילטר (lex):")
    print(sub)
    # נקבץ לפי (fi,k) ונציג את ה-top הראשון מכל channel (כדי לא להציף)
    grouped = defaultdict(list)
    for (fi, k, cidx), lst in top_map_chan.items():
        if len(lst) > 0:
            score, ngram, exi, start = lst[0]
            grouped[(fi, k)].append((score, cidx, ngram))
    for (fi, k) in sorted(grouped.keys()):
        arr = sorted(grouped[(fi,k)], key=lambda x: -x[0])[:min(5, len(grouped[(fi,k)]))]
        print(f"[פילטר #{fi} | חלון k={k}] דוגמאות n-gram מייצגות:")
        for rank, (score, cidx, ngram) in enumerate(arr, 1):
            print(f"  {rank:>2}. chan={cidx:<3d} score={score:>7.4f} | <{ngram}>")
        print()

    # B) תרומת פילטרים לפי מחלקה
    print("\nB) תרומת פילטרים לפי מחלקה (Probe על כניסת fc1):")
    print(sub)
    class_names = ["bavli", "yerushalmi"]  # לפי מודל שני-מחלקות שלך
    for cid, cname in enumerate(class_names):
        print(f"\n>> מחלקה: {cname}")
        # ניקח top M פיצ'רים
        top_feats = contrib_by_class[cid][:args.top_feat_contrib]
        for rank, (fidx, score) in enumerate(top_feats, 1):
            block, fi, k, cidx = feat_map[fidx]
            # מצא n-gram מייצג אם זה lex
            rep = ""
            if block == "lex":
                key = (fi, k, cidx)
                if key in top_map_chan and len(top_map_chan[key]) > 0:
                    rep = top_map_chan[key][0][1]  # ngram
            print(f"  {rank:>2}. [{block}] conv#{fi} k={k} chan={cidx:<3d} | logit≈{score:>7.4f}"
                  + (f" | ngram=<{rep}>" if rep else ""))

    # C) Saliency לדוגמה
    print("\nC) תרומת טוקנים (Saliency) – דוגמה בודדת:")
    print(sub)
    example_idx = min(args.example_index, all_lex_ids.size(0)-1)
    tokens, sal, tgt = token_saliency_lex(model, lv, all_lex_ids[example_idx], device=dev, target_class=None)
    row = all_lex_ids[example_idx].tolist()
    L = row.index(0) if 0 in row else len(tokens)
    tokens = tokens[:L]; sal = sal[:L]
    pairs = list(zip(tokens, sal))
    pairs.sort(key=lambda x: -x[1])
    print(f"דוגמה #{example_idx} | תחזית מחלקה: {class_names[tgt]}")
    for tok, s in pairs[:args.top_saliency]:
        print(f"  {tok:>28s}  ->  {s:.5f}")

    # D) ייצוא אופציונלי ל-CSV
    if args.export_csv:
        os.makedirs(args.export_dir, exist_ok=True)
        # top-ngrams (עם channel)
        rows = []
        for (fi, k, cidx), lst in top_map_chan.items():
            for rank, (score, ngram, exi, start) in enumerate(lst, 1):
                rows.append({"conv_index": fi, "kernel_size": k, "channel": cidx,
                             "rank": rank, "score": score, "ngram": ngram,
                             "example_index": exi, "start": start})
        pd.DataFrame(rows).to_csv(os.path.join(args.export_dir, "top_ngrams_lex_by_channel.csv"), index=False)

        # feature contributions
        rows = []
        for cid, cname in enumerate(class_names):
            for rank, (fidx, score) in enumerate(contrib_by_class[cid], 1):
                block, fi, k, cidx = feat_map[fidx]
                rows.append({"class": cname, "rank": rank, "feature_index": fidx,
                             "block": block, "conv_index": fi, "kernel_size": k,
                             "channel": cidx, "logit_score": score})
        pd.DataFrame(rows).to_csv(os.path.join(args.export_dir, "feature_class_contrib.csv"), index=False)

        # saliency לדוגמה
        pd.DataFrame({"token": [t for t,_ in pairs], "saliency": [float(x) for _,x in pairs]}).to_csv(
            os.path.join(args.export_dir, f"saliency_example_{example_idx}.csv"), index=False)

        print(f"\n[EXPORTED] קבצי CSV נשמרו תחת {args.export_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # פרמטרים זהים לאימון
    ap.add_argument("--bavli_dir", type=str, default="Data/csv_Bavli")
    ap.add_argument("--yeru_dir",  type=str, default="Data/csv_Yerushalmi")
    ap.add_argument("--out_dir",   type=str, default="models/cnn_structure_torch")

    ap.add_argument("--group_by", choices=["word","line","page","side","masekhet"], default="line")
    ap.add_argument("--min_len_lex", type=int, default=3)

    ap.add_argument("--max_len_lex",   type=int, default=256)
    ap.add_argument("--max_len_lemma", type=int, default=128)

    ap.add_argument("--emb_dim_lex",   type=int, default=128)
    ap.add_argument("--emb_dim_lemma", type=int, default=64)
    ap.add_argument("--filter_sizes",  type=str, default="2,3,4")
    ap.add_argument("--num_filters",   type=int, default=128)
    ap.add_argument("--dropout",       type=float, default=0.4)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)

    # בקרה לניתוח
    ap.add_argument("--top_k", type=int, default=10, help="כמה n-grams להציג לכל ערוץ של פילטר")
    ap.add_argument("--max_batches", type=int, default=50, help="כמה באצ'ים לסרוק לפני עצירה")
    ap.add_argument("--example_index", type=int, default=0, help="איזו דוגמה להציג ב-saliency")
    ap.add_argument("--top_saliency", type=int, default=20, help="כמה טוקנים עם saliency גבוה להציג")
    ap.add_argument("--top_feat_contrib", type=int, default=15, help="כמה פילטרים מובילים להדפיס לכל מחלקה")

    ap.add_argument("--export_csv", action="store_true")
    ap.add_argument("--export_dir", type=str, default="analysis_exports")

    args = ap.parse_args()
    main(args)
