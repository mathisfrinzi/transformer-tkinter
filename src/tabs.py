# Les onglets 2 à 6 de l'interface
import math

import numpy as np
from tkinter import ttk, messagebox

from . import textes as tx
from .data import tokenize, detokenize, SOS, EOS, UNK
from .model import greedy_translate
from .plots import (plt, FG, pca3, style3d, heat, make_canvas, finish, dark_text, put_text,
                    make_tree, make_combo)


# ============================================================================
# Onglets 2 et 5 : vectorisation (anglais = source, français = cible)
# ============================================================================
class VectorTab:
    def __init__(self, app, parent, side):
        self.app, self.side = app, side  # side = "src" ou "tgt"
        self.data, self.sel = None, 0
        t = tx.VEC
        left = ttk.Frame(parent, width=340)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        if side == "src":
            ttk.Label(left, text=t["label_en"]).pack(anchor="w")
            ttk.Entry(left, textvariable=app.sentence_var).pack(fill="x", pady=2)
            ttk.Button(left, text=t["btn_vectoriser"], command=self.run).pack(fill="x", pady=2)
        else:
            ttk.Label(left, text=t["label_fr"]).pack(anchor="w")
            ttk.Entry(left, textvariable=app.fr_var).pack(fill="x", pady=2)
            ttk.Button(left, text=t["btn_traduction"], command=self.use_translation).pack(fill="x", pady=2)
            ttk.Button(left, text=t["btn_vectoriser"], command=self.run).pack(fill="x", pady=2)
            ttk.Label(left, text=t["note_fr"]).pack(anchor="w")

        ttk.Label(left, text=t["label_tokens"]).pack(anchor="w", pady=(8, 0))
        self.tree = make_tree(left, t["cols_tokens"], (40, 150, 60), height=10)
        self.tree.pack(fill="x")
        self.tree.bind("<<TreeviewSelect>>", self.on_select)
        ttk.Label(left, text=t["label_inspection"]).pack(anchor="w", pady=(8, 0))
        self.info = dark_text(left, height=18)
        self.info.pack(fill="both", expand=True)

        self.fig, self.canvas = make_canvas(right)

    def use_translation(self):
        if not self.app.check_idle():
            return
        words = tokenize(self.app.sentence_var.get())
        if not words:
            return
        ys, _ = greedy_translate(self.app.model, self.app.sv, self.app.tv, words)
        self.app.fr_var.set(detokenize([self.app.tv.itos[i] for i in ys]))
        self.run()

    def run(self):
        if not self.app.check_idle():
            return
        t = tx.VEC
        try:
            enc = self.app.encode_sentence(self.app.sentence_var.get())
            if self.side == "src":
                self.data = {"tokens": enc["tokens"], "ids": enc["ids"], "raw": enc["raw"],
                             "emb": enc["emb"], "pe": enc["pe"], "out": enc["layers"][-1],
                             "out_name": t["sortie_encoder"]}
            else:
                words = tokenize(self.app.fr_var.get())
                if not words:
                    raise ValueError(tx.COMMUN["phrase_fr_vide"])
                ids = [SOS] + self.app.tv.ids(words)
                dec = self.app.decode_prefix(enc, ids)
                self.data = {"tokens": ["<sos>"] + words, "ids": ids, "raw": ["<sos>"] + words,
                             "emb": dec["emb"], "pe": dec["pe"], "out": dec["layers"][-1],
                             "out_name": t["sortie_decoder"]}
        except ValueError as e:
            messagebox.showwarning(tx.COMMUN["attention"], str(e))
            return
        self.tree.delete(*self.tree.get_children())
        for i, (r, idx) in enumerate(zip(self.data["raw"], self.data["ids"])):
            label = r if idx != UNK else t["unk"].format(r=r)
            self.tree.insert("", "end", iid=str(i), values=(i, label, idx))
        self.sel = 0
        self.draw()

    def on_select(self, _evt=None):
        s = self.tree.selection()
        if s and self.data:
            self.sel = int(s[0])
            self.draw()

    def draw(self):
        d = self.data
        if d is None:
            return
        t, c = tx.VEC, tx.COMMUN
        toks, E, P, O = d["tokens"], d["emb"], d["pe"], d["out"]
        n, s = len(toks), self.sel
        vocab = self.app.sv if self.side == "src" else self.app.tv
        table = (self.app.model.src_emb if self.side == "src" else self.app.model.tgt_emb).weight.detach().numpy()

        # texte d'inspection : voisins par similarité cosinus dans la table d'embeddings
        tid = d["ids"][s]
        w = table / (np.linalg.norm(table, axis=1, keepdims=True) + 1e-9)
        cos = w @ w[tid]
        order = [i for i in np.argsort(-cos) if i != tid and i >= 4][:5]
        txt = t["inspection"].format(
            tok=d["raw"][s], id=tid, itos=vocab.itos[tid], pos=s, d=E.shape[1],
            n_e=np.linalg.norm(E[s]), n_p=np.linalg.norm(P[s]), n_ep=np.linalg.norm(E[s] + P[s]),
            out=d["out_name"], n_o=np.linalg.norm(O[s]),
            e8=np.round(E[s][:8], 3), p8=np.round(P[s][:8], 3))
        txt += "\n".join(f"  {vocab.itos[i]:<12} {cos[i]:+.3f}" for i in order)
        put_text(self.info, txt)

        f = self.fig
        f.clear()
        heat(f, f.add_subplot(2, 2, 1), E, t["plot_emb"], ylabels=toks, cmap="coolwarm")
        heat(f, f.add_subplot(2, 2, 2), P, t["plot_pe"], ylabels=toks, cmap="coolwarm")

        # PCA 3D commune : embedding, embedding + PE, sortie du modèle
        ax = f.add_subplot(2, 2, 3, projection="3d")
        style3d(ax)
        pts = pca3(np.vstack([E, E + P, O]))
        a, b, cc = pts[:n], pts[n:2 * n], pts[2 * n:]
        ax.scatter(*a.T, color="#4fa3ff", label=c["embedding"], s=20)
        ax.scatter(*b.T, color="#ffa64f", label=c["emb_pe"], s=20)
        ax.scatter(*cc.T, color="#6fdc6f", label=d["out_name"], s=20)
        for i in range(n):
            col = "#ff4f4f" if i == s else "#888888"
            lw = 2.2 if i == s else 0.6
            ax.plot(*np.vstack([a[i], b[i], cc[i]]).T, color=col, lw=lw)
            ax.text(*b[i], toks[i], fontsize=7, color="#ff8080" if i == s else FG)
        ax.set_title(t["plot_pca"])
        ax.legend(fontsize=6, loc="upper left")

        ax4 = f.add_subplot(2, 2, 4)
        ax4.plot(E[s], label=c["embedding"], lw=1)
        ax4.plot(P[s], label=c["pos_enc"], lw=1)
        ax4.plot(E[s] + P[s], label=c["emb_pe"], lw=1.4)
        ax4.set_title(t["plot_token"].format(tok=d["raw"][s]))
        ax4.set_xlabel(t["plot_dim"])
        ax4.legend(fontsize=6)
        finish(f, self.canvas)


# ============================================================================
# Onglet 3 : encoder
# ============================================================================
class EncoderTab:
    def __init__(self, app, parent):
        self.app = app
        self.enc = None
        t = tx.ENC
        left = ttk.Frame(parent, width=260)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text=t["label_phrase"]).pack(anchor="w")
        ttk.Entry(left, textvariable=app.sentence_var).pack(fill="x", pady=2)
        ttk.Button(left, text=t["btn_encoder"], command=self.run).pack(fill="x", pady=4)

        self.layer_cb = make_combo(left, t["label_couche"], self.draw)
        self.head_cb = make_combo(left, t["label_tete"], self.draw)
        self.tok_cb = make_combo(left, t["label_token"], self.draw)
        ttk.Label(left, text=t["label_infos"]).pack(anchor="w", pady=(10, 0))
        self.info = dark_text(left, height=20)
        self.info.pack(fill="both", expand=True)

        nb = ttk.Notebook(right)
        nb.pack(fill="both", expand=True)
        pages = []
        for name in t["pages"]:
            fr = ttk.Frame(nb)
            nb.add(fr, text=name)
            pages.append(make_canvas(fr))
        (self.f1, self.c1), (self.f2, self.c2), (self.f3, self.c3) = pages

    def run(self):
        if not self.app.check_idle():
            return
        try:
            self.enc = self.app.encode_sentence(self.app.sentence_var.get())
        except ValueError as e:
            messagebox.showwarning(tx.COMMUN["attention"], str(e))
            return
        nl, nh = len(self.enc["attn"]), self.enc["attn"][0]["Q"].shape[0]
        self.layer_cb["values"] = [tx.COMMUN["couche"].format(i=i + 1) for i in range(nl)]
        self.head_cb["values"] = [tx.COMMUN["tete"].format(i=i + 1) for i in range(nh)]
        self.tok_cb["values"] = [f"{i}: {tok}" for i, tok in enumerate(self.enc["tokens"])]
        self.layer_cb.current(0)
        self.head_cb.current(0)
        self.tok_cb.current(0)
        self.draw()

    def draw(self):
        e = self.enc
        if e is None:
            return
        t = tx.ENC
        li, h, s = self.layer_cb.current(), self.head_cb.current(), self.tok_cb.current()
        toks, layers = e["tokens"], e["layers"]
        nL, n = len(layers), len(toks)
        names = [t["nom_entree"]] + [t["nom_couche"].format(i=i) for i in range(1, nL)]
        vec = np.array([L[s] for L in layers])  # le token choisi à chaque couche
        norms = np.linalg.norm(vec, axis=1)
        cos_in = [float(v @ vec[0] / (np.linalg.norm(v) * np.linalg.norm(vec[0]) + 1e-9)) for v in vec]

        a = e["attn"][li]
        W = a["weights"][h]
        txt = t["info_haut"].format(tok=toks[s])
        txt += "\n".join(t["info_ligne_norme"].format(nm=nm, nv=nv, c=c)
                         for nm, nv, c in zip(names, norms, cos_in))
        txt += t["info_attention"].format(l=li + 1, h=h + 1, dk=a["Q"].shape[-1], tok=toks[s])
        txt += "\n".join(t["info_ligne_attn"].format(tok=toks[j], w=W[s, j]) for j in np.argsort(-W[s]))
        put_text(self.info, txt)

        # page 1 : trajectoire du token à travers les couches
        f = self.f1
        f.clear()
        P = pca3(np.vstack(layers)).reshape(nL, n, 3)
        ax = f.add_subplot(1, 3, 1, projection="3d")
        style3d(ax)
        for k in range(n):
            sel = k == s
            ax.plot(P[:, k, 0], P[:, k, 1], P[:, k, 2], color="#ff4f4f" if sel else "#6a8fbf",
                    lw=2.4 if sel else 0.9, marker="o", ms=3)
            ax.text(*P[-1, k], toks[k], fontsize=7, color="#ff8080" if sel else FG)
        for l in range(nL):
            ax.text(*P[l, s], str(l), fontsize=7, color="#ffd24f")
        ax.set_title(t["plot_trajectoire"])
        heat(f, f.add_subplot(1, 3, 2), vec, t["plot_vecteur"].format(tok=toks[s]), ylabels=names, cmap="coolwarm")
        ax3 = f.add_subplot(1, 3, 3)
        ax3.plot(range(nL), norms, marker="o", label=t["plot_norme"])
        ax3b = ax3.twinx()
        ax3b.plot(range(nL), cos_in, marker="s", color="#ffa64f", label=t["plot_cos"])
        ax3.set_xticks(range(nL))
        ax3.set_xticklabels(names, rotation=45)
        ax3.set_title(t["plot_evolution"].format(tok=toks[s]))
        ax3.legend(loc="upper left", fontsize=6)
        ax3b.legend(loc="lower left", fontsize=6)
        finish(f, self.c1)

        # page 2 : Q, K, V, scores, poids
        f = self.f2
        f.clear()
        heat(f, f.add_subplot(2, 3, 1), a["Q"][h], t["plot_q"], ylabels=toks, cmap="coolwarm")
        heat(f, f.add_subplot(2, 3, 2), a["K"][h], t["plot_k"], ylabels=toks, cmap="coolwarm")
        heat(f, f.add_subplot(2, 3, 3), a["V"][h], t["plot_v"], ylabels=toks, cmap="coolwarm")
        heat(f, f.add_subplot(2, 3, 4), a["scores"][h], t["plot_scores"], ylabels=toks, xlabels=toks,
             cmap="magma", annotate=True)
        heat(f, f.add_subplot(2, 3, 5), W, t["plot_poids"], ylabels=toks, xlabels=toks,
             cmap="viridis", annotate=True)
        ax6 = f.add_subplot(2, 3, 6)
        ax6.bar(range(n), W[s], color="#4fa3ff")
        ax6.set_xticks(range(n))
        ax6.set_xticklabels(toks, rotation=60, ha="right")
        ax6.set_title(t["plot_regarde"].format(tok=toks[s]))
        ax6.set_ylim(0, 1)
        f.suptitle(t["plot_titre_page2"].format(l=li + 1, h=h + 1))
        finish(f, self.c2)

        # page 3 : toutes les têtes de la couche
        f = self.f3
        f.clear()
        nh = a["weights"].shape[0]
        cols = min(4, nh)
        rows = math.ceil(nh / cols)
        for k in range(nh):
            heat(f, f.add_subplot(rows, cols, k + 1), a["weights"][k], t["plot_tete"].format(k=k + 1),
                 ylabels=toks, xlabels=toks, cmap="viridis")
        f.suptitle(t["plot_titre_page3"].format(l=li + 1))
        finish(f, self.c3)


# ============================================================================
# Onglet 4 : decoder
# ============================================================================
class DecoderTab:
    def __init__(self, app, parent):
        self.app = app
        self.state = None
        t = tx.DEC
        left = ttk.Frame(parent, width=300)
        left.pack(side="left", fill="y", padx=6, pady=6)
        left.pack_propagate(False)
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text=tx.COMMUN["phrase_en"]).pack(anchor="w")
        ttk.Entry(left, textvariable=app.sentence_var).pack(fill="x", pady=2)
        ttk.Label(left, text=t["label_strategie"]).pack(anchor="w", pady=(4, 0))
        self.strategy = ttk.Combobox(left, state="readonly", values=t["strategies"])
        self.strategy.current(0)
        self.strategy.pack(fill="x")
        ttk.Button(left, text=t["btn_demarrer"], command=self.start).pack(fill="x", pady=(6, 2))
        ttk.Button(left, text=t["btn_etape"], command=self.step).pack(fill="x", pady=2)
        ttk.Button(left, text=t["btn_tout"], command=self.run_all).pack(fill="x", pady=2)

        self.layer_cb = make_combo(left, t["label_couche"], self.draw)
        self.head_cb = make_combo(left, t["label_tete"], self.draw)
        self.kind_cb = make_combo(left, t["label_qkv"], self.draw)
        self.kind_cb["values"] = t["kinds"]
        self.kind_cb.current(0)

        ttk.Label(left, text=t["label_generes"]).pack(anchor="w", pady=(8, 0))
        self.gen_txt = dark_text(left, height=4)
        self.gen_txt.pack(fill="x")
        ttk.Label(left, text=t["label_top"]).pack(anchor="w", pady=(8, 0))
        self.info = dark_text(left, height=14)
        self.info.pack(fill="both", expand=True)

        self.fig, self.canvas = make_canvas(right)

    def start(self):
        if not self.app.check_idle():
            return
        try:
            enc = self.app.encode_sentence(self.app.sentence_var.get())
        except ValueError as e:
            messagebox.showwarning(tx.COMMUN["attention"], str(e))
            return
        ids = [SOS]
        self.state = {"enc": enc, "ids": ids, "info": self.app.decode_prefix(enc, ids), "done": False}
        nl = len(self.state["info"]["self_attn"])
        nh = self.state["info"]["self_attn"][0]["Q"].shape[0]
        self.layer_cb["values"] = [tx.COMMUN["couche"].format(i=i + 1) for i in range(nl)]
        self.head_cb["values"] = [tx.COMMUN["tete"].format(i=i + 1) for i in range(nh)]
        self.layer_cb.current(0)
        self.head_cb.current(0)
        self.draw()

    def step(self):
        if not self.app.check_idle():
            return
        if self.state is None:
            self.start()
        st = self.state
        if st is None or st["done"]:
            return
        p = st["info"]["probs"][-1]
        if self.strategy.current() == 0:
            nxt = int(np.argmax(p))
        else:
            nxt = int(np.random.choice(len(p), p=p / p.sum()))
        st["ids"].append(nxt)
        if nxt == EOS or len(st["ids"]) > 30:
            st["done"] = True
        st["info"] = self.app.decode_prefix(st["enc"], st["ids"])
        self.app.fr_var.set(detokenize([self.app.tv.itos[i] for i in st["ids"]]))
        self.draw()

    def run_all(self):
        if self.state is None or self.state["done"]:
            self.start()
        while self.state and not self.state["done"]:
            self.step()

    def draw(self):
        st = self.state
        if st is None:
            return
        t = tx.DEC
        tv = self.app.tv
        info, enc = st["info"], st["enc"]
        gen = [tv.itos[i] for i in st["ids"]]
        src = enc["tokens"]
        li, h = self.layer_cb.current(), self.head_cb.current()
        sa, ca = info["self_attn"][li], info["cross_attn"][li]

        gtxt = " ".join(gen)
        if st["done"]:
            gtxt += "\n\n" + t["fini"]
        put_text(self.gen_txt, gtxt)

        p = info["probs"][-1]
        top = np.argsort(-p)[:10]
        txt = t["distrib_fin"] if st["done"] else t["distrib"]
        txt += "\n".join(t["ligne_top"].format(r=r + 1, tok=tv.itos[i], p=p[i] * 100) for r, i in enumerate(top))
        put_text(self.info, txt)

        f = self.fig
        f.clear()
        heat(f, f.add_subplot(2, 3, 1), sa["weights"][h], t["plot_self"], ylabels=gen, xlabels=gen,
             annotate=True)
        heat(f, f.add_subplot(2, 3, 2), ca["weights"][h], t["plot_cross"], ylabels=gen, xlabels=src,
             cmap="plasma", annotate=True)
        ax = f.add_subplot(2, 3, 3)
        ax.barh(range(10)[::-1], p[top], color="#6fdc6f")
        ax.set_yticks(range(10)[::-1])
        ax.set_yticklabels([tv.itos[i] for i in top])
        ax.set_xlim(0, 1)
        ax.set_title(t["plot_proba"])

        # Q/K/V : self-attention (K et V viennent du decoder) ou cross-attention (K et V de l'encoder)
        self_kind = self.kind_cb.current() == 0
        a = sa if self_kind else ca
        klabels = gen if self_kind else src
        qui = t["qui_decoder"] if self_kind else t["qui_encoder"]
        heat(f, f.add_subplot(2, 3, 4), a["Q"][h], t["plot_q"], ylabels=gen, cmap="coolwarm")
        heat(f, f.add_subplot(2, 3, 5), a["K"][h], t["plot_k"].format(qui=qui), ylabels=klabels, cmap="coolwarm")
        heat(f, f.add_subplot(2, 3, 6), a["V"][h], t["plot_v"].format(qui=qui), ylabels=klabels, cmap="coolwarm")
        f.suptitle(t["plot_titre"].format(l=li + 1, h=h + 1, s=len(gen) - 1))
        finish(f, self.canvas)


# ============================================================================
# Onglet 6 : traduction et diagnostic
# ============================================================================
class DiagTab:
    def __init__(self, app, parent):
        self.app = app
        self.steps = None
        t = tx.DIAG
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text=tx.COMMUN["phrase_en"]).pack(side="left")
        ttk.Entry(top, textvariable=app.sentence_var, width=50).pack(side="left", padx=4)
        ttk.Button(top, text=t["btn_traduire"], command=self.run).pack(side="left", padx=4)
        ttk.Button(top, text=t["btn_effacer"], command=self.clear_runs).pack(side="right")

        mid = ttk.Frame(parent)
        mid.pack(fill="both", expand=True, padx=6)
        left = ttk.Frame(mid, width=380)
        left.pack(side="left", fill="y")
        left.pack_propagate(False)
        right = ttk.Frame(mid)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text=t["label_traduction"]).pack(anchor="w")
        self.out = dark_text(left, height=3)
        self.out.configure(font=("TkDefaultFont", 12))
        self.out.pack(fill="x")
        ttk.Label(left, text=t["label_pas_a_pas"]).pack(anchor="w", pady=(8, 0))
        self.tree = make_tree(left, t["cols_pas"], (45, 80, 60, 185), height=14)
        self.tree.pack(fill="both", expand=True)
        self.fig, self.canvas = make_canvas(right)

        ttk.Label(parent, text=t["label_comparaison"]).pack(anchor="w", padx=6)
        widths = (35, 60, 80, 60, 50, 50, 60, 70, 55, 50, 75, 75, 75, 60)
        self.runs_tree = make_tree(parent, t["cols_runs"], widths, height=5)
        self.runs_tree.pack(fill="x", padx=6, pady=(0, 6))

    def clear_runs(self):
        self.app.runs.clear()
        self.refresh()

    def run(self):
        if not self.app.check_idle():
            return
        words = tokenize(self.app.sentence_var.get())
        if not words:
            return
        tv = self.app.tv
        ys, self.steps = greedy_translate(self.app.model, self.app.sv, tv, words)
        trad = detokenize([tv.itos[i] for i in ys])
        self.app.fr_var.set(trad)
        unk = [w for w in words if w not in self.app.sv.stoi]
        put_text(self.out, trad + (tx.DIAG["inconnus"].format(mots=", ".join(unk)) if unk else ""))
        self.tree.delete(*self.tree.get_children())
        for k, (i, p, alts) in enumerate(self.steps):
            self.tree.insert("", "end", values=(k + 1, tv.itos[i], f"{p:.3f}",
                                                ", ".join(f"{tv.itos[j]} {q:.2f}" for j, q in alts)))
        self.refresh()

    def refresh(self):
        t, c = tx.DIAG, tx.COMMUN
        app = self.app
        tv = app.tv
        f = self.fig
        f.clear()

        ax1 = f.add_subplot(2, 2, 1)
        if self.steps:
            ax1.bar(range(len(self.steps)), [p for _, p, _ in self.steps], color="#6fdc6f")
            ax1.set_xticks(range(len(self.steps)))
            ax1.set_xticklabels([tv.itos[i] for i, _, _ in self.steps], rotation=45, ha="right")
            ax1.set_ylim(0, 1)
        ax1.set_title(t["plot_proba"])

        ax2 = f.add_subplot(2, 2, 2)
        if app.losses:
            ax2.plot(range(1, len(app.losses) + 1), app.losses, marker=".", label=c["loss"])
            ax2b = ax2.twinx()
            ax2b.plot(range(1, len(app.accs) + 1), app.accs, color="#ffa64f", label=c["acc_tokens"])
            ax2b.set_ylim(0, 1)
            ax2.legend(loc="upper center", fontsize=6)
            ax2b.legend(loc="center right", fontsize=6)
        ax2.set_title(t["plot_loss"])
        ax2.set_xlabel(c["epoch"])

        ax3 = f.add_subplot(2, 2, 3)
        for r in app.runs:
            ax3.plot(range(1, len(r["losses"]) + 1), r["losses"], label=f"#{r['id']}")
        if app.runs:
            ax3.legend(fontsize=6)
        ax3.set_title(t["plot_cmp_loss"])
        ax3.set_xlabel(c["epoch"])

        ax4 = f.add_subplot(2, 2, 4)
        if app.runs:
            x = np.arange(len(app.runs))
            ax4.bar(x - 0.2, [r["acc"] for r in app.runs], 0.4, label=c["acc_tokens"])
            ax4.bar(x + 0.2, [r["exact"] for r in app.runs], 0.4, label=t["exact_match"])
            ax4.set_xticks(x)
            ax4.set_xticklabels([f"#{r['id']}" for r in app.runs])
            ax4.set_ylim(0, 1)
            ax4.legend(fontsize=6)
        ax4.set_title(t["plot_cmp_acc"])
        finish(f, self.canvas)

        # tableau récapitulatif des configurations déjà entraînées
        self.runs_tree.delete(*self.runs_tree.get_children())
        for r in app.runs:
            cfg = r["cfg"]
            self.runs_tree.insert("", "end", values=(
                r["id"], cfg["d_model"], tx.PE_LABELS[cfg["pe"]], f"{cfg['n_enc']}/{cfg['n_dec']}",
                cfg["n_heads"], cfg["d_ff"], cfg["dropout"], cfg["lr"], len(r["losses"]), cfg["batch"],
                f"{r['losses'][-1]:.4f}" if r["losses"] else "-",
                f"{r['acc']:.3f}", f"{r['exact']:.3f}", f"{r['time']:.1f}s"))
